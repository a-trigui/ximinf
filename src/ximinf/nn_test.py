import jax
import jax.numpy as jnp
import jax.scipy as jsp
import ximinf.nn_inference as nninf 
import ximinf.nn_train as nntr
import numpy as np


def evaluate_on_test(models, test_sets, param_groups, cfg, gpu):
    """Prints the test accuracy of each group and returns them as a list of floats."""
    accuracies = []
    print("Test accuracy per group")
    print("-" * 40)
    for g, (model, (x, y), group) in enumerate(zip(models, test_sets, param_groups)):
        model.eval()
        _, acc = nntr.run_epoch(model, x, y, cfg["batch_size"], gpu)
        acc = float(acc)
        accuracies.append(acc)
        name = group if isinstance(group, str) else "+".join(group)
        print(f"Group {g} ({name}): {acc:.2%}")
    return accuracies

def sample_reference_point(rng_key, priors, param_names):
    """
    Sample a parameter-space reference point uniformly from the prior ranges.

    Each parameter is sampled independently from a uniform distribution
    between the lower and upper bounds specified in ``priors``. The function
    also returns an updated JAX random key for subsequent random-number
    generation.

    Parameters
    ----------
    rng_key : jax.Array
        JAX PRNG key used to generate the random sample.

    priors : dict
        Dictionary containing the prior definition for each parameter.
        For every parameter in ``param_names``, ``priors[name]["range"]``
        must contain the lower and upper bounds of the prior interval.

    param_names : sequence of str
        Names of the parameters to sample. The ordering determines the
        ordering of the returned parameter vector.

    Returns
    -------
    rng_key : jax.Array
        Updated JAX PRNG key.

    theta : jax.Array
        One-dimensional array containing the sampled parameter values,
        ordered according to ``param_names``.

    Notes
    -----
    The sampling is uniform in the parameterization defined by the prior
    ranges. No normalization using ``param_stats`` is applied.
    """
    rng_key, subkey = jax.random.split(rng_key)

    param_names = list(param_names)

    lows = jnp.array([priors[name]["range"][0] for name in param_names])
    highs = jnp.array([priors[name]["range"][1] for name in param_names])

    u = jax.random.uniform(subkey, shape=(len(param_names),))
    theta = lows + u * (highs - lows)

    return rng_key, theta

def one_sample_step_groups(
    rng_key,
    xi,
    theta_star,
    priors,
    param_names,
    models_per_group,
    visible_indices,
    group_indices,
    group_names_list,
    param_stats,
    n_warmup,
    n_samples,
):
    
    """
    Perform posterior sampling for one simulated observation and compute
    its TARP rank statistic.

    A reference point is first sampled uniformly from the prior parameter
    ranges. The posterior distribution conditioned on ``xi`` is then
    sampled using the group-specific neural-network inference models.
    The TARP statistic is computed as the fraction of posterior samples
    lying closer to the reference point than the true parameter
    ``theta_star``.

    Parameters
    ----------
    rng_key : jax.Array
        JAX PRNG key used for reference-point generation and posterior
        sampling.

    xi : jax.Array
        Observed or simulated data conditioned upon when evaluating the
        posterior.

    theta_star : jax.Array
        True parameter values associated with ``xi``.

    priors : dict
        Prior definitions for the model parameters.

    param_names : sequence of str
        Names of the inferred parameters. The ordering must be consistent
        with the parameter vectors used by the inference models.

    models_per_group : sequence
        Collection of trained neural-network inference models, with one
        model associated with each parameter group.

    visible_indices : sequence
        Indices specifying which parameters are visible to the corresponding
        group-specific inference models.

    group_indices : sequence
        Indices defining the parameter grouping used by the inference
        models.

    group_names_list : sequence
        Names identifying the parameter groups.

    param_stats : dict
        Statistics used to transform posterior samples back to the original
        parameterization. For each parameter, ``param_stats[name]`` must
        contain ``"mu"`` and ``"sigma"``.

    n_warmup : int
        Number of warm-up iterations used by the posterior sampler.

    n_samples : int
        Number of posterior samples to draw after warm-up.

    Returns
    -------
    f_val : jax.Array
        TARP rank statistic, defined as the fraction of posterior samples
        whose Euclidean distance from the sampled reference point is smaller
        than the distance between ``theta_star`` and the same reference
        point.

    posterior_unnormed : jax.Array
        Posterior samples transformed from the normalized parameterization
        back to the original parameterization using ``param_stats``.
    """
    rng_key, key_r0, key_mcmc = jax.random.split(rng_key, 3)

    _, theta_r0 = sample_reference_point(key_r0, priors, param_names)

    def log_post(theta):
        return nninf.log_prob_fn_groups(
            theta,
            models_per_group,
            xi,
            priors,
            visible_indices,
            group_indices,
            group_names_list,
        )

    rng_key, posterior = nninf.sample_posterior(
        log_post, n_warmup, n_samples, theta_star, key_mcmc
    )

    mus = jnp.array([param_stats[name]["mu"] for name in param_names])
    sigmas = jnp.array([param_stats[name]["sigma"] for name in param_names])
    
    posterior_unnormed = posterior * sigmas + mus
    # theta_star_unnormed = theta_star * sigmas + mus
    # theta_r0_unnormed = theta_r0 * sigmas + mus

    # d_star = jnp.linalg.norm(theta_star_unnormed - theta_r0_unnormed)
    # d_samples = jnp.linalg.norm(posterior_unnormed - theta_r0_unnormed, axis=1)

    d_star = jnp.linalg.norm(theta_star - theta_r0)
    d_samples = jnp.linalg.norm(posterior - theta_r0, axis=1)

    f_val = jnp.mean(d_samples < d_star)

    return f_val, posterior_unnormed


def compute_ecp_tarp_groups(
    models_per_group,
    x_list,
    theta_star_list,
    alpha_list,
    priors,
    param_names,
    visible_indices,
    group_indices,
    group_names_list,
    param_stats,
    n_warmup,
    n_samples,
    rng_key,
):
    """
    Compute empirical coverage probabilities using TARP for grouped models.

    The function evaluates the TARP statistic for each simulated observation
    and its corresponding true parameter value using ``jax.lax.scan``.
    Empirical coverage probabilities are then computed for each requested
    credibility level.

    Parameters
    ----------
    models_per_group : sequence
        Collection of trained neural-network inference models, with one
        model associated with each parameter group.

    x_list : jax.Array
        Collection of simulated observations used to evaluate posterior
        coverage.

    theta_star_list : jax.Array
        True parameter values corresponding to the observations in
        ``x_list``.

    alpha_list : sequence of float
        Significance levels at which empirical coverage is evaluated.
        For each ``alpha``, the function computes the fraction of TARP
        statistics satisfying ``f < 1 - alpha``.

    priors : dict
        Prior definitions for the inferred parameters.

    param_names : sequence of str
        Names of the inferred parameters.

    visible_indices : sequence
        Indices specifying the parameters visible to each group-specific
        inference model.

    group_indices : sequence
        Indices defining the parameter grouping used by the inference
        models.

    group_names_list : sequence
        Names identifying the parameter groups.

    param_stats : dict
        Statistics used to transform posterior samples from the normalized
        parameterization to the original parameterization. Each parameter
        must have ``"mu"`` and ``"sigma"`` entries.

    n_warmup : int
        Number of warm-up iterations used for each posterior sample.

    n_samples : int
        Number of posterior samples generated for each observation.

    rng_key : jax.Array
        JAX PRNG key used for all random sampling operations.

    Returns
    -------
    ecp_vals : list of jax.Array
        Empirical coverage probabilities corresponding to the values in
        ``alpha_list``. Each value is the fraction of TARP statistics
        satisfying ``f < 1 - alpha``.

    f_vals : jax.Array
        TARP statistic for each observation in ``x_list``.

    posteriors : jax.Array
        Posterior samples obtained for each observation.

    rng_key : jax.Array
        Updated JAX PRNG key after all sampling operations.
    """

    def scan_step(rng_key, xi_theta):
        xi, theta_star = xi_theta
        rng_key, subkey = jax.random.split(rng_key)

        f_val, posterior = one_sample_step_groups(
            subkey,
            xi,
            theta_star,
            priors,
            param_names,
            models_per_group,
            visible_indices,
            group_indices,
            group_names_list,
            param_stats,
            n_warmup,
            n_samples,
        )

        return rng_key, (f_val, posterior)
    
    rng_key, (f_vals, posteriors) = jax.lax.scan(
        scan_step, rng_key, (x_list, theta_star_list)
    )

    ecp_vals = [jnp.mean(f_vals < (1.0 - alpha)) for alpha in alpha_list]

    return ecp_vals, f_vals, posteriors, rng_key

def autocorr_fft(x, max_lag):
    """
    Compute the autocorrelation function of a one-dimensional sequence using
    an FFT-based estimator.

    The input sequence is first centered by subtracting its mean. The
    autocorrelation is then computed efficiently in Fourier space using
    zero-padding to avoid circular convolution effects. The resulting
    autocorrelation is normalized by its zero-lag value such that the
    autocorrelation at lag zero is equal to one.

    Parameters
    ----------
    x : jax.Array
        One-dimensional input sequence for which the autocorrelation is
        estimated.
    max_lag : int
        Number of non-negative lags to return. The returned array contains
        lags from 0 to ``max_lag - 1``.

    Returns
    -------
    jax.Array
        Normalized autocorrelation function for the non-negative lags
        ``0, ..., max_lag - 1``. The value at lag zero is equal to one.

    Notes
    -----
    The estimator is biased and energy-normalized. The input is centered
    before computing the autocorrelation, and the sequence is zero-padded
    to twice its original length before applying the FFT in order to avoid
    circular convolution effects.
    """

    n = x.shape[0]

    # Remove mean (important for statistical correctness)
    x = x - jnp.mean(x)

    # Zero-padding to avoid circular convolution effects
    nfft = 2 * n

    # FFT
    X = jnp.fft.rfft(x, n=nfft)

    # Power spectrum
    S = X * jnp.conj(X)

    # Inverse FFT -> autocorrelation
    acf_full = jnp.fft.irfft(S, n=nfft)

    # Normalize by zero-lag value
    acf_full = acf_full / acf_full[0]

    # Keep only non-negative lags
    return acf_full[:max_lag]

# ---------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------
def set_eval(models):
    """Disable dropout etc. on every group model."""
    for m in models:
        m.eval()


def group_names_as_lists(param_groups):
    """['a', ['b', 'c']] -> [['a'], ['b', 'c']]"""
    return [[g] if isinstance(g, str) else list(g) for g in param_groups]


def normalize_priors(priors, param_stats):
    out = {}
    for name, prior in priors.items():
        mu, sigma = param_stats[name]["mu"], param_stats[name]["sigma"]
        out[name] = {"range": (prior["range"] - mu) / sigma, "type": prior["type"]}
    return out


def unnormalize_array(arr, names, param_stats, dh):
    """
    arr: (..., D) normalised values, columns ordered as `names`.
    Works for a single vector (D,) or samples (n, D).
    """
    as_dict = {name: arr[..., i] for i, name in enumerate(names)}
    un = dh.unnormalize(as_dict, param_stats)
    return jnp.stack([un[name] for name in names], axis=-1)


# ---------------------------------------------------------------
# 2. Pick test samples whose label is "true" (last group)
# ---------------------------------------------------------------
def select_true_test_samples(test_sets, n_params, n_max=100):
    """
    Uses the last group's test set, where all parameters are visible.
    x columns are [data, mask, m_norm, params], so the last n_params columns
    are theta and everything before them is the base input.

    Returns:
        theta_star : (n, n_params) parameters of the true samples
        xy_test    : (n, F) inputs without theta (data + mask + m_norm)
    """
    x_last, y_last = test_sets[-1]

    mask_true = y_last[:, 0] == 1
    n_sims = int(jnp.minimum(n_max, jnp.sum(mask_true)))
    true_idx = jnp.nonzero(mask_true, size=n_sims, fill_value=0)[0]

    theta_star = x_last[true_idx, -n_params:]
    xy_test = x_last[true_idx, :-n_params]
    return theta_star, xy_test


# ---------------------------------------------------------------
# 3. Posterior
# ---------------------------------------------------------------
def make_log_post(models, test_data, norm_priors, visible_indices,
                  group_indices, group_names_list, nninf):
    def log_post(theta):
        return nninf.log_prob_fn_groups(
            theta, models, test_data, norm_priors,
            visible_indices, group_indices, group_names_list,
        )
    return log_post


def sample_posterior_for_index(index, theta_star, xy_test, models, param_groups,
                               global_param_names, priors, param_stats, key,
                               gpu, dh, nninf, n_warmup=200, n_samples=2000):
    """
    Runs MCMC for test sample `index`.
    Returns (key, post_unnormed, theta_star_unnormed).
    """
    theta_star_unnormed = unnormalize_array(
        theta_star[index], global_param_names, param_stats, dh
    )

    group_names_list = group_names_as_lists(param_groups)
    visible_indices, group_indices = nninf.preprocess_groups(param_groups, global_param_names)
    norm_priors = normalize_priors(priors, param_stats)

    test_data = jax.device_put(xy_test[index], gpu)
    theta_init = jax.device_put(theta_star[index], gpu)   # CHANGED: was theta_star[index]

    log_post = make_log_post(models, test_data, norm_priors, visible_indices,
                             group_indices, group_names_list, nninf)

    print("Launch MCMC ...")
    key, post = nninf.sample_posterior(
        log_post,
        n_warmup=n_warmup,
        n_samples=n_samples,
        init_position=theta_init,
        rng_key=key,
    )
    print("...finished")

    post_unnormed = unnormalize_array(post, global_param_names, param_stats, dh)
    return key, post_unnormed, theta_star_unnormed


def prior_ranges(priors, names):
    return [(float(priors[n]["range"][0]), float(priors[n]["range"][1])) for n in names]
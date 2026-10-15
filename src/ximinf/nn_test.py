import jax
import jax.numpy as jnp
import ximinf.nn_inference as nninf 
import ximinf.nn_train as nntr
import ximinf.data_helper as dh
import numpy as np

def evaluate_on_test(models, test_sets, param_groups, cfg, gpu):
    """Prints the test accuracy of each group and returns them as a list of floats."""
    accuracies = []
    print("Test accuracy per group")
    print("-" * 40)
    for g, (model, (x, y), group) in enumerate(zip(models, test_sets, param_groups)):
        model.eval()
        _, acc = nntr.run_epoch(model, x, y, cfg["batch_size"], gpu, pos_weight=1)
        acc = float(acc)
        accuracies.append(acc)
        name = dh.format_groups([group])
        print(f"Group {g} ({name}): {acc:.2%}")
    return accuracies

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

def fit_cosmologix_datasets(
    list_of_dfs, fitter, fiducial_cosmo,
    param_names, params_of_interest,
    build_likelihood,
):
    """
    Fit each simulated dataset and extract the Gaussian posterior approximation.

    Parameters
    ----------
    build_likelihood : callable
        Function taking a DataFrame and returning a likelihood object.

    Returns
    -------
    mu_list  : array (Nsims, ndim_interest)
    cov_list : array (Nsims, ndim_interest, ndim_interest)
    """
    indices = [param_names.index(p) for p in params_of_interest]
    Nsims = len(list_of_dfs)
    ndim = len(params_of_interest)

    mu_list = np.zeros((Nsims, ndim))
    cov_list = np.zeros((Nsims, ndim, ndim))

    for i, df in enumerate(list_of_dfs):
        print(f'Iteration {i+1}/{Nsims}', end='\r')

        likelihood = build_likelihood(df)

        result = fitter.fit([likelihood], fixed=fiducial_cosmo)

        mu = np.array(result["x"])
        cov = np.array(result["inverse_FIM"])

        mu_list[i] = mu[indices]
        cov_list[i] = cov[np.ix_(indices, indices)]

    return mu_list, cov_list

def sample_from_gaussians(mu_list, cov_list, n_posterior_samples=2000, rng=None):
    """
    Draw posterior samples from Gaussian approximations.

    Parameters
    ----------
    mu_list : array (Nsims, ndim)
        Posterior means.
    cov_list : array (Nsims, ndim, ndim)
        Posterior covariance matrices.
    n_posterior_samples : int
        Number of posterior samples per simulation.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    samples : array (Nsims, n_posterior_samples, ndim)
        Posterior samples for each simulation.
    """
    rng = rng or np.random.default_rng()
    Nsims, ndim = mu_list.shape

    samples = np.zeros((Nsims, n_posterior_samples, ndim))

    for i in range(Nsims):
        samples[i] = rng.multivariate_normal(
            mu_list[i], cov_list[i], size=n_posterior_samples
        )

    return samples

# =====================================================================
# 1. Parameter bookkeeping (used identically by both pipelines)
# =====================================================================
def prior_bounds(priors, names):
    """(low, high) arrays, ordered like ``names``, from ``priors[n]['range']``."""
    low = np.array([float(priors[n]["range"][0]) for n in names])
    high = np.array([float(priors[n]["range"][1]) for n in names])
    return low, high


def stack_thetas(params, names):
    """dict of per-sim arrays -> (N_sims, ndim) array ordered like ``names``."""
    return np.column_stack([np.asarray(params[n]) for n in names])

# =====================================================================
# 2. TARP core -- backend independent, shared by both pipelines
# =====================================================================
def sample_reference_points(rng, prior_low, prior_high, n):
    """
    Draw ``n`` TARP reference points uniformly in the prior box.

    ``rng`` can be a seed or a ``numpy.random.Generator``. Re-using the same
    reference points for several methods gives a paired comparison: the
    difference between the ECP curves then comes from the posteriors only.
    """
    rng = np.random.default_rng(rng)
    return rng.uniform(prior_low, prior_high, size=(n, len(prior_low)))


def tarp_f_values(posterior_samples, true_thetas, theta_ref, prior_low, prior_high):
    """
    TARP statistic f_i = fraction of posterior samples closer to the
    reference point than the truth. Distances are measured in units of the
    prior width, so no parameter dominates and the metric does not depend
    on how the parameters happen to be normalized.

    posterior_samples : (N, S, ndim); true_thetas, theta_ref : (N, ndim)
    """
    width = np.asarray(prior_high) - np.asarray(prior_low)
    post = np.asarray(posterior_samples)
    d_samples = np.linalg.norm((post - theta_ref[:, None, :]) / width, axis=-1)
    d_true = np.linalg.norm((np.asarray(true_thetas) - theta_ref) / width, axis=-1)
    return np.mean(d_samples < d_true[:, None], axis=1)


def ecp_from_f(f_values, alphas):
    """Empirical coverage: ECP(alpha) = mean(f < 1 - alpha)."""
    f_values = np.asarray(f_values)
    alphas = np.asarray(alphas)
    return np.mean(f_values[None, :] < (1.0 - alphas)[:, None], axis=1)


def run_tarp_from_samples(
    posterior_samples, true_thetas, prior_low, prior_high,
    n_alpha=31, rng=None, theta_ref=None,
):
    """
    Full TARP from posterior samples. Same signature/returns as before
    (``alphas, ecp, f_values``), plus an optional ``theta_ref`` so that
    several methods can share the same reference points.
    """
    prior_low, prior_high = np.asarray(prior_low), np.asarray(prior_high)
    if theta_ref is None:
        theta_ref = sample_reference_points(
            rng, prior_low, prior_high, len(posterior_samples)
        )
    f_values = tarp_f_values(
        posterior_samples, true_thetas, theta_ref, prior_low, prior_high
    )
    alphas = np.linspace(0.0, 1.0, n_alpha)
    return alphas, ecp_from_f(f_values, alphas), f_values


def tarp_ks_pvalue(f_values):
    """
    KS test of the f-values against U(0,1). This is the quantity that is
    uniform for a calibrated posterior; the ECP curve is just a transform
    of it, so testing the f-values directly is cleaner than testing the
    ECP values.
    """
    from scipy.stats import kstest
    return kstest(np.asarray(f_values), "uniform").pvalue


def samples_to_mean_cov(posterior_samples):
    """
    (N, S, ndim) samples -> (mu_list, cov_list), the same objects that
    ``fit_cosmologix_datasets`` returns. Lets the pull plot / any Gaussian
    diagnostic run unchanged on the NN posteriors.
    """
    s = np.asarray(posterior_samples)
    mu = s.mean(axis=1)
    c = s - mu[:, None, :]
    cov = np.einsum("nsi,nsj->nij", c, c) / (s.shape[1] - 1)
    return mu, cov


# =====================================================================
# 3. Samplers -- the only method-specific part
# =====================================================================
# cosmologix: keep fit_cosmologix_datasets and sample_from_gaussians as is.

def make_log_post(models, xi, norm_priors, visible_indices,
                  group_indices, group_names_list):
    def log_post(theta):
        return nninf.log_prob_fn_groups(
            theta, models, xi, norm_priors,
            visible_indices, group_indices, group_names_list,
        )
    return log_post


def sample_nn_posteriors(
    models_per_group, x_list, theta_init_list, norm_priors, param_names,
    visible_indices, group_indices, group_names_list, param_stats,
    n_warmup, n_samples, rng_key,
):
    """
    MCMC-sample the NN posterior for every observation.

    Returns
    -------
    posteriors : (N, n_samples, ndim) array in PHYSICAL units
    rng_key    : updated key

    ``theta_init_list`` (normalized, (N, ndim)) is only the chain starting
    point. No TARP logic in here.
    """
    def scan_step(rng_key, xi_theta):
        xi, theta_init = xi_theta
        rng_key, subkey = jax.random.split(rng_key)
        log_post = make_log_post(
            models_per_group, xi, norm_priors,
            visible_indices, group_indices, group_names_list,
        )
        _, post = nninf.sample_posterior(
            log_post, n_warmup, n_samples, theta_init, subkey
        )
        return rng_key, post

    rng_key, post_norm = jax.lax.scan(scan_step, rng_key, (x_list, theta_init_list))
    return dh.unnormalize(post_norm, param_names, param_stats), rng_key
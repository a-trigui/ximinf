import jax
import jax.numpy as jnp
import ximinf.nn_inference as nninf 
import ximinf.nn_train as nntr
import ximinf.data_helper as dh
import numpy as np

def evaluate_on_test(models, test_sets, param_groups, cfg, gpu):
    """
    Evaluate and print classification accuracy on test sets across all groups.

    Parameters
    ----------
    models : list of nnx.Module
        Trained neural network models for each parameter group.
    test_sets : list of tuple
        List of ``(x, y)`` feature and label pairs for each group.
    param_groups : list
        List of parameter groups defining the model structure.
    cfg : dict
        Configuration dictionary containing ``"batch_size"``.
    gpu : jax.Device
        Device on which evaluation is performed.

    Returns
    -------
    list of float
        Test accuracy achieved for each parameter group.
    """
    accuracies = []
    print("Test accuracy per group")
    print("-" * 40)
    for g, (model, (x, y), group) in enumerate(zip(models, test_sets, param_groups)):
        model.eval()
        _, acc = nntr.run_epoch(model, x, y, cfg["batch_size"], gpu)
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
    """
    Set all group neural network models to evaluation mode.

    Parameters
    ----------
    models : list of nnx.Module
        Collection of neural network models to place in evaluation mode.
    """
    for m in models:
        m.eval()

# ---------------------------------------------------------------
# 2. Pick test samples whose label is "true" (last group)
# ---------------------------------------------------------------
def select_true_test_samples(test_sets, n_params, n_max=100):
    """
    Extract true positive samples and separate features from parameters.

    Uses the final group's test set where all parameters are visible as the
    last ``n_params`` columns of the input tensor.

    Parameters
    ----------
    test_sets : list of tuple
        List of ``(x, y)`` feature and label pairs for each group.
    n_params : int
        Number of parameters positioned at the end of the input features.
    n_max : int, optional
        Maximum number of positive samples to select. Default is 100.

    Returns
    -------
    theta_star : jax.Array
        True parameter values for the selected samples with shape
        ``(n_selected, n_params)``.
    xy_test : jax.Array
        Observation features excluding parameters with shape
        ``(n_selected, F)``.
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
    list_of_dfs : list of pandas.DataFrame
        Simulated supernova observation datasets to fit.
    fitter : object
        Fitter instance providing a ``fit`` method.
    fiducial_cosmo : dict
        Fixed fiducial cosmology dictionary passed to the fitter.
    param_names : list of str
        Complete ordered list of all model parameter names.
    params_of_interest : list of str
        Subset of parameter names to extract from the fit results.
    build_likelihood : callable
        Function taking a dataset DataFrame and returning a likelihood object.

    Returns
    -------
    mu_list : numpy.ndarray
        Posterior means for the parameters of interest with shape
        ``(Nsims, ndim_interest)``.
    cov_list : numpy.ndarray
        Posterior covariance matrices for the parameters of interest with
        shape ``(Nsims, ndim_interest, ndim_interest)``.
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
    """
    Extract lower and upper prior bounds for an ordered sequence of parameters.

    Parameters
    ----------
    priors : dict
        Dictionary mapping parameter names to prior specifications containing
        a ``"range"`` key.
    names : sequence of str
        Sequence of parameter names defining the order of the output bounds.

    Returns
    -------
    low : numpy.ndarray
        1D array of lower prior bounds.
    high : numpy.ndarray
        1D array of upper prior bounds.
    """
    low = np.array([float(priors[n]["range"][0]) for n in names])
    high = np.array([float(priors[n]["range"][1]) for n in names])
    return low, high


def stack_thetas(params, names):
    """
    Stack per-simulation parameter arrays into a 2D parameter matrix.

    Parameters
    ----------
    params : dict
        Dictionary mapping parameter names to 1D arrays of sampled values.
    names : sequence of str
        Sequence of parameter names defining the column order.

    Returns
    -------
    numpy.ndarray
        2D array of shape ``(N_sims, len(names))`` containing stacked parameter values.
    """
    return np.column_stack([np.asarray(params[n]) for n in names])

# =====================================================================
# 2. TARP core -- backend independent, shared by both pipelines
# =====================================================================
def sample_reference_points(rng, prior_low, prior_high, n):
    """
    Draw TARP reference points uniformly distributed within the prior box.

    Parameters
    ----------
    rng : int, numpy.random.Generator, or None
        Random number generator or seed.
    prior_low : array-like
        Lower bounds for each parameter dimension.
    prior_high : array-like
        Upper bounds for each parameter dimension.
    n : int
        Number of reference points to sample.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, len(prior_low))`` containing the sampled reference points.
    """
    rng = np.random.default_rng(rng)
    return rng.uniform(prior_low, prior_high, size=(n, len(prior_low)))


def tarp_f_values(posterior_samples, true_thetas, theta_ref, prior_low, prior_high):
    """
    Compute TARP statistic values across simulated datasets.

    Calculates the fraction of posterior samples closer to the reference point
    than the true parameter value, normalizing parameter distances by the
    prior range width.

    Parameters
    ----------
    posterior_samples : array-like
        Posterior samples with shape ``(N, S, ndim)`` for ``N`` simulations
        and ``S`` samples per simulation.
    true_thetas : array-like
        True parameter values with shape ``(N, ndim)``.
    theta_ref : array-like
        Reference points with shape ``(N, ndim)``.
    prior_low : array-like
        Lower bounds of parameter priors.
    prior_high : array-like
        Upper bounds of parameter priors.

    Returns
    -------
    numpy.ndarray
        1D array of length ``N`` containing the TARP statistic for each simulation.
    """
    width = np.asarray(prior_high) - np.asarray(prior_low)
    post = np.asarray(posterior_samples)
    d_samples = np.linalg.norm((post - theta_ref[:, None, :]) / width, axis=-1)
    d_true = np.linalg.norm((np.asarray(true_thetas) - theta_ref) / width, axis=-1)
    return np.mean(d_samples < d_true[:, None], axis=1)


def ecp_from_f(f_values, alphas):
    """
    Compute Empirical Coverage Probability (ECP) curve from TARP f-values.

    Parameters
    ----------
    f_values : array-like
        1D array of TARP test statistic values.
    alphas : array-like
        Nominal credibility levels at which to evaluate coverage.

    Returns
    -------
    numpy.ndarray
        Empirical coverage probabilities corresponding to each level in ``alphas``.
    """
    f_values = np.asarray(f_values)
    alphas = np.asarray(alphas)
    return np.mean(f_values[None, :] < (1.0 - alphas)[:, None], axis=1)


def run_tarp_from_samples(
    posterior_samples, true_thetas, prior_low, prior_high,
    n_alpha=31, rng=None, theta_ref=None,
):
    """
    Run full TARP coverage evaluation from posterior sample distributions.

    Parameters
    ----------
    posterior_samples : array-like
        Posterior samples with shape ``(N, S, ndim)``.
    true_thetas : array-like
        True parameters with shape ``(N, ndim)``.
    prior_low : array-like
        Lower bounds of parameter priors.
    prior_high : array-like
        Upper bounds of parameter priors.
    n_alpha : int, optional
        Number of credibility grid points between 0 and 1. Default is 31.
    rng : int, numpy.random.Generator, or None, optional
        Random generator used if ``theta_ref`` is not provided.
    theta_ref : array-like, optional
        Precomputed reference points with shape ``(N, ndim)``. If ``None``,
        points are sampled uniformly in the prior box.

    Returns
    -------
    alphas : numpy.ndarray
        Credibility levels grid from 0 to 1.
    ecp : numpy.ndarray
        Empirical coverage probabilities evaluated at ``alphas``.
    f_values : numpy.ndarray
        TARP statistic values for each simulation.
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
    Compute Kolmogorov-Smirnov test p-value comparing TARP f-values against U(0, 1).

    Parameters
    ----------
    f_values : array-like
        1D array of TARP test statistic values across simulations.

    Returns
    -------
    float
        p-value of the two-sided Kolmogorov-Smirnov test.
    """
    from scipy.stats import kstest
    return kstest(np.asarray(f_values), "uniform").pvalue


def samples_to_mean_cov(posterior_samples):
    """
    Compute posterior mean vectors and covariance matrices from MCMC samples.

    Parameters
    ----------
    posterior_samples : array-like
        Posterior samples of shape ``(N, S, ndim)`` for ``N`` simulations
        and ``S`` samples per simulation.

    Returns
    -------
    mu : numpy.ndarray
        Sample mean vectors with shape ``(N, ndim)``.
    cov : numpy.ndarray
        Sample covariance matrices with shape ``(N, ndim, ndim)``.
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
    """
    Construct a closure that computes the joint autoregressive log-posterior.

    Parameters
    ----------
    models : sequence of callable
        Neural network models for each parameter group.
    xi : jax.Array
        Observation features tensor.
    norm_priors : dict
        Prior specifications for normalized parameters.
    visible_indices : sequence of array-like
        Visible parameter indices per group.
    group_indices : sequence of array-like
        Target parameter indices per group.
    group_names_list : sequence of list
        Parameter names per group.

    Returns
    -------
    callable
        Function taking parameter vector ``theta`` and returning the scalar
        log-probability.
    """
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
    Draw MCMC posterior samples using the neural likelihood ratio estimators.

    Samples the joint posterior for each observation using NUTS window adaptation
    and unnormalizes the resulting chains back to physical parameter units.

    Parameters
    ----------
    models_per_group : sequence of callable
        Neural network models for each autoregressive group.
    x_list : jax.Array
        Batch of observation features of shape ``(N, F)``.
    theta_init_list : jax.Array
        Normalized initial parameter positions for each chain of shape
        ``(N, ndim)``.
    norm_priors : dict
        Prior specifications for normalized parameters.
    param_names : list of str
        Global ordered parameter names.
    visible_indices : sequence of array-like
        Visible parameter indices for each group.
    group_indices : sequence of array-like
        Target parameter indices for each group.
    group_names_list : sequence of list
        Names of parameters in each group.
    param_stats : dict
        Mean and standard deviation normalization statistics for each
        parameter.
    n_warmup : int
        Number of warmup iterations for NUTS adaptation.
    n_samples : int
        Number of posterior samples to draw after warmup.
    rng_key : jax.Array
        JAX PRNG key.

    Returns
    -------
    posteriors : jax.Array
        Unnormalized posterior samples in physical parameter units with shape
        ``(N, n_samples, ndim)``.
    rng_key : jax.Array
        Updated JAX PRNG key.
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
    return dh.unnormalize_array(post_norm, param_names, param_stats), rng_key
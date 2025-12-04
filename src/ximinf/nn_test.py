# Import libraries
import jax
import jax.numpy as jnp
import blackjax
from functools import partial
from tqdm.notebook import tqdm

def distance(theta1, theta2):
    """
    Compute the Euclidean distance between two points in NDIM space.

    Parameters
    ----------
    theta1 : array-like
        First point in NDIM-dimensional space.
    theta2 : array-like
        Second point in NDIM-dimensional space.

    Returns
    -------
    float
        The Euclidean distance between `theta1` and `theta2`.
    """
    diff = theta1 - theta2
    return jnp.linalg.norm(diff)

def log_prior(theta, bounds):
    """
    Compute the log-prior probability for the parameter `theta`, 
    assuming uniform prior within given bounds.

    Parameters
    ----------
    theta : array-like
        The parameter values for which the prior is to be calculated.
    bounds : jnp.ndarray, optional
        The bounds on each parameter (default is the global `BOUNDS`).

    Returns
    -------
    float
        The log-prior of `theta`, or negative infinity if `theta` is out of bounds.
    """

    in_bounds = jnp.all((theta >= bounds[:, 0]) & (theta <= bounds[:, 1]))
    return jnp.where(in_bounds, 0.0, -jnp.inf)

def log_prob_fn(theta, model, xy_noise, bounds):
    """
    Compute the log-probability for the parameter `theta` using a 
    log-prior and the log-likelihood from the neural likelihood ratio approximation.

    Parameters
    ----------
    theta : array-like
        The parameter values for which the log-probability is computed.
    model : callable
        A function that takes `theta` and produces model logits for computing the likelihood.
    xy_noise : array-like
        Input data with added noise for evaluating the likelihood.

    Returns
    -------
    float
        The log-probability, which is the sum of the log-prior and the log-likelihood.
    """

    lp = log_prior(theta, bounds)
    lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)
    xy_flat = xy_noise.squeeze()
    inp = jnp.concatenate([xy_flat, theta])[None, :]
    logits = model(inp)
    p = jax.nn.sigmoid(logits).squeeze()
    p = jnp.clip(p, 1e-6, 1 - 1e-6)
    log_like = jnp.log(p) - jnp.log1p(-p)
    return lp + log_like

def sample_reference_point(rng_key, bounds):
    """
    Sample a reference point within the given bounds uniformly.

    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        The random key used for sampling.
    bounds : jnp.ndarray, optional
        The bounds for each parameter (default is the global `BOUNDS`).

    Returns
    -------
    tuple
        A tuple containing the updated `rng_key` and the sampled reference point `theta`.
    """
    ndim = bounds.shape[0]
    rng_key, subkey = jax.random.split(rng_key)
    u = jax.random.uniform(subkey, shape=(ndim,))
    span = bounds[:, 1] - bounds[:, 0]
    theta = bounds[:, 0] + u * span
    return rng_key, theta

def inference_loop(rng_key, kernel, initial_state, num_samples):
    """
    Perform an inference loop using a Markov Chain Monte Carlo (MCMC) kernel.

    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        The random key used for sampling.
    kernel : callable
        The MCMC kernel (e.g., NUTS) used for updating the state.
    initial_state : object
        The initial state of the MCMC chain.
    num_samples : int
        The number of samples to generate in the chain.

    Returns
    -------
    jax.numpy.ndarray
        The sampled states from the inference loop.
    """

    def one_step(state, rng):
        state, _ = kernel(rng, state)
        return state, state
    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)
    return states

@partial(jax.jit, static_argnums=(0, 1, 2))
def sample_posterior(log_prob, n_warmup, n_samples, init_position, rng_key):
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (warmup_state, params), _ = warmup.run(warmup_key, init_position, num_steps=n_warmup)
    kernel = blackjax.nuts(log_prob, **params).step
    rng_key, sample_key = jax.random.split(rng_key)
    states = inference_loop(sample_key, kernel, warmup_state, n_samples)
    return rng_key, states.position


# ========== JIT‐compiled per‐sample step ==========
@partial(jax.jit, static_argnums=(3, 4, 5))
def one_sample_step(rng_key, xi, theta_star, n_warmup, n_samples, model, bounds):
    """
    Sample from the posterior distribution using Hamiltonian Monte Carlo (HMC) 
    with NUTS (No-U-Turn Sampler) for a given `log_prob`.

    Parameters
    ----------
    log_prob : callable
        The log-probability function for the model and parameters.
    n_warmup : int
        The number of warmup steps to adapt the sampler.
    n_samples : int
        The number of samples to generate after warmup.
    init_position : array-like
        The initial position for the chain (parameter values).
    rng_key : jax.random.PRNGKey
        The random key used for sampling.

    Returns
    -------
    jax.numpy.ndarray
        The sampled positions (parameters) from the posterior distribution.
    """

    # Draw a random reference
    rng_key, theta_r0 = sample_reference_point(rng_key, bounds)

    def log_post(theta):
        return log_prob_fn(theta, model, xi, bounds)

    # Run MCMC
    rng_key, posterior = sample_posterior(log_post, n_warmup, n_samples, theta_star, rng_key)

    # Compute e-c-p distances
    d_star = distance(theta_star, theta_r0)
    d_samples = jnp.linalg.norm(posterior - theta_r0, axis=1)
    f_val = jnp.mean(d_samples < d_star)

    return rng_key, f_val, posterior


def batched_one_sample_step(rng_keys, x_batch, theta_star_batch, n_warmup, n_samples, model, bounds):
    """
    Vectorized wrapper over `one_sample_step` using jax.vmap.

    Parameters
    ----------
    rng_keys : jax.random.PRNGKey
        Batch of random keys.
    x_batch : array-like
        Batch of input data.
    theta_star_batch : array-like
        Batch of true parameter values.
    n_warmup : int
        Number of warmup steps.
    n_samples : int
        Number of samples.
    model : callable
        The model function.
    bounds : array-like
        Parameter bounds.

    Returns
    -------
    tuple
        (rng_keys, f_vals, posterior_samples)
    """
    return jax.vmap(
        lambda rng, x, theta: one_sample_step(rng, x[None, :], theta, n_warmup, n_samples, model, bounds),
        in_axes=(0, 0, 0)
    )(rng_keys, x_batch, theta_star_batch)


def compute_ecp_tarp_jitted(model, x_list, theta_star_list, alpha_list, n_warmup, n_samples, rng_key, bounds):
    """
    Compute expected coverage probabilities (ECP) using vectorized sampling.

    Parameters
    ----------
    model : callable
        The model function.
    x_list : array-like
        List of input data.
    theta_star_list : array-like
        List of true parameter values.
    alpha_list : list of float
        List of alpha values for ECP computation.
    n_warmup : int
        Number of warmup steps.
    n_samples : int
        Number of samples.
    rng_key : jax.random.PRNGKey
        Random key.
    bounds : array-like
        Parameter bounds.

    Returns
    -------
    tuple
        (ecp_vals, f_vals, posterior_uns, rng_key)
    """
    N = x_list.shape[0]
    rng_key, split_key = jax.random.split(rng_key)
    rng_keys = jax.random.split(split_key, N)

    # Batched MCMC and distance evaluation
    _, f_vals, posterior_uns = batched_one_sample_step(
        rng_keys, x_list, theta_star_list, n_warmup, n_samples, model, bounds
    )

    # Compute ECP values for each alpha
    ecp_vals = [jnp.mean(f_vals < (1 - alpha)) for alpha in alpha_list]

    return ecp_vals, f_vals, posterior_uns, rng_key


def compute_ecp_tarp_jitted_with_progress(model, x_list, theta_star_list, alpha_list,
                                          n_warmup, n_samples, rng_key, bounds,
                                          batch_size=20):
    """
    Compute ECP using JITed MCMC in batches with progress reporting via tqdm.

    Parameters
    ----------
    model : callable
        The model function.
    x_list : array-like
        List of input data.
    theta_star_list : array-like
        List of true parameter values.
    alpha_list : list of float
        List of alpha values for ECP computation.
    n_warmup : int
        Number of warmup steps.
    n_samples : int
        Number of samples.
    rng_key : jax.random.PRNGKey
        Random key.
    bounds : array-like
        Parameter bounds.
    batch_size : int, optional
        Batch size for processing (default is 20).

    Returns
    -------
    tuple
        (ecp_vals, posterior_uns, rng_key)
    """
    N = x_list.shape[0]

    posterior_list = []
    f_vals_list = []

    for start in tqdm(range(0, N, batch_size), desc="Computing ECP batches"):
        end = min(start + batch_size, N)
        x_batch = x_list[start:end]
        theta_batch = theta_star_list[start:end]

        # Compute ECP and posterior for batch
        _, f_vals_batch, posterior_batch, rng_key = compute_ecp_tarp_jitted(
            model, x_batch, theta_batch, alpha_list,
            n_warmup, n_samples, rng_key, bounds
        )

        posterior_list.append(posterior_batch)
        f_vals_list.append(f_vals_batch)

    # Concatenate across batches
    posterior_uns = jnp.concatenate(posterior_list, axis=0)
    f_vals_all = jnp.concatenate(f_vals_list, axis=0)

    # Compute final ECP for each alpha
    ecp_vals = [jnp.mean(f_vals_all < (1 - alpha)) for alpha in alpha_list]

    return ecp_vals, posterior_uns, rng_key
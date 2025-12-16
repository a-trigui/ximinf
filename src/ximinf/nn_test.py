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

def log_prob_fn_groups(theta, models_per_group, x, bounds, param_groups, global_param_names):
    """
    Compute the sum of log-likelihoods for all groups given full theta.

    Parameters
    ----------
    theta : jnp.ndarray, shape (n_params,)
        Full parameter vector.
    models_per_group : list
        List of DeepSetClassifier models, one per group.
    x : jnp.ndarray
        Input data sample (shape: (data_features + ... + n_params))
    bounds : jnp.ndarray
        Parameter bounds.
    param_groups : list
        List of parameter groups.
    param_names : list
        List of all parameter names in order.

    Returns
    -------
    float
        Sum of log-likelihoods over all groups.
    """
    log_lik_sum = 0.0

    # Use everything except the last n_params entries as data
    data_part = x[:-len(theta)].reshape(1, -1)   # 2D
    # If mask is required, you can extract it similarly here

    for g, group in enumerate(param_groups):
        # Determine visible parameters for this group
        prev_groups = [
            p
            for i in range(g)
            for p in (param_groups[i] if isinstance(param_groups[i], list) else [param_groups[i]])
        ]
        group_list = [group] if isinstance(group, str) else group
        visible_param_names = prev_groups + group_list

        # Get visible theta values
        visible_idx = jnp.array([global_param_names.index(name) for name in visible_param_names])
        theta_visible = theta[visible_idx].reshape(1, -1)  # make 2D

        # Concatenate data with visible parameters
        input_g = jnp.concatenate([data_part, theta_visible], axis=-1)

        # Forward pass through the model
        logits = models_per_group[g](input_g)
        p = jax.nn.sigmoid(logits)

        log_lik_sum += jnp.log(p) + jnp.log(1 - p)

    return jnp.squeeze(log_lik_sum) + log_prior(theta, bounds)




@partial(jax.jit, static_argnums=(0, 1, 2))
def sample_posterior(log_prob, n_warmup, n_samples, init_position, rng_key):
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (warmup_state, params), _ = warmup.run(warmup_key, init_position, num_steps=n_warmup)
    kernel = blackjax.nuts(log_prob, **params).step
    rng_key, sample_key = jax.random.split(rng_key)
    states = inference_loop(sample_key, kernel, warmup_state, n_samples)
    return rng_key, states.position


def one_sample_step_groups(rng_key, xi, theta_star, n_warmup, n_samples, 
                           models_per_group, bounds, param_groups, param_names):
    """
    Sample from posterior using sum of log-likelihoods over all groups.
    """
    rng_key, theta_r0 = sample_reference_point(rng_key, bounds)

    def log_post(theta):
        return log_prob_fn_groups(theta, models_per_group, xi, bounds, param_groups, param_names)

    rng_key, posterior = sample_posterior(log_post, n_warmup, n_samples, theta_star, rng_key)
    d_star = distance(theta_star, theta_r0)
    d_samples = jnp.linalg.norm(posterior - theta_r0, axis=1)
    f_val = jnp.mean(d_samples < d_star)

    return rng_key, f_val, posterior


def batched_one_sample_step_groups(rng_keys, x_batch, theta_star_batch,
                                   n_warmup, n_samples, models_per_group, bounds, param_groups, param_names):
    return jax.vmap(
        lambda rng, x, theta: one_sample_step_groups(rng, x[None, :], theta, n_warmup, n_samples,
                                                     models_per_group, bounds, param_groups, param_names),
        in_axes=(0, 0, 0)
    )(rng_keys, x_batch, theta_star_batch)

def compute_ecp_tarp_jitted_groups(models_per_group, x_list, theta_star_list, alpha_list,
                                   n_warmup, n_samples, rng_key, bounds,
                                   param_groups, param_names):
    """
    Batched ECP computation using multiple group models.
    """
    N = x_list.shape[0]
    rng_key, split_key = jax.random.split(rng_key)
    rng_keys = jax.random.split(split_key, N)

    # Batched MCMC and distance evaluation
    _, f_vals, posterior_uns = batched_one_sample_step_groups(
        rng_keys, x_list, theta_star_list, n_warmup, n_samples,
        models_per_group, bounds, param_groups, param_names
    )

    # Compute ECP values for each alpha
    ecp_vals = [jnp.mean(f_vals < (1 - alpha)) for alpha in alpha_list]

    return ecp_vals, f_vals, posterior_uns, rng_key

def compute_ecp_tarp_jitted_with_progress_groups(models_per_group, x_list, theta_star_list, alpha_list,
                                                 n_warmup, n_samples, rng_key, bounds,
                                                 param_groups, param_names, batch_size=20):
    N = x_list.shape[0]

    posterior_list = []
    f_vals_list = []

    for start in tqdm(range(0, N, batch_size), desc="Computing ECP batches"):
        end = min(start + batch_size, N)
        x_batch = x_list[start:end]
        theta_batch = theta_star_list[start:end]

        # Compute ECP and posterior for batch
        _, f_vals_batch, posterior_batch, rng_key = compute_ecp_tarp_jitted_groups(
            models_per_group, x_batch, theta_batch, alpha_list,
            n_warmup, n_samples, rng_key, bounds,
            param_groups, param_names
        )

        posterior_list.append(posterior_batch)
        f_vals_list.append(f_vals_batch)

    posterior_uns = jnp.concatenate(posterior_list, axis=0)
    f_vals_all = jnp.concatenate(f_vals_list, axis=0)

    ecp_vals = [jnp.mean(f_vals_all < (1 - alpha)) for alpha in alpha_list]

    return ecp_vals, posterior_uns, rng_key

# Import libraries
import jax
import jax.numpy as jnp
import blackjax
from functools import partial
from tqdm.notebook import tqdm

def preprocess_groups(param_groups, global_param_names):
    """
    Precompute all index arrays needed by log_prob_fn_groups.
    """
    visible_indices = []
    group_indices = []

    prev = []
    for group in param_groups:
        group_list = [group] if isinstance(group, str) else group
        visible = prev + group_list

        visible_idx = jnp.array(
            [global_param_names.index(name) for name in visible]
        )
        group_idx = jnp.array(
            [global_param_names.index(name) for name in group_list]
        )

        visible_indices.append(visible_idx)
        group_indices.append(group_idx)

        prev = visible

    return visible_indices, group_indices

@jax.jit
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
    return jnp.linalg.norm(theta1 - theta2)

def log_group_prior(theta, bounds, group_indices):
    """
    Log prior for a single parameter group.
    Uniform within bounds, -inf otherwise.
    """
    theta_g = theta[group_indices]
    bounds_g = bounds[group_indices]

    in_bounds = jnp.all(
        (theta_g >= bounds_g[:, 0]) &
        (theta_g <= bounds_g[:, 1])
    )

    return jnp.where(in_bounds, 0.0, -jnp.inf)



@jax.jit
def sample_reference_point(rng_key, bounds):
    rng_key, subkey = jax.random.split(rng_key)
    u = jax.random.uniform(subkey, shape=(bounds.shape[0],))
    theta = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    return rng_key, theta


@partial(jax.jit, static_argnums=2)
def inference_loop(rng_key, initial_state, kernel, num_samples):

    def one_step(state, rng):
        new_state, _ = kernel(rng, state)
        return new_state, new_state.position

    keys = jax.random.split(rng_key, num_samples)
    _, positions = jax.lax.scan(one_step, initial_state, keys)
    return positions


def log_prob_fn_groups(
    theta,
    models_per_group,
    data,
    bounds,
    visible_indices,
    group_indices,
):
    """
    Fully JAX-compatible log-prob.
    """
    log_r_sum = 0.0
    log_p_sum = 0.0

    data = data.reshape(1, -1)

    for g in range(len(models_per_group)):
        v_idx = visible_indices[g]
        g_idx = group_indices[g]

        theta_visible = theta[v_idx].reshape(1, -1)
        input_g = jnp.concatenate([data, theta_visible], axis=-1)

        logits = models_per_group[g](input_g)
        log_r_sum += logits.squeeze()  # log r = logit directly

        log_p_sum += log_group_prior(theta, bounds, g_idx)

    return log_r_sum + log_p_sum

def make_nuts_kernel(log_prob, init_position, rng_key, n_warmup):
    """
    Run warmup ONCE and return a reusable kernel and initial state.
    """
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)
    (state, params), _ = warmup.run(rng_key, init_position, num_steps=n_warmup)
    kernel = blackjax.nuts(log_prob, **params).step
    return kernel, state

@partial(jax.jit, static_argnums=(2,3))
def one_sample_step_groups(
        rng_key,
        theta_star,
        kernel,
        init_state,
        bounds,
        n_samples,
    ):
    """
    Sample from posterior for a single ECP point.
    rng_key is split internally for reference point and MCMC sampling.
    """
    # Split rng_key for reference point and NUTS
    rng_key, key_r0, key_mcmc = jax.random.split(rng_key, 3)

    # Sample reference point
    _, theta_r0 = sample_reference_point(key_r0, bounds)

    # Sample posterior using the precomputed kernel
    posterior = inference_loop(key_mcmc, init_state, kernel, num_samples=n_samples)

    # Compute distance-based ECP statistic
    d_star = distance(theta_star, theta_r0)
    d_samples = jnp.linalg.norm(posterior - theta_r0, axis=1)
    f_val = jnp.mean(d_samples < d_star)

    return f_val, posterior, rng_key


def compute_ecp_tarp_groups(
    x_list,
    theta_star_list,
    alpha_list,
    kernel,
    init_state,
    bounds,
    n_samples,
    rng_key,
):
    f_vals = []
    posteriors = []

    for i in range(x_list.shape[0]):
        rng_key, subkey = jax.random.split(rng_key)

        f_val, posterior, rng_key = one_sample_step_groups(
            subkey,
            x_list[i],
            theta_star_list[i],
            kernel,
            init_state,
            bounds,
            n_samples  # You need to pass n_samples explicitly
        )


        f_vals.append(f_val)
        posteriors.append(posterior)

    f_vals = jnp.stack(f_vals)
    posteriors = jnp.stack(posteriors)

    ecp_vals = [jnp.mean(f_vals < (1 - alpha)) for alpha in alpha_list]

    return ecp_vals, f_vals, posteriors, rng_key


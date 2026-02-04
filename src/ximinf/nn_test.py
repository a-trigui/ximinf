import jax
import jax.numpy as jnp
import blackjax
from functools import partial

# ----------------------------
# Utilities
# ----------------------------
def preprocess_groups(param_groups, global_param_names):
    visible_indices = []
    group_indices = []
    prev = []
    for group in param_groups:
        group_list = [group] if isinstance(group, str) else group
        visible = prev + group_list
        visible_idx = jnp.array([global_param_names.index(name) for name in visible])
        group_idx = jnp.array([global_param_names.index(name) for name in group_list])
        visible_indices.append(visible_idx)
        group_indices.append(group_idx)
        prev = visible
    return visible_indices, group_indices

@jax.jit
def distance(theta1, theta2):
    print('DISTANCE')
    return jnp.linalg.norm(theta1 - theta2)

# def log_group_prior(theta, bounds, group_indices):
#     theta_g = theta[group_indices]
#     bounds_g = bounds[group_indices]
#     in_bounds = jnp.all((theta_g >= bounds_g[:, 0]) & (theta_g <= bounds_g[:, 1]))
#     return jnp.where(in_bounds, 0.0, -jnp.inf)


# def log_group_prior(theta, bounds, group_indices):
#     theta_g = theta[group_indices]
#     bounds_g = bounds[group_indices]

#     mu = 0.5 * (bounds_g[:, 0] + bounds_g[:, 1])
#     sigma = (bounds_g[:, 1] - bounds_g[:, 0]) / (2.0 * 1.96)

#     log_norm = -jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))
#     log_exp = -0.5 * ((theta_g - mu) / sigma) ** 2

#     return jnp.sum(log_norm + log_exp)


@jax.jit
def sample_reference_point(rng_key, bounds):
    print('SAMPLE REFERENCE')
    rng_key, subkey = jax.random.split(rng_key)
    u = jax.random.uniform(subkey, shape=(bounds.shape[0],))
    theta = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    return rng_key, theta

def inference_loop(initial_state, kernel, num_samples, rng_key):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state.position
    
    rng_key, sample_key = jax.random.split(rng_key)
    keys = jax.random.split(sample_key, num_samples)
    _, positions = jax.lax.scan(one_step, initial_state, keys)

    return rng_key, positions

# @partial(jax.jit, static_argnums=(1,))
def log_prob_single_group(theta_visible, model, xi, g_idx, theta, bounds):
    # print(f'LOG PROB GROUP {g_idx+1}')
    input_g = jnp.concatenate([xi, theta_visible], axis=-1)
    logits = model(input_g).squeeze()
    p = jax.nn.sigmoid(logits)
    log_r = jnp.log(p) - jnp.log1p(-p)
    # log_p = log_group_prior(theta, bounds, g_idx)
    return log_r #+ log_p

def log_prob_fn_groups(theta, models_per_group, xi, bounds, visible_indices, group_indices):
    xi = xi.reshape(1, -1)
    log_sum = 0.0

    for v_idx, g_idx, model in zip(visible_indices, group_indices, models_per_group):
        theta_visible = theta[v_idx].reshape(1, -1)
        log_prob = log_prob_single_group(theta_visible, model, xi, g_idx, theta, bounds)
        log_sum += log_prob

    return log_sum

def build_kernel(log_prob, init_position, n_warmup, rng_key):
    # print('BUILD KERNEL')
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)
    rng_key, warmup_key = jax.random.split(rng_key)
    (warmup_state, params), _ = warmup.run(warmup_key, init_position, num_steps=n_warmup)
    kernel = blackjax.nuts(log_prob, **params).step
    return rng_key, kernel, warmup_state

# @partial(jax.jit, static_argnums=(0, 1, 2))
def sample_posterior(log_prob, n_warmup, n_samples, init_position, rng_key):
    # print('SAMPLE POSTERIOR')
    rng_key, kernel, warmup_state = build_kernel(log_prob, init_position, n_warmup, rng_key)
    rng_key, positions = inference_loop(warmup_state, kernel, n_samples, rng_key)
    return rng_key, positions

# ----------------------------
# Per-sample posterior
# ----------------------------
def one_sample_step_groups(rng_key, xi, theta_star, bounds,
                           models_per_group, visible_indices, group_indices,
                           n_warmup, n_samples):
    # Split rng for reference and MCMC
    rng_key, key_r0, key_mcmc = jax.random.split(rng_key, 3)
    
    # Reference point
    _, theta_r0 = sample_reference_point(key_r0, bounds)
    
    # Define log-prob for this xi
    def log_post(theta):
        return log_prob_fn_groups(theta, models_per_group, xi, bounds, visible_indices, group_indices)
    
    # Sample from posterior
    rng_key, posterior = sample_posterior(log_post, n_warmup, n_samples, theta_star, key_mcmc)

    # Compute ECP statistic
    d_star = distance(theta_star, theta_r0)
    d_samples = jnp.linalg.norm(posterior - theta_r0, axis=1)
    f_val = jnp.mean(d_samples < d_star)
    
    return f_val, posterior, rng_key


def compute_ecp_tarp_groups(models_per_group, x_list, theta_star_list, alpha_list,
                            bounds, visible_indices, group_indices,
                            n_warmup, n_samples, rng_key):

    def scan_step(rng_key, xi_theta):
        xi, theta_star = xi_theta
        rng_key, subkey = jax.random.split(rng_key)
        f_val, posterior, rng_key = one_sample_step_groups(
            subkey, xi, theta_star, bounds,
            models_per_group, visible_indices, group_indices,
            n_warmup, n_samples
        )
        return rng_key, (f_val, posterior)

    rng_key, (f_vals, posteriors) = jax.lax.scan(
        scan_step, rng_key, (x_list, theta_star_list)
    )

    ecp_vals = [jnp.mean(f_vals < (1 - alpha)) for alpha in alpha_list]
    return ecp_vals, f_vals, posteriors, rng_key

import jax
import jax.numpy as jnp
import blackjax
from functools import partial
from tqdm.notebook import tqdm

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
    return jnp.linalg.norm(theta1 - theta2)

def log_group_prior(theta, bounds, group_indices):
    theta_g = theta[group_indices]
    bounds_g = bounds[group_indices]
    in_bounds = jnp.all((theta_g >= bounds_g[:, 0]) & (theta_g <= bounds_g[:, 1]))
    return jnp.where(in_bounds, 0.0, -jnp.inf)

@jax.jit
def sample_reference_point(rng_key, bounds):
    rng_key, subkey = jax.random.split(rng_key)
    u = jax.random.uniform(subkey, shape=(bounds.shape[0],))
    theta = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    return rng_key, theta

@partial(jax.jit, static_argnums=(2, 3))  # kernel and num_samples are static
def inference_loop(rng_key, initial_state, kernel, num_samples):
    def one_step(state, rng):
        new_state, _ = kernel(rng, state)
        return new_state, new_state.position

    keys = jax.random.split(rng_key, num_samples)
    _, positions = jax.lax.scan(one_step, initial_state, keys)
    return positions


# def log_prob_fn_groups(theta, models_per_group, xi, bounds, visible_indices, group_indices):
#     log_r_sum = 0.0
#     log_p_sum = 0.0
#     xi = xi.reshape(1, -1)
#     for g in range(len(models_per_group)):
#         v_idx = visible_indices[g]
#         g_idx = group_indices[g]
#         theta_visible = theta[v_idx].reshape(1, -1)
#         input_g = jnp.concatenate([xi, theta_visible], axis=-1)
#         logits = models_per_group[g](input_g)
#         log_r_sum += logits.squeeze()
#         log_p_sum += log_group_prior(theta, bounds, g_idx)
#     return log_r_sum + log_p_sum

# def log_prob_fn_groups(theta, models_per_group, xi, bounds, visible_indices, group_indices):
#     xi = xi.reshape(1, -1)
    
#     def log_prob_single_group(v_idx, g_idx, model):
#         theta_visible = theta[v_idx].reshape(1, -1)
#         input_g = jnp.concatenate([xi, theta_visible], axis=-1)
#         log_r = model(input_g).squeeze()
#         log_p = log_group_prior(theta, bounds, g_idx)
#         return log_r + log_p

#     # Vectorize over groups
#     log_probs = jax.vmap(log_prob_single_group)(visible_indices, group_indices, models_per_group)
#     return jnp.sum(log_probs)

def log_prob_fn_groups(theta, models_per_group, xi, bounds, visible_indices, group_indices):
    xi = xi.reshape(1, -1)
    log_r_sum = 0.0
    log_p_sum = 0.0

    for v_idx, g_idx, model in zip(visible_indices, group_indices, models_per_group):
        theta_visible = theta[v_idx].reshape(1, -1)
        input_g = jnp.concatenate([xi, theta_visible], axis=-1)
        log_r_sum += model(input_g).squeeze()
        log_p_sum += log_group_prior(theta, bounds, g_idx)

    return log_r_sum + log_p_sum


# ----------------------------
# Per-sample posterior
# ----------------------------
def one_sample_step_groups(rng_key, xi, theta_star, bounds,
                           models_per_group, visible_indices, group_indices,
                           n_warmup, n_samples):
    # Split rng for reference and MCMC
    rng_key, key_r0, key_mcmc, key_kernel = jax.random.split(rng_key, 4)
    
    # Reference point
    _, theta_r0 = sample_reference_point(key_r0, bounds)
    
    # Define log-prob for this xi
    def log_post(theta):
        return log_prob_fn_groups(theta, models_per_group, xi, bounds, visible_indices, group_indices)
    
    # Run warmup and build kernel
    warmup = blackjax.window_adaptation(blackjax.nuts, log_post)
    (state, params), _ = warmup.run(key_kernel, theta_star, num_steps=n_warmup)
    kernel = blackjax.nuts(log_post, **params).step
    
    # Sample posterior
    posterior = inference_loop(key_mcmc, state, kernel, int(n_samples))
    
    # Compute ECP statistic
    d_star = distance(theta_star, theta_r0)
    d_samples = jnp.linalg.norm(posterior - theta_r0, axis=1)
    f_val = jnp.mean(d_samples < d_star)
    
    return f_val, posterior, rng_key

# ----------------------------
# Batched ECP computation
# ----------------------------
# def compute_ecp_tarp_groups(models_per_group, x_list, theta_star_list, alpha_list,
#                             bounds, visible_indices, group_indices,
#                             n_warmup, n_samples, rng_key):
#     f_vals = []
#     posteriors = []
#     for i in range(x_list.shape[0]):
#         print(f"Processing sample {i+1}/{x_list.shape[0]}")
#         rng_key, subkey = jax.random.split(rng_key)
#         f_val, posterior, rng_key = one_sample_step_groups(
#             subkey, x_list[i], theta_star_list[i], bounds,
#             models_per_group, visible_indices, group_indices,
#             n_warmup, n_samples
#         )
#         f_vals.append(f_val)
#         posteriors.append(posterior)
#     f_vals = jnp.stack(f_vals)
#     posteriors = jnp.stack(posteriors)
#     ecp_vals = [jnp.mean(f_vals < (1 - alpha)) for alpha in alpha_list]
#     return ecp_vals, f_vals, posteriors, rng_key

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

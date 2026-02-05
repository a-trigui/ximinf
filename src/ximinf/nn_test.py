import jax
import jax.numpy as jnp
import blackjax

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

def log_group_prior(theta, priors, group_names, group_indices):
    """
    theta: flat array
    priors: dict of name -> {'range', 'type'}
    group_names: list of names in this group
    group_indices: list/array of indices in theta corresponding to group_names
    """
    logp = 0.0
    for idx, name in zip(group_indices, group_names):
        val = theta[idx]
        info = priors[name]
        low, high = info["range"]
        ptype = info["type"]

        if ptype == "uniform":
            logp_i = jnp.where(
                (val >= low) & (val <= high),
                -jnp.log(high - low),
                -jnp.inf,
            )
        elif ptype == "gaussian":
            mean = 0.5 * (low + high)
            sigma = (high - low) / (2.0 * 1.96)
            logp_i = (
                -0.5 * ((val - mean) / sigma) ** 2
                - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))
            )
        elif ptype == "half-gaussian":
            sigma = high / 1.96
            logp_i = jnp.where(
                val >= 0.0,
                jnp.log(2.0)
                - jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))
                - 0.5 * (val / sigma) ** 2,
                -jnp.inf,
            )
        elif ptype == "log-uniform":
            logp_i = jnp.where(
                (val >= low) & (val <= high),
                -jnp.log(val) - jnp.log(jnp.log(high / low)),
                -jnp.inf,
            )
        else:
            raise ValueError(f"Unknown prior type '{ptype}'")

        logp += logp_i

    return logp


def inference_loop(initial_state, kernel, num_samples, rng_key):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state.position
    
    rng_key, sample_key = jax.random.split(rng_key)
    keys = jax.random.split(sample_key, num_samples)
    _, positions = jax.lax.scan(one_step, initial_state, keys)

    return rng_key, positions

def log_prob_single_group(
    theta_visible,
    model,
    xi,
    theta,
    priors,
    group_names,
    group_indices
):
    input_g = jnp.concatenate([xi, theta_visible], axis=-1)
    logits = model(input_g).squeeze()

    prob = jax.nn.sigmoid(logits)
    log_r = jnp.log(prob) - jnp.log1p(-prob)

    log_p = log_group_prior(theta, priors, group_names, group_indices)


    return log_r + log_p


def log_prob_fn_groups(
    theta,
    models_per_group,
    xi,
    priors,
    visible_indices,
    group_indices,
    group_names_list,
):
    xi = xi.reshape(1, -1)
    log_sum = 0.0

    for v_idx, g_idx, group_names, model in zip(
        visible_indices, group_indices, group_names_list, models_per_group
    ):
        theta_visible = theta[v_idx].reshape(1, -1)

        log_prob = log_prob_single_group(
            theta_visible,
            model,
            xi,
            theta,
            priors,
            group_names,
            g_idx   # <--- only the indices for this group
        )


        log_sum += log_prob

    return log_sum


def build_kernel(log_prob, init_position, n_warmup, rng_key):
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)
    rng_key, warmup_key = jax.random.split(rng_key)
    (warmup_state, params), _ = warmup.run(warmup_key, init_position, num_steps=n_warmup)
    kernel = blackjax.nuts(log_prob, **params).step
    return rng_key, kernel, warmup_state

def sample_posterior(log_prob, n_warmup, n_samples, init_position, rng_key):
    rng_key, kernel, warmup_state = build_kernel(log_prob, init_position, n_warmup, rng_key)
    rng_key, positions = inference_loop(warmup_state, kernel, n_samples, rng_key)
    return rng_key, positions

# @jax.jit
# def sample_reference_point(rng_key, priors, param_names):
#     """
#     Sample a reference point uniformly over parameter ranges,
#     consistent with the new prior structure.
#     """
#     rng_key, subkey = jax.random.split(rng_key)

#     lows = jnp.array([priors[name]["range"][0] for name in param_names])
#     highs = jnp.array([priors[name]["range"][1] for name in param_names])

#     u = jax.random.uniform(subkey, shape=(len(param_names),))
#     theta = lows + u * (highs - lows)

#     return rng_key, theta

# def one_sample_step_groups(
#     rng_key,
#     xi,
#     theta_star,
#     priors,
#     param_names,
#     models_per_group,
#     visible_indices,
#     group_names_list,
#     n_warmup,
#     n_samples,
# ):
#     rng_key, key_r0, key_mcmc = jax.random.split(rng_key, 3)

#     _, theta_r0 = sample_reference_point(key_r0, priors)

#     def log_post(theta):
#         return log_prob_fn_groups(
#             theta,
#             models_per_group,
#             xi,
#             priors,
#             param_names,
#             visible_indices,
#             group_names_list,
#         )

#     rng_key, posterior = sample_posterior(
#         log_post, n_warmup, n_samples, theta_star, key_mcmc
#     )

#     d_star = distance(theta_star, theta_r0)
#     d_samples = jnp.linalg.norm(posterior - theta_r0, axis=1)
#     f_val = jnp.mean(d_samples < d_star)

#     return f_val, posterior, rng_key



# def compute_ecp_tarp_groups(
#     models_per_group,
#     x_list,
#     theta_star_list,
#     alpha_list,
#     priors,
#     param_names,
#     visible_indices,
#     group_names_list,
#     n_warmup,
#     n_samples,
#     rng_key,
# ):
#     def scan_step(rng_key, xi_theta):
#         xi, theta_star = xi_theta
#         rng_key, subkey = jax.random.split(rng_key)

#         f_val, posterior, rng_key = one_sample_step_groups(
#             subkey,
#             xi,
#             theta_star,
#             priors,
#             param_names,
#             models_per_group,
#             visible_indices,
#             group_names_list,
#             n_warmup,
#             n_samples,
#         )

#         return rng_key, (f_val, posterior)

#     rng_key, (f_vals, posteriors) = jax.lax.scan(
#         scan_step, rng_key, (x_list, theta_star_list)
#     )

#     ecp_vals = [jnp.mean(f_vals < (1.0 - alpha)) for alpha in alpha_list]

#     return ecp_vals, f_vals, posteriors, rng_key

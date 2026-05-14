import jax
import jax.numpy as jnp
import ximinf.nn_inference as nninf 

@jax.jit
def sample_reference_point(rng_key, priors, param_names):
    """
    Sample a reference point uniformly over parameter ranges,
    consistent with the new prior structure.
    """
    rng_key, subkey = jax.random.split(rng_key)

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
    group_names_list,
    n_warmup,
    n_samples,
):
    rng_key, key_r0, key_mcmc = jax.random.split(rng_key, 3)

    _, theta_r0 = sample_reference_point(key_r0, priors)

    def log_post(theta):
        return nninf.log_prob_fn_groups(
            theta,
            models_per_group,
            xi,
            priors,
            param_names,
            visible_indices,
            group_names_list,
        )

    rng_key, posterior = nninf.sample_posterior(
        log_post, n_warmup, n_samples, theta_star, key_mcmc
    )

    d_star = nninf.distance(theta_star, theta_r0)
    d_samples = jnp.linalg.norm(posterior - theta_r0, axis=1)
    f_val = jnp.mean(d_samples < d_star)

    return f_val, posterior, rng_key



def compute_ecp_tarp_groups(
    models_per_group,
    x_list,
    theta_star_list,
    alpha_list,
    priors,
    param_names,
    visible_indices,
    group_names_list,
    n_warmup,
    n_samples,
    rng_key,
):
    def scan_step(rng_key, xi_theta):
        xi, theta_star = xi_theta
        rng_key, subkey = jax.random.split(rng_key)

        f_val, posterior, rng_key = one_sample_step_groups(
            subkey,
            xi,
            theta_star,
            priors,
            param_names,
            models_per_group,
            visible_indices,
            group_names_list,
            n_warmup,
            n_samples,
        )

        return rng_key, (f_val, posterior)

    rng_key, (f_vals, posteriors) = jax.lax.scan(
        scan_step, rng_key, (x_list, theta_star_list)
    )

    ecp_vals = [jnp.mean(f_vals < (1.0 - alpha)) for alpha in alpha_list]

    return ecp_vals, f_vals, posteriors, rng_key
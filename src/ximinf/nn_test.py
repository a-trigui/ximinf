import jax
import jax.numpy as jnp
import jax.scipy as jsp
import ximinf.nn_inference as nninf 
import ximinf.nn_train as nntr


def evaluate_models_per_group(
    models_per_group,
    param_groups,
    all_group_param_slices,
    data_test,
    mask_test,
    M_norm_test,
    batch_size=128,
):
    # Set models to evaluation mode
    for model_g in models_per_group:
        model_g.eval()  # disable dropout, etc.

    metrics_per_group = []

    # Loop over groups
    for g, model_g in enumerate(models_per_group):

        print(f"\n=== Evaluating model for group {g}: {param_groups[g]} ===")

        chosen_test = all_group_param_slices[g]["chosen_test"]
        labels_test = all_group_param_slices[g]["labels_test"]

        num_samples = labels_test.shape[0]

        all_logits = []
        all_labels = []

        for i in range(0, num_samples, batch_size):

            xb = jnp.concatenate(
                [
                    data_test[i:i + batch_size],
                    mask_test[i:i + batch_size],
                    M_norm_test[i:i + batch_size],
                    chosen_test[i:i + batch_size],
                ],
                axis=-1,
            )

            yb = labels_test[i:i + batch_size, None].astype(jnp.int32)

            # Model predictions
            logits = nntr.pred_step(model_g, xb)
            all_logits.append(logits)
            all_labels.append(yb)

        # Merge batches
        all_logits = jnp.concatenate(all_logits, axis=0)
        all_labels = jnp.concatenate(all_labels, axis=0)

        all_preds = (
            jsp.special.expit(all_logits) > 0.5
        ).astype(jnp.int32)

        # Confusion matrix components
        TP = jnp.sum((all_preds == 1) & (all_labels == 1))
        TN = jnp.sum((all_preds == 0) & (all_labels == 0))
        FP = jnp.sum((all_preds == 1) & (all_labels == 0))
        FN = jnp.sum((all_preds == 0) & (all_labels == 1))

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP + 1e-8)
        sensitivity = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)

        print(
            f"Group {g} ({param_groups[g]}): "
            f"Accuracy={accuracy:.3f}, "
            f"Precision={precision:.3f}, "
            f"Sensitivity={sensitivity:.3f}, "
            f"Specificity={specificity:.3f}"
        )

        metrics_per_group.append({
            "accuracy": accuracy,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
        })

    return metrics_per_group

def sample_reference_point(rng_key, priors, param_names):
    """
    Sample a reference point uniformly over parameter ranges,
    consistent with the new prior structure.
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
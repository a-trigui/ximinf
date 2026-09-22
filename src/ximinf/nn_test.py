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
    """
    Evaluate binary classifiers independently for each parameter group.

    Each model is evaluated on the corresponding test subset defined by
    ``all_group_param_slices``. The input features are constructed by
    concatenating the test data, mask, normalized magnitude information,
    and the selected parameters for the group. Predictions are obtained
    from the model logits using a sigmoid threshold of 0.5.

    For each parameter group, the function computes the accuracy, precision,
    sensitivity (true positive rate), and specificity (true negative rate)
    from the resulting confusion matrix.

    Parameters
    ----------
    models_per_group : sequence
        Sequence of trained classification models, with one model
        corresponding to each parameter group.

    param_groups : sequence
        Labels or identifiers describing the parameter group associated with
        each model.

    all_group_param_slices : sequence of dict
        Group-specific test-set information. Each element must contain
        ``"chosen_test"`` and ``"labels_test"`` entries corresponding to the
        selected parameters and binary labels for that group.

    data_test : jax.Array
        Test data used as model input.

    mask_test : jax.Array
        Test-set mask values concatenated with the input data.

    M_norm_test : jax.Array
        Normalized test-set magnitude information concatenated with the model
        inputs.

    batch_size : int, optional
        Number of test samples processed in each batch. Default is 128.

    Returns
    -------
    metrics_per_group : list of dict
        List containing one dictionary per parameter group. Each dictionary
        contains the following metrics:

        ``"accuracy"``
            Fraction of correctly classified samples.

        ``"precision"``
            Fraction of predicted positive samples that are true positives.

        ``"sensitivity"``
            Fraction of positive samples correctly identified by the model.

        ``"specificity"``
            Fraction of negative samples correctly identified by the model.

    Notes
    -----
    The sigmoid probability is thresholded at 0.5 to obtain binary
    predictions. A small value of ``1e-8`` is added to the denominators of
    precision, sensitivity, and specificity to avoid division by zero.
    """
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
    Sample a parameter-space reference point uniformly from the prior ranges.

    Each parameter is sampled independently from a uniform distribution
    between the lower and upper bounds specified in ``priors``. The function
    also returns an updated JAX random key for subsequent random-number
    generation.

    Parameters
    ----------
    rng_key : jax.Array
        JAX PRNG key used to generate the random sample.

    priors : dict
        Dictionary containing the prior definition for each parameter.
        For every parameter in ``param_names``, ``priors[name]["range"]``
        must contain the lower and upper bounds of the prior interval.

    param_names : sequence of str
        Names of the parameters to sample. The ordering determines the
        ordering of the returned parameter vector.

    Returns
    -------
    rng_key : jax.Array
        Updated JAX PRNG key.

    theta : jax.Array
        One-dimensional array containing the sampled parameter values,
        ordered according to ``param_names``.

    Notes
    -----
    The sampling is uniform in the parameterization defined by the prior
    ranges. No normalization using ``param_stats`` is applied.
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
    
    """
    Perform posterior sampling for one simulated observation and compute
    its TARP rank statistic.

    A reference point is first sampled uniformly from the prior parameter
    ranges. The posterior distribution conditioned on ``xi`` is then
    sampled using the group-specific neural-network inference models.
    The TARP statistic is computed as the fraction of posterior samples
    lying closer to the reference point than the true parameter
    ``theta_star``.

    Parameters
    ----------
    rng_key : jax.Array
        JAX PRNG key used for reference-point generation and posterior
        sampling.

    xi : jax.Array
        Observed or simulated data conditioned upon when evaluating the
        posterior.

    theta_star : jax.Array
        True parameter values associated with ``xi``.

    priors : dict
        Prior definitions for the model parameters.

    param_names : sequence of str
        Names of the inferred parameters. The ordering must be consistent
        with the parameter vectors used by the inference models.

    models_per_group : sequence
        Collection of trained neural-network inference models, with one
        model associated with each parameter group.

    visible_indices : sequence
        Indices specifying which parameters are visible to the corresponding
        group-specific inference models.

    group_indices : sequence
        Indices defining the parameter grouping used by the inference
        models.

    group_names_list : sequence
        Names identifying the parameter groups.

    param_stats : dict
        Statistics used to transform posterior samples back to the original
        parameterization. For each parameter, ``param_stats[name]`` must
        contain ``"mu"`` and ``"sigma"``.

    n_warmup : int
        Number of warm-up iterations used by the posterior sampler.

    n_samples : int
        Number of posterior samples to draw after warm-up.

    Returns
    -------
    f_val : jax.Array
        TARP rank statistic, defined as the fraction of posterior samples
        whose Euclidean distance from the sampled reference point is smaller
        than the distance between ``theta_star`` and the same reference
        point.

    posterior_unnormed : jax.Array
        Posterior samples transformed from the normalized parameterization
        back to the original parameterization using ``param_stats``.
    """
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
    """
    Compute empirical coverage probabilities using TARP for grouped models.

    The function evaluates the TARP statistic for each simulated observation
    and its corresponding true parameter value using ``jax.lax.scan``.
    Empirical coverage probabilities are then computed for each requested
    credibility level.

    Parameters
    ----------
    models_per_group : sequence
        Collection of trained neural-network inference models, with one
        model associated with each parameter group.

    x_list : jax.Array
        Collection of simulated observations used to evaluate posterior
        coverage.

    theta_star_list : jax.Array
        True parameter values corresponding to the observations in
        ``x_list``.

    alpha_list : sequence of float
        Significance levels at which empirical coverage is evaluated.
        For each ``alpha``, the function computes the fraction of TARP
        statistics satisfying ``f < 1 - alpha``.

    priors : dict
        Prior definitions for the inferred parameters.

    param_names : sequence of str
        Names of the inferred parameters.

    visible_indices : sequence
        Indices specifying the parameters visible to each group-specific
        inference model.

    group_indices : sequence
        Indices defining the parameter grouping used by the inference
        models.

    group_names_list : sequence
        Names identifying the parameter groups.

    param_stats : dict
        Statistics used to transform posterior samples from the normalized
        parameterization to the original parameterization. Each parameter
        must have ``"mu"`` and ``"sigma"`` entries.

    n_warmup : int
        Number of warm-up iterations used for each posterior sample.

    n_samples : int
        Number of posterior samples generated for each observation.

    rng_key : jax.Array
        JAX PRNG key used for all random sampling operations.

    Returns
    -------
    ecp_vals : list of jax.Array
        Empirical coverage probabilities corresponding to the values in
        ``alpha_list``. Each value is the fraction of TARP statistics
        satisfying ``f < 1 - alpha``.

    f_vals : jax.Array
        TARP statistic for each observation in ``x_list``.

    posteriors : jax.Array
        Posterior samples obtained for each observation.

    rng_key : jax.Array
        Updated JAX PRNG key after all sampling operations.
    """

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
import jax
import jax.numpy as jnp
import blackjax


from ximinf.data_helper import group_names_as_lists


# ----------------------------
# Utilities
# ----------------------------
def preprocess_groups(param_groups, global_param_names):
    """
    Construct visible and group-specific parameter indices for each group.

    Parameters
    ----------
    param_groups : list
        Groups of parameter names. Each group can be either a single
        parameter name or an iterable of parameter names. Parameters from
        previous groups are included in the visible parameters of subsequent
        groups.
    global_param_names : list
        Ordered list of all parameter names defining the global parameter
        vector.

    Returns
    -------
    visible_indices : list of jax.Array
        For each group, indices of all parameters visible to that group,
        including parameters from preceding groups and the parameters in
        the current group.
    group_indices : list of jax.Array
        For each group, indices corresponding only to the parameters in that
        group.

    Notes
    -----
    The ordering of the indices follows the ordering of the parameter names
    in `global_param_names` as determined by their occurrence in the
    constructed visible or group-specific name lists.
    """
    visible_indices = []
    group_indices = []
    prev = []
    for group_list in group_names_as_lists(param_groups):
        visible = prev + group_list
        visible_indices.append(jnp.array([global_param_names.index(name) for name in visible]))
        group_indices.append(jnp.array([global_param_names.index(name) for name in group_list]))
        prev = visible
    return visible_indices, group_indices

def log_group_prior(theta, priors, group_names, group_indices):
    """
    Compute the joint log-prior probability for a parameter group.

    Parameters
    ----------
    theta : jax.Array
        Flat parameter vector containing the parameters of the full
        inference problem.
    priors : dict
        Dictionary mapping parameter names to prior specifications. Each
        specification must contain ``"range"`` and ``"type"`` entries.
        Supported prior types are ``"uniform"``, ``"gaussian"``,
        ``"half-gaussian"``, ``"positive-gaussian"``, ``"log-uniform"``,
        and ``"exponential"``.
    group_names : list
        Names of the parameters belonging to the current group.
    group_indices : array-like
        Indices into `theta` corresponding to `group_names`.

    Returns
    -------
    logp : jax.Array or float
        Sum of the log-prior probabilities of the parameters in the group.
        Parameters outside the support of their prior contribute
        ``-jnp.inf``.
    
    Notes
    -----
    The prior parameterizations are defined from the lower and upper bounds
    stored in ``priors[name]["range"]``. For Gaussian-based priors, the
    standard deviation is chosen such that the interval corresponds to
    approximately 95% of the underlying Gaussian distribution.

    The exponential prior is parameterized such that its 95th percentile is
    equal to the upper bound of the specified range.
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
        elif ptype == "positive-gaussian":
            mu = 0.5 * (low + high)
            sigma = (high - low) / (2.0 * 1.96)

            alpha = (0.0 - mu) / sigma
            Phi_alpha = 0.5 * (1.0 + jax.scipy.special.erf(alpha / jnp.sqrt(2.0)))

            log_norm = -jnp.log(1.0 - Phi_alpha)

            logp_i = jnp.where(
                val >= 0.0,
                -jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi))
                - 0.5 * ((val - mu) / sigma) ** 2
                + log_norm,
                -jnp.inf,
            )
        elif ptype == "log-uniform":
            logp_i = jnp.where(
                (val >= low) & (val <= high),
                -jnp.log(val) - jnp.log(jnp.log(high / low)),
                -jnp.inf,
            )
        elif ptype == "exponential":
            lam = -jnp.log(1.0 - 0.95) / high

            logp_i = jnp.where(
                val >= 0.0,
                jnp.log(lam) - lam * val,
                -jnp.inf,
            )
        else:
            raise ValueError(f"Unknown prior type '{ptype}'")

        logp += logp_i

    return logp

def make_transform(low, high):
    scale = high - low

    def to_theta(z):
        return low + scale * jax.nn.sigmoid(z)

    def to_z(theta, eps=1e-6):
        u = jnp.clip((theta - low) / scale, eps, 1 - eps)
        return jnp.log(u) - jnp.log1p(-u)          # logit

    def log_det_jac(z):
        # d theta / d z = scale * sigmoid(z) * (1 - sigmoid(z))
        return jnp.sum(jnp.log(scale) + jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z))

    return to_theta, to_z, log_det_jac


def inference_loop(initial_state, kernel, num_samples, rng_key):
    """
    Draw posterior samples using a BlackJAX transition kernel.

    Parameters
    ----------
    initial_state : blackjax state
        Initial sampler state from which the Markov chain is started.
    kernel : callable
        BlackJAX transition kernel. It must accept a random key and sampler
        state and return the updated state together with sampler information.
    num_samples : int
        Number of MCMC samples to generate.
    rng_key : jax.Array
        JAX pseudo-random number generator key.

    Returns
    -------
    rng_key : jax.Array
        Updated JAX random key after generating the sample keys.
    positions : jax.Array
        Sampled parameter positions collected along the MCMC chain. The
        leading dimension has length `num_samples`.
    """
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
    group_indices,
):
    """
    Evaluate the group-specific neural likelihood ratio and prior.

    Parameters
    ----------
    theta_visible : jax.Array
        Parameters visible to the current group, including the parameters
        belonging to the group and any parameters from preceding groups.
        Expected to be compatible with the model input shape.
    model : callable
        Neural network model for the current parameter group. The model
        receives the concatenation of `xi` and `theta_visible` and returns
        a scalar logit.
    xi : jax.Array
        Simulation summary or observation features supplied to the neural
        network.
    theta : jax.Array
        Full parameter vector used to evaluate the group prior.
    priors : dict
        Dictionary containing the prior specification for each parameter.
    group_names : list
        Names of the parameters belonging to the current group.
    group_indices : array-like
        Indices into `theta` corresponding to `group_names`.

    Returns
    -------
    log_prob : jax.Array
        Sum of the neural-network output logit and the log-prior probability
        of the current parameter group.

    Notes
    -----
    The likelihood-trick and the output sigmoid are inverse function of each other. They cancel and we can directly use the logits.
    """
    input_g = jnp.concatenate([xi, theta_visible], axis=-1)
    logits = model(input_g).squeeze()

    # prob = jax.nn.sigmoid(logits)
    # log_r = jnp.log(prob) - jnp.log1p(-prob)

    log_p = log_group_prior(theta, priors, group_names, group_indices)

    return logits + log_p #+ log_r 

def log_prob_fn_groups(
    theta,
    models_per_group,
    xi,
    priors,
    visible_indices,
    group_indices,
    group_names_list,
):
    """
    Evaluate the joint log-probability across all parameter groups.

    Parameters
    ----------
    theta : jax.Array
        Full parameter vector at which the joint log-probability is evaluated.
    models_per_group : sequence of callable
        Neural network model associated with each parameter group.
    xi : jax.Array
        Simulation summary or observation features supplied to each group
        model.
    priors : dict
        Dictionary containing the prior specification for each parameter.
    visible_indices : sequence of array-like
        Indices defining the parameters visible to each group.
    group_indices : sequence of array-like
        Indices defining the parameters belonging to each group.
    group_names_list : sequence of list
        Parameter names associated with each group.

    Returns
    -------
    log_sum : jax.Array
        Joint log-probability obtained by summing the group-specific neural
        network contributions and log-prior terms.
    """
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
            g_idx
        )

        log_sum += log_prob

    return log_sum


def build_kernel(log_prob, init_position, n_warmup, rng_key):
    """
    Adapt and construct a BlackJAX NUTS transition kernel.

    Parameters
    ----------
    log_prob : callable
        Function returning the log-probability for a given parameter
        position.
    init_position : jax.Array
        Initial parameter position used to initialize the warmup procedure.
    n_warmup : int
        Number of warmup steps used for BlackJAX window adaptation.
    rng_key : jax.Array
        JAX pseudo-random number generator key.

    Returns
    -------
    rng_key : jax.Array
        Updated JAX random key after the warmup key has been consumed.
    kernel : callable
        Adapted BlackJAX NUTS transition kernel.
    warmup_state : blackjax state
        Final sampler state produced by the warmup procedure, suitable for
        starting posterior sampling.
    """
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)
    rng_key, warmup_key = jax.random.split(rng_key)
    (warmup_state, params), _ = warmup.run(warmup_key, init_position, num_steps=n_warmup)
    kernel = blackjax.nuts(log_prob, **params).step
    return rng_key, kernel, warmup_state

def sample_posterior(log_prob, n_warmup, n_samples, init_position, rng_key):
    """
    Adapt a NUTS sampler and draw posterior samples.

    Parameters
    ----------
    log_prob : callable
        Function returning the log-probability of a parameter position.
    n_warmup : int
        Number of warmup steps used for NUTS adaptation.
    n_samples : int
        Number of posterior samples to generate after warmup.
    init_position : jax.Array
        Initial parameter position for the warmup and sampling procedure.
    rng_key : jax.Array
        JAX pseudo-random number generator key.

    Returns
    -------
    rng_key : jax.Array
        Updated JAX random key after warmup and posterior sampling.
    positions : jax.Array
        Posterior parameter samples with leading dimension `n_samples`..
    """
    rng_key, kernel, warmup_state = build_kernel(log_prob, init_position, n_warmup, rng_key)
    rng_key, positions = inference_loop(warmup_state, kernel, n_samples, rng_key)
    return rng_key, positions

def make_box_transform(priors, param_names):
    """
    Build the bijection between the prior box and unbounded space.
 
    All parameters are assumed to have a ``"uniform"`` prior. Each parameter
    theta_i in [low_i, high_i] is written as
 
        theta_i = low_i + (high_i - low_i) * sigmoid(z_i),   z_i in R.
 
    Parameters
    ----------
    priors : dict
        Dictionary mapping parameter names to prior specifications. Each
        specification must contain a ``"range"`` entry ``(low, high)``.
    param_names : sequence of str
        Parameter names, in the same order as the entries of the flat
        parameter vector `theta`.
 
    Returns
    -------
    to_theta : callable
        Maps an unbounded vector z to theta inside the prior box.
    to_z : callable
        Inverse map (logit of the rescaled theta). Values are clipped
        slightly inside the box so that the result is finite.
    log_det_jac : callable
        Maps z to log|det d theta / d z|, the log-Jacobian of `to_theta`.
    """
    low = jnp.array([priors[n]["range"][0] for n in param_names], dtype=float)
    high = jnp.array([priors[n]["range"][1] for n in param_names], dtype=float)
    scale = high - low
    log_scale = jnp.log(scale)
 
    def to_theta(z):
        return low + scale * jax.nn.sigmoid(z)
 
    def to_z(theta, eps=1e-6):
        p = jnp.clip((theta - low) / scale, eps, 1.0 - eps)
        return jnp.log(p) - jnp.log1p(-p)
 
    def log_det_jac(z):
        # d theta / d z = scale * sigmoid(z) * (1 - sigmoid(z))
        return jnp.sum(
            log_scale + jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z)
        )
 
    return to_theta, to_z, log_det_jac
 
 
def make_log_prob_unbounded(
    to_theta,
    log_det_jac,
    models_per_group,
    xi,
    priors,
    visible_indices,
    group_indices,
    group_names_list,
):
    """
    Build the log-density of the posterior in unbounded space.
 
    Parameters
    ----------
    to_theta : callable
        Maps z to theta inside the prior support.
    log_det_jac : callable
        Maps z to the log-Jacobian of `to_theta`.
    models_per_group : sequence of callable
        Neural network model associated with each parameter group.
    xi : jax.Array
        Simulation summary or observation features supplied to each group
        model.
    priors : dict
        Dictionary containing the prior specification for each parameter.
    visible_indices : sequence of array-like
        Indices defining the parameters visible to each group.
    group_indices : sequence of array-like
        Indices defining the parameters belonging to each group.
    group_names_list : sequence of list
        Parameter names associated with each group.
 
    Returns
    -------
    log_prob_unbounded : callable
        Function z -> log q(z), where
        q(z) = p(theta = g(z) | x) * |g'(z)|.
        The networks and priors are only evaluated at theta = g(z), which
        always lies inside the prior support.
    """
 
    def log_prob_unbounded(z):
        theta = to_theta(z)
        log_post = log_prob_fn_groups(
            theta,
            models_per_group,
            xi,
            priors,
            visible_indices,
            group_indices,
            group_names_list,
        )
        return log_post + log_det_jac(z)
 
    return log_prob_unbounded
 
 
def sample_posterior_bounded(
    models_per_group,
    xi,
    priors,
    param_names,
    visible_indices,
    group_indices,
    group_names_list,
    theta_init,
    n_warmup,
    n_samples,
    rng_key,
):
    """
    Adapt a NUTS sampler in unbounded space and return samples in theta-space.
 
    Parameters
    ----------
    models_per_group : sequence of callable
        Neural network model associated with each parameter group.
    xi : jax.Array
        Simulation summary or observation features supplied to each group
        model.
    priors : dict
        Dictionary containing the prior specification for each parameter
        (all of type ``"uniform"``).
    param_names : sequence of str
        Parameter names, in the order of the flat parameter vector.
    visible_indices : sequence of array-like
        Indices defining the parameters visible to each group.
    group_indices : sequence of array-like
        Indices defining the parameters belonging to each group.
    group_names_list : sequence of list
        Parameter names associated with each group.
    theta_init : jax.Array
        Initial parameter vector inside the prior box (e.g. a prior draw).
    n_warmup : int
        Number of warmup steps used for NUTS adaptation.
    n_samples : int
        Number of posterior samples to generate after warmup.
    rng_key : jax.Array
        JAX pseudo-random number generator key.
 
    Returns
    -------
    rng_key : jax.Array
        Updated JAX random key after warmup and posterior sampling.
    theta_samples : jax.Array
        Posterior samples in the original (bounded) parameter space, with
        leading dimension `n_samples`.
    z_samples : jax.Array
        Posterior samples in the unbounded space, with leading dimension
        `n_samples`.
    """
    to_theta, to_z, log_det_jac = make_box_transform(priors, param_names)
 
    log_prob_unbounded = make_log_prob_unbounded(
        to_theta,
        log_det_jac,
        models_per_group,
        xi,
        priors,
        visible_indices,
        group_indices,
        group_names_list,
    )
 
    z_init = to_z(theta_init)
 
    rng_key, z_samples = sample_posterior(
        log_prob_unbounded, n_warmup, n_samples, z_init, rng_key
    )
    theta_samples = jax.vmap(to_theta)(z_samples)
 
    return rng_key, theta_samples, z_samples

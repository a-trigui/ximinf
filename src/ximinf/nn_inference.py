# Standard
import os
import json

# JAX and Flax (new NNX API)
import jax  # Automatic differentiation library
import jax.numpy as jnp  # Numpy for JAX
from functools import partial
from flax import nnx

# Checkpointing
import orbax.checkpoint as ocp  # Checkpointing library
ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')
import pathlib  # File path handling library

# Miscellaneous
from IPython.display import clear_output  # To display JAX trees in Jupyter

# Other libraries
import blackjax  # MCMC library with JAX support

# Modules
import ximinf.nn_train as nntr

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

def log_prob_fn(theta, model, xy_noise):
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

    lp = log_prior(theta)
    lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)
    xy_flat = xy_noise.squeeze()
    inp = jnp.concatenate([xy_flat, theta])[None, :]
    logits = model(inp)
    p = jax.nn.sigmoid(logits).squeeze()
    p = jnp.clip(p, 1e-6, 1 - 1e-6)
    log_like = jnp.log(p) - jnp.log1p(-p)
    return lp + log_like

def sample_reference_point(rng_key, bounds, ndim):
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

def sample_posterior(log_prob, n_warmup, n_samples, init_position, rng_key):
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (warmup_state, params), _ = warmup.run(warmup_key, init_position, num_steps=n_warmup)
    kernel = blackjax.nuts(log_prob, **params).step
    rng_key, sample_key = jax.random.split(rng_key)
    states = inference_loop(sample_key, kernel, warmup_state, n_samples)
    return rng_key, states.position

# ========== JIT‐compiled per‐sample step ==========
@partial(jax.jit, static_argnums=(3, 4))
def one_sample_step(rng_key, model, minmax, xi, theta_star, n_warmup, n_samples):
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
    rng_key, theta_r0 = sample_reference_point(rng_key)

    def log_post(theta):
        return log_prob_fn(theta, model, xi)
    
    a_min, a_max, b_min, b_max = minmax

    # Run MCMC
    rng_key, posterior = sample_posterior(log_post, n_warmup, n_samples, theta_star, rng_key)

    # Un-normalize reference and chain samples
    span = jnp.array([a_max - a_min,
                      b_max - b_min])
    theta_r = theta_r0 #* span + jnp.array([a_min, b_min])
    theta_star_un = theta_star #* span + jnp.array([a_min, b_min])
    posterior_un = posterior #* span + jnp.array([a_min, b_min])

    # Compute e-c-p component using full NDIM norm
    d_star = distance(theta_star_un, theta_r)
    # compute distances for all samples in one call
    d_samples = jnp.linalg.norm(posterior_un - theta_r, axis=1)
    f_val = jnp.mean(d_samples < d_star)

    return rng_key, f_val, posterior_un

def batched_one_sample_step(rng_keys, x_batch, theta_star_batch, n_warmup, n_samples):
    """
    Vectorized wrapper over `one_sample_step` using jax.vmap.

    Parameters
    ----------
    rng_keys : jnp.ndarray
        Array of PRNGKeys, one per batch element.
    x_batch : jnp.ndarray
        Batched input data, shape (N, D).
    theta_star_batch : jnp.ndarray
        Batched ground-truth parameters, shape (N, NDIM).
    n_warmup : int
        Number of warmup steps for MCMC.
    n_samples : int
        Number of posterior samples per run.

    Returns
    -------
    tuple
        - rng_keys_out : jnp.ndarray, updated RNG keys.
        - f_vals : jnp.ndarray, shape (N,), scalar per-sample coverage indicators.
        - posterior_uns : jnp.ndarray, shape (N, n_samples, NDIM), unnormalized posterior samples.
    """

    # vmap over one_sample_step: (rng, x, theta) -> (rng_out, f_val, posterior_un)
    return jax.vmap(
        lambda rng, x, theta: one_sample_step(rng, x[None, :], theta, n_warmup, n_samples),
        in_axes=(0, 0, 0)
    )(rng_keys, x_batch, theta_star_batch)

def load_nn(path):
    # Define the checkpoint directory
    ckpt_dir = os.path.abspath(path)
    ckpt_dir = pathlib.Path(ckpt_dir).resolve()

    # Ensure the folder is removed before saving
    if ckpt_dir.exists()==False:
        # Make an error
        raise ValueError(f"Checkpoint directory {ckpt_dir} does not exist. Please check the path.")
    
    # Load model configuration
    config_path = ckpt_dir / 'config.json'
    if not config_path.exists():
        raise ValueError("Model config file not found in checkpoint directory.")
    
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    Nsize_p = model_config['Nsize_p']
    Nsize_r = model_config['Nsize_r']
    phi_batch = model_config['phi_batch']

    # 1. Re-create the checkpointer
    checkpointer = ocp.StandardCheckpointer()

    # Split the model into GraphDef (structure) and State (parameters + buffers)
    abstract_model = nnx.eval_shape(lambda: nntr.DeepSetClassifier(0.05, Nsize_p, Nsize_r, phi_batch, rngs=nnx.Rngs(0)))
    abs_graphdef, abs_rngkey, abs_rngcount, _ = nnx.split(abstract_model, nnx.RngKey, nnx.RngCount, ...)

    # 3. Restore
    state_restored = checkpointer.restore(ckpt_dir / 'state')
    #jax.tree.map(np.testing.assert_array_equal, abstract_state, state_restored)
    print('NNX State restored: ')
    nnx.display(state_restored)

    model = nnx.merge(abs_graphdef, abs_rngkey, abs_rngcount, state_restored)

    nnx.display(model)

    return model
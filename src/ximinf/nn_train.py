# Standard and scientific
import os
import json
import numpy as np  # Numerical Python
import scipy as sp

# JAX and Flax (new NNX API)
import jax  # Automatic differentiation library
import jax.numpy as jnp  # Numpy for JAX
from flax import nnx  # The Flax NNX API

# Optimization
import optax  # Optimisers for JAX

# Checkpointing
import orbax.checkpoint as ocp  # Checkpointing library
ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')

# Cosmology
from astropy.cosmology import Planck18

def rm_cosmo(z, magobs, magabs, ref_mag=19.3, z_max=0.1, n_grid=100_000):
    """
    Interpolate Planck18 distance modulus and compute residuals to the cosmology
    
    Parameters
    ----------
    z : array-like (JAX array)
        Redshift values of the dataset.
    magobs : array-like (JAX array)
        Observed magnitudes.
    magabs : array-like (JAX array)
        Absolute magnitudes.
    ref_mag : float, optional
        Reference magnitude to normalize magnitudes (default=19.3).
    z_max : float, optional
        Maximum redshift for interpolation grid (default=0.2).
    n_grid : int, optional
        Number of points in the interpolation grid (default=1_000_000).

    Returns
    -------
    mu_planck18 : jax.numpy.ndarray
        Interpolated distance modulus.
    magobs_corr : jax.numpy.ndarray
        Observed magnitudes corrected for cosmology.
    magabs_corr : jax.numpy.ndarray
        Absolute magnitudes corrected for cosmology.
    """
    print('Building Planck18 interpolation...')
    z_grid = np.linspace(1e-12, z_max, n_grid)
    mu_grid = Planck18.distmod(z_grid).value
    mu_interp_fn = sp.interpolate.interp1d(z_grid, mu_grid, kind='linear', bounds_error=False, fill_value='extrapolate')
    print('... done')

    print('Interpolating mu for dataset...')
    mu_np = mu_interp_fn(np.array(z))
    mu_planck18 = jnp.array(mu_np)
    print('... done')

    magobs_corr = magobs - mu_planck18 + ref_mag
    magabs_corr = magabs + ref_mag

    return mu_planck18, magobs_corr, magabs_corr


def gaussian(x, mu, sigma):
    """
    Compute the normalized Gaussian function.
    """
    prefactor = 1 / (np.sqrt(2 * np.pi * sigma**2))
    exponent = np.exp(-((x - mu)**2) / (2 * sigma**2))
    return prefactor * exponent

# linear model
def linear(x,a,b): 
    """
    Linear model: y = a * x + b

    Parameters
    ----------
    x : array-like
        Input values.
    a : float
        Slope of the line.
    b : float
        Intercept of the line.

    Returns
    -------
    array-like
        Output of the linear model applied to x.
    """
    return a*x + b

# Jax LHS sampler
def lhs_jax(key, n_dim, n):
    """
    Generate Latin Hypercube Samples (LHS) in [0, 1]^n_dim using JAX.

    Each of the `n_dim` dimensions is divided into `n` strata, and 
    points are randomly placed within each stratum to ensure space-filling coverage.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for reproducibility.
    n_dim : int
        Number of dimensions (features).
    n : int
        Number of samples.

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n, n_dim) with values in [0, 1], representing the LHS sample.
    """

    # Create a matrix of shape (n, n_dim) where each column is a permutation of 0..n-1
    keys = jax.random.split(key, n_dim)
    perms = [jax.random.permutation(k, n) for k in keys]
    bins = jnp.stack(perms, axis=1).astype(jnp.float32)  # shape (n, n_dim)
    
    # Now jitter inside each bin
    key, subkey = jax.random.split(key)
    jitter = jax.random.uniform(subkey, (n, n_dim))
    
    return (bins + jitter) / n

# Jax train-test split 
def train_test_split_jax(X, y, test_size=0.3, shuffle=False, key=None):
    """
    Split arrays into random train and test subsets using JAX.

    Parameters
    ----------
    X : jax.numpy.ndarray
        Input features of shape (N, ...).
    y : jax.numpy.ndarray
        Corresponding labels of shape (N, ...).
    test_size : float, optional
        Fraction of the dataset to use as test data (default is 0.25).
    shuffle : bool, optional
        Whether to shuffle the data before splitting (default is False).
    key : jax.random.PRNGKey, optional
        Random key used for shuffling (required if shuffle=True).

    Returns
    -------
    X_train : jax.numpy.ndarray
        Training subset of inputs.
    X_test : jax.numpy.ndarray
        Test subset of inputs.
    y_train : jax.numpy.ndarray
        Training subset of labels.
    y_test : jax.numpy.ndarray
        Test subset of labels.
    """

    N = X.shape[0]
    N_test = int(jnp.floor(test_size * N))
    N_train= N - N_test

    if shuffle:
        perm = jax.random.permutation(key, N)
        X, y = X[perm], y[perm]

    return X[:N_train], X[N_train:], y[:N_train], y[N_train:]

@nnx.jit
def l2_loss(model, alpha):
    """
    Compute L2 regularization loss for model parameters.

    Parameters
    ----------
    params : list
        List of model parameters (weights and biases).
    alpha : float
        Regularization coefficient (penalty term).

    Returns
    -------
    float
        L2 regularization loss.
    """
    params_tree = nnx.state(model, nnx.Param)
    params = jax.tree.leaves(params_tree)

    return alpha * sum((param ** 2).sum() for param in params)

@nnx.jit
def loss_fn(model, batch, l2_reg=1e-5):
    """
    Compute the total loss, which is the sum of the data loss and L2 regularization.

    Parameters
    ----------
    model : nn.Module
        The neural network model to compute predictions.
    batch : tuple
        A tuple containing the input batch `x_batch` and corresponding `labels`.
    l2_reg : float, optional
        The regularization coefficient for L2 regularization (default is 1e-5).

    Returns
    -------
    tuple
        A tuple containing:
        - float: the total loss (data loss + L2 regularization)
        - array: the predicted logits
    """

    x_batch, labels = batch
    logits = model(x_batch)
    data_loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()

    # Compute l2 regularisation loss
    l2 = l2_loss(model, alpha=l2_reg)

    loss = data_loss + l2
    return loss, logits

@nnx.jit
# Define the accuracy function
def accuracy_fn(model, batch):
    """
    Compute accuracy by comparing predicted and true labels.

    Parameters
    ----------
    model : nn.Module
        The neural network model to compute predictions.
    batch : tuple
        A tuple containing the input batch `x_batch` and corresponding `labels`.

    Returns
    -------
    float
        Accuracy score (proportion of correct predictions).
    """

    x_batch, labels = batch
    logits = model(x_batch)  # Ensure shape matches labels
    preds = (jax.nn.sigmoid(logits) > 0.5)
    comp = labels > 0.5
    accuracy = jnp.mean(preds == comp)
    return accuracy

@nnx.jit
def train_step(model, optimizer: nnx.Optimizer, batch):
    """
    Perform a single training step: compute gradients and update model parameters.

    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    optimizer : nnx.Optimizer
        The optimizer used to update model parameters.
    batch : tuple
        A tuple containing the input batch `x_batch` and corresponding `labels`.
    """

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)

    # Update optimizer (in-place)
    optimizer.update(grads)

@nnx.jit
def pred_step(model, x_batch):
    """
    Perform a prediction step: compute model logits for a given input batch.

    Parameters
    ----------
    model : nn.Module
        The model used for prediction.
    x_batch : array-like
        Input data batch for which predictions are to be made.

    Returns
    -------
    array
        The model's logits for the input batch.
    """
  
    logits = model(x_batch)
    return logits

class Phi(nnx.Module):
    def __init__(self, Nsize, n_cols, *, rngs):
        self.linear1 = nnx.Linear(n_cols, Nsize, rngs=rngs)
        self.linear2 = nnx.Linear(Nsize, Nsize, rngs=rngs)

    def __call__(self, x):
        h = nnx.relu(self.linear1(x))
        h = nnx.relu(self.linear2(h))
        return h


class Rho(nnx.Module):
    def __init__(self, Nsize_p, Nsize_r, n_params, *, rngs):
        self.linear1 = nnx.Linear(Nsize_p + n_params, Nsize_r, rngs=rngs)
        self.linear2 = nnx.Linear(Nsize_r, 1, rngs=rngs)

    def __call__(self, dropout, pooled_features, theta):
        x = jnp.concatenate([pooled_features, theta], axis=-1)
        x = nnx.relu(self.linear1(x))
        x = dropout(x)
        return self.linear2(x)



class DeepSetClassifier(nnx.Module):
    def __init__(self, dropout_rate, Nsize_p, Nsize_r,
                 n_cols, n_params, *, rngs):

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.n_cols   = n_cols
        self.n_params = n_params

        self.phi = Phi(Nsize_p, n_cols, rngs=rngs)
        self.rho = Rho(Nsize_p, Nsize_r, n_params, rngs=rngs)

    def __call__(self, input_data):
        N, input_dim = input_data.shape

        # Compute M first from input size
        # Total input columns = M*n_cols + n_params + M (mask)
        M = (input_dim - self.n_params) // (self.n_cols + 1)

        # Reshape data columns
        data = input_data[:, :M*self.n_cols].reshape(N, M, self.n_cols)

        # Slice mask (last M columns)
        mask = input_data[:, -M-self.n_params:-self.n_params]         # shape (N, M)

        # Parameters
        theta = input_data[:, -self.n_params:]  # shape (N, n_params)

        # print(theta)

        # Apply Phi
        h = self.phi(self.dropout, data)

        # Apply mask
        h_masked = h * mask[..., None]

        # Pool (masked average)mask_sum = jnp.sum(mask, axis=1, keepdims=True)
        mask_sum = jnp.sum(mask, axis=1, keepdims=True)
        mask_sum = jnp.where(mask_sum == 0, 1.0, mask_sum)
        pooled = jnp.sum(h_masked, axis=1) / mask_sum # Try jnp.sqrt(mask_sum) ?

        # Apply Rho
        return self.rho(self.dropout, pooled, theta)



def save_nn(model, path, model_config):
    ckpt_dir = os.path.abspath(path)
    ckpt_dir = ocp.test_utils.erase_and_create_empty(ckpt_dir)

    # Split the model into GraphDef (structure) and State (parameters + buffers)
    _, _, _, state = nnx.split(model, nnx.RngKey, nnx.RngCount, ...)

    # Display for debugging (optional)
    # nnx.display(state)

    # Initialize the checkpointer
    checkpointer = ocp.StandardCheckpointer()

    # Save State (parameters & non-trainable variables)
    checkpointer.save(ckpt_dir / 'state', state)

    # Save model configuration for later loading
    with open(ckpt_dir / 'config.json', 'w') as f:
        json.dump(model_config, f)
# Standard and scientific
import os
import json
import numpy as np  # Numerical Python
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import clear_output

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

def rm_cosmo(z, magobs, ref_mag=19.3, z_max=0.1, n_grid=100_000):
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

    return mu_planck18, magobs_corr


def gaussian(x, mu, sigma):
    """
    Compute the normalized Gaussian function.

    Parameters
    ----------
    x : array-like
        Input values.
    mu : float
        Mean of the Gaussian.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    array-like
        The values of the Gaussian function evaluated at x.
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

def train_loop(model,
               optimizer,
               train_data,
               train_labels,
               test_data,
               test_labels,
               key,
               epochs,
               batch_size,
               patience,
               metrics_history,
               M,
               N,
               plot_flag=False):
    """
    Train loop with early stopping and optional plotting.
    """

    # Initialise stopping criteria
    best_train_loss = jnp.inf
    best_test_loss = jnp.inf
    strikes = 0

    model.train()

    for epoch in range(epochs):
        # Shuffle the training data using JAX.
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(train_data))
        train_data = train_data[perm]
        train_labels = train_labels[perm]
        del perm
        
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for i in range(0, len(train_data), batch_size):
            # Get the current batch of data and labels
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            
            # Perform a training step
            loss, _ = loss_fn(model, (batch_data, batch_labels))
            accuracy = accuracy_fn(model, (batch_data, batch_labels))
            epoch_train_loss += loss
            # Multiply batch accuracy by batch size to get number of correct predictions
            epoch_train_correct += accuracy * len(batch_data)
            epoch_train_total += len(batch_data)
            train_step(model, optimizer, (batch_data, batch_labels))
        
        # Log the training metrics.
        current_train_loss = epoch_train_loss / (len(train_data) / batch_size)
        metrics_history['train_loss'].append(current_train_loss)
        # Compute overall epoch accuracy
        metrics_history['train_accuracy'].append(epoch_train_correct / epoch_train_total)

        epoch_test_loss = 0
        epoch_test_correct = 0
        epoch_test_total = 0

        # Compute the metrics on the test set using the same batching as training
        for i in range(0, len(test_data), batch_size):
            batch_data = test_data[i:i+batch_size]
            batch_labels = test_labels[i:i+batch_size]

            loss, _ = loss_fn(model, (batch_data, batch_labels))
            accuracy = accuracy_fn(model, (batch_data, batch_labels))
            epoch_test_loss += loss
            epoch_test_correct += accuracy * len(batch_data)
            epoch_test_total += len(batch_data)

        # Log the test metrics.
        current_test_loss = epoch_test_loss / (len(test_data) / batch_size)
        metrics_history['test_loss'].append(current_test_loss)
        metrics_history['test_accuracy'].append(epoch_test_correct / epoch_test_total)
        
        # Early Stopping Check
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss  # Update best test loss
            strikes = 0
        elif current_train_loss >= best_train_loss:
            strikes = 0
        elif current_test_loss > best_test_loss and current_train_loss < best_train_loss:
            strikes += 1
        elif current_train_loss < best_train_loss:
            best_train_loss = current_train_loss # Update best train loss

        if strikes >= patience:
            print(f"\n Early stopping at epoch {epoch+1} due to {patience} consecutive increases in loss gap \n")
            break

        # Plotting (optional)
        if plot_flag and epoch % 1 == 0:
            clear_output(wait=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Loss subplot
            ax1.set_title(f'Loss for M:{M} and N:{N}')
            for dataset in ('train', 'test'):
                ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
            ax1.legend()
            ax1.set_yscale("log")

            # Accuracy subplot
            ax2.set_title('Accuracy')
            for dataset in ('train', 'test'):
                ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
            ax2.legend()

            plt.show()

    return model, metrics_history, key


def save_nn(model, path, model_config):
    """
    Save a neural network model to a checkpoint.

    Parameters
    ----------
    model : nnx.Module
        The model to save.
    path : str
        Path to the checkpoint directory.
    model_config : dict
        Configuration dictionary for the model.
    """
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
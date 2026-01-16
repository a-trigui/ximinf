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

def train_test_split_indices_jax(N, test_size=0.3, shuffle=False, key=None, fixed_test_idx=None):
    """
    Generate train/test indices in JAX, optionally using a fixed test set.

    Parameters
    ----------
    N : int
        Total number of samples.
    test_size : float
        Fraction of the dataset to use as test data.
    shuffle : bool
        Whether to shuffle before splitting (ignored if fixed_test_idx is provided).
    key : jax.random.PRNGKey
        Random key used for shuffling (required if shuffle=True and fixed_test_idx is None).
    fixed_test_idx : jax.numpy.ndarray, optional
        Predefined indices to use as test set (persistent across rounds).

    Returns
    -------
    train_idx : jax.numpy.ndarray
        Indices for the training set.
    test_idx : jax.numpy.ndarray
        Indices for the test set.
    """

    N_test = int(jnp.floor(test_size * N))

    if fixed_test_idx is None:
        if shuffle:
            perm = jax.random.permutation(key, N)
        else:
            perm = jnp.arange(N)
        test_idx = perm[:N_test]
    else:
        test_idx = fixed_test_idx

    train_idx = jnp.setdiff1d(jnp.arange(N), test_idx)
    return train_idx, test_idx


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
    """
    Neural network module for the Phi network in a Deep Set architecture.
    """
    def __init__(self, Nsize, n_cols, *, rngs):
        self.linear1 = nnx.Linear(n_cols, Nsize, rngs=rngs) #+n_params
        self.linear2 = nnx.Linear(Nsize, Nsize, rngs=rngs)
        self.linear3 = nnx.Linear(Nsize, Nsize, rngs=rngs)

    def __call__(self, data):
        h = data
        
        h = nnx.relu(self.linear1(h))
        h = nnx.relu(self.linear2(h))
        h = nnx.relu(self.linear3(h))
        return h


class Rho(nnx.Module):
    """
    Neural network module for the Rho network in a Deep Set architecture
    with separate LayerNorm for pooled features and theta.
    """
    def __init__(self, Nsize_p, Nsize_r, N_size_params, *, rngs):
        self.linear1 = nnx.Linear(Nsize_p + N_size_params, Nsize_r, rngs=rngs) #
        self.linear2 = nnx.Linear(Nsize_r, Nsize_r, rngs=rngs)
        self.linear3 = nnx.Linear(Nsize_r, 1, rngs=rngs)

    def __call__(self, dropout, pooled_features, params):
        # Concatenate pooled features and embedding
        x = jnp.concatenate([pooled_features, params], axis=-1)

        x = nnx.relu(self.linear1(x))
        x = dropout(x)

        x = nnx.relu(self.linear2(x)) #leaky_relu
        x = dropout(x)

        return self.linear3(x)


class DeepSetClassifier(nnx.Module):
    """
    Deep Set Classifier model combining Phi and Rho networks.
    """
    def __init__(self, dropout_rate, Nsize_p, Nsize_r,
                 n_cols, n_params, *, rngs):

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.n_cols   = n_cols
        self.n_params = n_params

        self.phi = Phi(Nsize_p, n_cols, rngs=rngs)
        self.rho = Rho(Nsize_p, Nsize_r, n_params, rngs=rngs)

    def __call__(self, input_data):
        # ----------------------------------------------------
        # Accept both shape (N, D) and (D,) without failing
        # ----------------------------------------------------
        if input_data.ndim == 1:
            input_data = input_data[None, :]

        N = input_data.shape[0]
        input_dim = input_data.shape[1]

        # Compute M first from input size
        # Total input columns = M*n_cols + n_params + M (mask)
        M = (input_dim - self.n_params) // (self.n_cols + 1)

        # Reshape data columns
        data = input_data[:, :M*self.n_cols].reshape(N, M, self.n_cols)

        # Slice mask (last M columns)
        mask = input_data[:, -M-self.n_params:-self.n_params]         # shape (N, M)

        # Parameters
        theta = input_data[:, -self.n_params:]  # shape (N, n_params)

        # Apply Phi
        h = self.phi(data)

        # Apply mask
        h_masked = h * mask[..., None]

        # Pool (masked average)
        mask_sum = jnp.sum(mask, axis=1, keepdims=True)
        mask_sum = jnp.where(mask_sum == 0, 1.0, mask_sum)
        pooled = jnp.sum(h_masked, axis=1) / mask_sum

        # pooled_N = jnp.concatenate([pooled, mask_sum], axis=-1)

        # Apply Rho
        return self.rho(self.dropout, pooled, theta)

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
               cpu,
               gpu,
               group_id,
               group_params,
               plot_flag=False):
    """
    Train loop with early stopping and optional plotting.
    """

    # Initialise stopping criteria
    best_train_loss = jnp.inf
    best_test_loss = jnp.inf
    best_train_accuracy = 0.0
    best_test_accuracy = 0.0
    strikes = 0

    model.train()

    for epoch in range(epochs):
        
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        
        for i in range(0, len(train_data), batch_size):
            # Get the current batch of data and labels
            batch_data = jax.device_put(train_data[i:i+batch_size], gpu)
            batch_labels = jax.device_put(train_labels[i:i+batch_size], gpu)
            
            # Perform a training step
            loss, _ = loss_fn(model, (batch_data, batch_labels))
            accuracy = accuracy_fn(model, (batch_data, batch_labels))
            epoch_train_loss += loss
            # Multiply batch accuracy by batch size to get number of correct predictions
            epoch_train_accuracy += accuracy * len(batch_data)
            train_step(model, optimizer, (batch_data, batch_labels))
        
        # Log the training metrics.
        current_train_loss = epoch_train_loss / (len(train_data) / batch_size)
        current_train_accuracy = epoch_train_accuracy / len(train_data)
        metrics_history['train_loss'].append(current_train_loss)
        # Compute overall epoch accuracy
        metrics_history['train_accuracy'].append(current_train_accuracy)

        epoch_test_loss = 0
        epoch_test_accuracy = 0

        # Compute the metrics on the test set using the same batching as training
        for i in range(0, len(test_data), batch_size):
            batch_data = jax.device_put(test_data[i:i+batch_size], gpu)
            batch_labels = jax.device_put(test_labels[i:i+batch_size], gpu)

            loss, _ = loss_fn(model, (batch_data, batch_labels))
            accuracy = accuracy_fn(model, (batch_data, batch_labels))
            epoch_test_loss += loss
            epoch_test_accuracy += accuracy * len(batch_data)

        # Log the test metrics.
        current_test_loss = epoch_test_loss / (len(test_data) / batch_size)
        current_test_accuracy = epoch_test_accuracy / len(test_data)
        metrics_history['test_loss'].append(current_test_loss)
        metrics_history['test_accuracy'].append(current_test_accuracy)
        
        # Early Stopping Check
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss  # Update best test loss
            strikes = 0
        # elif current_test_accuracy > best_test_accuracy:
        #     best_test_accuracy = current_test_accuracy  # Update best test accuracy
        #     strikes = 0
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

            print(f"=== Training model for group {group_id}: {group_params} ===")

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

        if epoch == epochs-1:
            print(f"\n Reached maximum epochs: {epochs} \n")

    return model, metrics_history, key

def save_autoregressive_nn(models_per_group, path, model_config):
    """
    Save an autoregressive stack of NNX models.

    Parameters
    ----------
    models_per_group : list[nnx.Module]
        One model per autoregressive group.
    path : str
        Checkpoint directory.
    model_config : dict
        Full model configuration (shared + per-group).
    """
    ckpt_dir = os.path.abspath(path)
    ckpt_dir = ocp.test_utils.erase_and_create_empty(ckpt_dir)

    checkpointer = ocp.StandardCheckpointer()

    for g, model in enumerate(models_per_group):
        # Split model into graph-independent state
        _, _, _, state = nnx.split(model, nnx.RngKey, nnx.RngCount, ...)
        checkpointer.save(ckpt_dir / f"state_group_{g}", state)

    # Save configuration
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)

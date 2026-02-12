# Standard and scientific
import os
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output
import subprocess
import numpy as np

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

def rm_cosmo(z, magobs, ref_mag=19.3, package='cosmologix'):
    """
    Compute distance modulus and residuals directly at dataset redshifts.

    Parameters
    ----------
    z : array-like (JAX array)
        Redshift values of the dataset.
    magobs : array-like (JAX array)
        Observed magnitudes.
    ref_mag : float, optional
        Reference magnitude to normalize magnitudes (default=19.3).

    Returns
    -------
    mu_planck18 : jax.numpy.ndarray
        Distance modulus at the dataset redshifts.
    magobs_corr : jax.numpy.ndarray
        Observed magnitudes corrected for cosmology.
    """
    # Direct evaluation
    if package == 'astropy':
        z_np = np.array(z)
        mu_planck18 = jnp.array(Planck18.distmod(z_np).value)
    elif package == 'cosmologix':
        from cosmologix import distances, parameters
        mu_planck18 = distances.mu(parameters.get_cosmo_params('Planck18'), z, dtype=jnp.float32)
    else:
        raise ValueError('The distance modulus must be calculated using either astropy or cosmologix')

    # Correct observed magnitudes
    magobs_corr = magobs - mu_planck18 + ref_mag

    return mu_planck18, magobs_corr

def normalize(data_dict, stats_dict):
    normed = {}
    for k, v in data_dict.items():
        if k in stats_dict:
            mu = stats_dict[k]['mu']
            sigma = stats_dict[k]['sigma']
            normed[k] = (v - mu) / sigma
        else:
            normed[k] = v  # leave untouched
    return normed

def unnormalize(normed_params, param_stats):
    unnormed = {}
    for k, v in normed_params.items():
        if k in param_stats:
            mu = param_stats[k]['mu']
            sigma = param_stats[k]['sigma']
            unnormed[k] = v * sigma + mu  # inverse of normalization
        else:
            unnormed[k] = v  # leave untouched
    return unnormed

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
def loss_fn(model, batch):
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

    loss = data_loss
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
    def __init__(self, Nsize, n_cols, n_params, *, rngs):
        self.linear1 = nnx.Linear(n_cols + n_params, 2*Nsize, use_bias=False, rngs=rngs)
        self.ln1     = nnx.LayerNorm(2*Nsize, rngs=rngs)
        self.linear2 = nnx.Linear(2*Nsize, 2*Nsize, use_bias=False, rngs=rngs)
        self.ln2     = nnx.LayerNorm(2*Nsize, rngs=rngs)
        self.linear3 = nnx.Linear(2*Nsize, 2*Nsize, use_bias=False, rngs=rngs)
        self.ln3     = nnx.LayerNorm(2*Nsize, rngs=rngs)
        self.linear4 = nnx.Linear(2*Nsize, Nsize, rngs=rngs)

    def __call__(self, data, params):
        h = jnp.concatenate([data, params], axis=-1)

        h = self.linear1(h)
        h = self.ln1(h)
        h = nnx.relu(h)

        h = self.linear2(h)
        h = self.ln2(h)
        h = nnx.relu(h)

        h = self.linear3(h)
        h = self.ln3(h)
        h = nnx.relu(h)

        h = self.linear4(h)

        return h

class Rho(nnx.Module):
    """
    Neural network module for the Rho network in a Deep Set architecture
    with separate LayerNorm for pooled features and theta.
    """
    def __init__(self, Nsize_p, Nsize_r, N_size_params, *, rngs):
        self.linear1 = nnx.Linear(Nsize_p + N_size_params, Nsize_r, use_bias=False, rngs=rngs)
        self.ln1     = nnx.LayerNorm(Nsize_r, rngs=rngs)
        self.linear2 = nnx.Linear(Nsize_r, Nsize_r, use_bias=False, rngs=rngs)
        self.ln2     = nnx.LayerNorm(Nsize_r, rngs=rngs)
        self.linear3 = nnx.Linear(Nsize_r, Nsize_r, use_bias=False, rngs=rngs)
        self.ln3     = nnx.LayerNorm(Nsize_r, rngs=rngs)
        self.linear4 = nnx.Linear(Nsize_r, 1, rngs=rngs)

    def __call__(self, dropout, pooled_features, params):
        # Concatenate pooled features and embedding
        x = jnp.concatenate([pooled_features, params], axis=-1)

        x = self.linear1(x)
        x = self.ln1(x)
        x = nnx.relu(x)
        x = dropout(x)

        x = self.linear2(x)
        x = self.ln2(x)
        x = nnx.relu(x)
        x = dropout(x)

        x = self.linear3(x)
        x = self.ln3(x)
        x = nnx.relu(x)
        x = dropout(x)

        return self.linear4(x)

class DeepSetClassifier(nnx.Module):
    """
    Deep Set Classifier model combining Phi and Rho networks.
    """
    def __init__(self, dropout_rate, Nsize_p, Nsize_r,
                 n_cols, n_params, *, rngs):

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.n_cols   = n_cols
        self.n_params = n_params

        self.phi = Phi(Nsize_p, n_cols, n_params, rngs=rngs)
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

        theta_fill = jnp.broadcast_to(theta[:, None, :], (N, M, self.n_params))

        # Apply Phi
        h = self.phi(data, theta_fill)

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

        # -------------------------------------------------
        # Gate early stopping on minimum accuracy
        # -------------------------------------------------
        if current_test_accuracy >= 0.7:
            if strikes >= patience:
                print(
                    f"\n Early stopping at epoch {epoch+1} "
                    f"(accuracy >= 0.7 and {patience} strikes) \n"
                )
                break

        # Plotting (optional)
        if plot_flag and epoch % 1 == 0:
            clear_output(wait=True)

            print(f"=== Training model for group {group_id}: {group_params} ===")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Loss subplot
            ax1.set_title(f'Loss for M:{M} and N:{N} with patience:{patience}')
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
    with open(ckpt_dir / "config.pkl", "wb") as f:
        pickle.dump(model_config, f)

def print_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    used, total = map(int, result.stdout.strip().split(','))
    print(f"GPU memory used: {used} MB / {total} MB")
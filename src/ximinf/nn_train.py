# Standard and scientific
import matplotlib.pyplot as plt
from IPython.display import clear_output
import subprocess

# JAX and Flax (new NNX API)
import jax  # Automatic differentiation library
import jax.numpy as jnp  # Numpy for JAX
from flax import nnx  # The Flax NNX API
import numpy as np

# Optimization
import optax  # Optimisers for JAX

def print_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    used, total = map(int, result.stdout.strip().split(','))
    print(f"GPU memory used: {used} MB / {total} MB")

def setup_jax_device():
    # Try GPU backends in priority order
    gpu = None
    for backend in ("METAL", "cuda", "gpu"):
        try:
            devs = jax.devices(backend)
        except RuntimeError:
            continue
        if devs:
            gpu = devs[0]
            break

    # Fallback
    cpu = jax.devices("cpu")[0]

    # Use GPU if found
    if gpu is not None and gpu.platform == "cuda":
        print_gpu_memory()
        device = gpu
    elif gpu is not None:
        device = gpu
    else:
        device = cpu

    jax.default_device(device)

    backend = jax.default_backend()
    print(backend)

    return device, backend, cpu, gpu

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

def train_loop(model,
               optimizer,
               train_data,
               train_labels,
               val_data,
               val_labels,
               key,
               epochs,
               batch_size,
               patience,
               metrics_history,
               M,
               N,
               gpu,
               group_id,
               group_params,
               plot_flag=False):
    """
    Train loop with early stopping and optional plotting.
    """

    # Initialise stopping criteria
    best_train_loss = jnp.inf
    best_val_loss = jnp.inf
    strikes = 0

    model.train()

    for epoch in range(epochs):

        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)

        train_data = train_data[perm]
        train_labels = train_labels[perm]
        
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

        model.eval()

        epoch_val_loss = 0
        epoch_val_accuracy = 0

        # Compute the metrics on the val set using the same batching as training
        for i in range(0, len(val_data), batch_size):
            batch_data = jax.device_put(val_data[i:i+batch_size], gpu)
            batch_labels = jax.device_put(val_labels[i:i+batch_size], gpu)

            loss, _ = loss_fn(model, (batch_data, batch_labels))
            accuracy = accuracy_fn(model, (batch_data, batch_labels))
            epoch_val_loss += loss
            epoch_val_accuracy += accuracy * len(batch_data)

        # Log the val metrics.
        current_val_loss = epoch_val_loss / (len(val_data) / batch_size)
        current_val_accuracy = epoch_val_accuracy / len(val_data)
        metrics_history['val_loss'].append(current_val_loss)
        metrics_history['val_accuracy'].append(current_val_accuracy)
        
        # Early Stopping Check
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss  # Update best val loss
            strikes = 0
        # elif current_val_accuracy > best_val_accuracy:
        #     best_val_accuracy = current_val_accuracy  # Update best val accuracy
        #     strikes = 0
        elif current_train_loss >= best_train_loss:
            strikes = 0
        elif current_val_loss > best_val_loss and current_train_loss < best_train_loss:
            strikes += 1
        elif current_train_loss < best_train_loss:
            best_train_loss = current_train_loss # Update best train loss

        # -------------------------------------------------
        # Gate early stopping on minimum accuracy
        # -------------------------------------------------
        if current_val_accuracy >= 0.7:
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
            for dataset in ('train', 'val'):
                ax1.plot(np.arange(1,epoch+2,1), metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
            ax1.legend()
            ax1.set_yscale("log")

            # Accuracy subplot
            ax2.set_title('Accuracy')
            for dataset in ('train', 'val'):
                ax2.plot(np.arange(1,epoch+2,1), metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
            ax2.legend()

            plt.show()
            plt.close(fig)

        if epoch == epochs-1:
            print(f"\n Reached maximum epochs: {epochs} \n")

    return model, metrics_history, key
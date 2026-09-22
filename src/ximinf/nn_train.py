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
    """
    Print the currently used and total GPU memory reported by ``nvidia-smi``.

    Returns
    -------
    None
        The memory usage is printed to standard output.

    Notes
    -----
    This function relies on the NVIDIA System Management Interface
    (``nvidia-smi``) being installed and available on the system path. It
    queries memory usage in MiB without units in the command output.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    used, total = map(int, result.stdout.strip().split(','))
    print(f"GPU memory used: {used} MB / {total} MB")

def setup_jax_device():
    """
    Detect an available JAX accelerator and set it as the default device.

    The function checks for accelerator backends in the order ``"METAL"``,
    ``"cuda"``, and ``"gpu"``. If none are available, it falls back to the
    first CPU device.

    Returns
    -------
    device : jax.Device
        Selected JAX device used as the default device.
    backend : str
        Name of the JAX backend associated with the selected default device.
    cpu : jax.Device
        First available CPU device.
    gpu : jax.Device or None
        First detected accelerator device, or ``None`` if no accelerator
        backend is available.

    Notes
    -----
    GPU memory usage is printed when a CUDA device is selected. The selected
    device is registered using ``jax.default_device`` before querying the
    default backend.
    """
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
    Compute the binary cross-entropy loss for a batch.

    Parameters
    ----------
    model : nnx.Module
        Neural network model used to compute the prediction logits.
    batch : tuple
        Tuple containing ``x_batch`` and ``labels``, where ``x_batch`` is the
        input batch and ``labels`` contains the corresponding binary targets.

    Returns
    -------
    loss : jax.Array
        Mean binary cross-entropy loss over the batch.
    logits : jax.Array
        Raw model logits for the input batch.

    Notes
    -----
    The binary cross-entropy is computed directly from the logits using
    ``optax.sigmoid_binary_cross_entropy``.
    """

    x_batch, labels = batch
    logits = model(x_batch)
    data_loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()

    loss = data_loss
    return loss, logits

@nnx.jit
def accuracy_fn(model, batch):
    """
    Compute binary classification accuracy for a batch.

    Parameters
    ----------
    model : nnx.Module
        Neural network model used to compute prediction logits.
    batch : tuple
        Tuple containing ``x_batch`` and ``labels``, where ``x_batch`` is the
        input batch and ``labels`` contains the corresponding binary targets.

    Returns
    -------
    accuracy : jax.Array
        Fraction of samples for which the predicted binary class matches the
        target class.

    Notes
    -----
    Predictions are obtained by applying a sigmoid to the model logits and
    thresholding the resulting probabilities at 0.5. Target labels are
    similarly interpreted using a threshold of 0.5.
    """

    x_batch, labels = batch
    logits = model(x_batch)  # Ensure shape matches labels
    preds = (jax.nn.sigmoid(logits) > 0.5)
    comp = labels > 0.5
    accuracy = jnp.mean(preds == comp)
    return accuracy

@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch):
    """
    Perform one optimization step on a training batch.

    Parameters
    ----------
    model : nnx.Module
        Neural network model whose parameters are optimized.
    optimizer : nnx.Optimizer
        NNX optimizer used to update the model parameters.
    batch : tuple
        Tuple containing ``x_batch`` and ``labels`` for the current training
        batch.

    Returns
    -------
    None
        The model and optimizer are updated in place.
    """

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)

    # Update optimizer (in-place)
    optimizer.update(grads)

@nnx.jit
def pred_step(model, x_batch):
    """
    Compute model logits for an input batch.

    Parameters
    ----------
    model : nnx.Module
        Neural network model used to generate predictions.
    x_batch : array-like
        Input data batch.

    Returns
    -------
    logits : jax.Array
        Raw model logits for the input batch.

    Notes
    -----
    The returned values are logits and are not passed through a sigmoid.
    """
  
    logits = model(x_batch)
    return logits

def train_loop(
    model,
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
    plot_flag=False,
):
    """
    Train a neural network with validation monitoring and early stopping.

    Parameters
    ----------
    model : nnx.Module
        Neural network model to train.
    optimizer : nnx.Optimizer
        Optimizer used to update the model parameters.
    train_data : jax.Array
        Training input data.
    train_labels : jax.Array
        Binary labels corresponding to `train_data`.
    val_data : jax.Array
        Validation input data.
    val_labels : jax.Array
        Binary labels corresponding to `val_data`.
    key : jax.Array
        JAX pseudo-random number generator key used to shuffle the training
        data at each epoch.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Number of samples processed in each batch.
    patience : int
        Number of consecutive epochs without validation-loss improvement
        allowed before early stopping, provided the validation accuracy has
        reached the minimum threshold.
    metrics_history : dict
        Dictionary containing lists used to store training and validation
        losses and accuracies. The keys ``"train_loss"``,
        ``"train_accuracy"``, ``"val_loss"``, and ``"val_accuracy"`` are
        expected.
    M : int
        Simulation or dataset parameter used for plotting and display.
    N : int
        Dataset parameter used for plotting and display.
    gpu : jax.Device
        Device to which each training and validation batch is transferred.
    group_id : int or str
        Identifier of the parameter group currently being trained.
    group_params : object
        Parameter names or configuration associated with the current group,
        used for plotting and display.
    plot_flag : bool, optional
        Whether to display training and validation loss and accuracy plots
        after each epoch. Default is ``False``.

    Returns
    -------
    model : nnx.Module
        Trained neural network model.
    metrics_history : dict
        Updated dictionary containing the training and validation metrics
        accumulated over the completed epochs.
    key : jax.Array
        Updated JAX random number generator key.
    """

    # Initialise stopping criteria
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
            epoch_train_loss += loss * len(batch_data)
            # Multiply batch accuracy by batch size to get number of correct predictions
            epoch_train_accuracy += accuracy * len(batch_data)
            train_step(model, optimizer, (batch_data, batch_labels))
        
        # Log the training metrics.
        current_train_loss = epoch_train_loss / len(train_data)
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
            epoch_val_loss += loss * len(batch_data)
            epoch_val_accuracy += accuracy * len(batch_data)

        # Log the val metrics.
        current_val_loss = epoch_val_loss / len(val_data)
        current_val_accuracy = epoch_val_accuracy / len(val_data)
        metrics_history['val_loss'].append(current_val_loss)
        metrics_history['val_accuracy'].append(current_val_accuracy)

        # Early Stopping Check V2
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss  # Update best val loss
            strikes = 0
        else:
            strikes += 1

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
            ax2.set_title(f'Accuracy : {metrics_history["val_accuracy"][-1]:.3f}')
            for dataset in ('train', 'val'):
                ax2.plot(np.arange(1,epoch+2,1), metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
            ax2.legend()

            plt.show()
            plt.close(fig)

        if epoch == epochs-1:
            print(f"\n Reached maximum epochs: {epochs} \n")

    return model, metrics_history, key
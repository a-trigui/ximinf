# Standard and scientific
# Standard and scientific
import matplotlib.pyplot as plt
from collections import deque
from IPython.display import clear_output

# JAX and Flax (new NNX API)
import jax  # Automatic differentiation library
import jax.numpy as jnp  # Numpy for JAX
from flax import nnx  # The Flax NNX API
import numpy as np

import ximinf.nn_inference as nninf

# Optimization
import optax  # Optimisers for JAX

def print_jax_memory():
    """
    Print the number, total size, and top live JAX memory allocations.

    Inspects all currently live JAX arrays and prints summary statistics
    including total count, aggregate gigabytes, and detailed shapes and
    data types for the 15 largest arrays.
    """
    arrs = jax.live_arrays()
    print(f"{len(arrs)} live arrays")
    print(f"Total: {sum(a.nbytes for a in arrs) / 1e9:.3f} GB")

    for i, a in enumerate(sorted(arrs, key=lambda x: -x.nbytes)[:15]):
        print(
            f"{i:2d}: "
            f"{a.nbytes / 1e9:.3f} GB "
            f"{a.shape} "
            f"{a.dtype}"
        )

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
    Compute sigmoid binary cross-entropy loss and logits for a batch.

    Parameters
    ----------
    model : nnx.Module
        Neural network model mapping input features to raw logits.
    batch : tuple of jax.Array
        Tuple ``(x_batch, labels)`` containing feature inputs and binary
        targets.

    Returns
    -------
    loss : jax.Array
        Scalar mean binary cross-entropy loss.
    logits : jax.Array
        Model prediction logits for the batch.
    """
    x_batch, labels = batch
    logits = model(x_batch)
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    return loss, logits

@nnx.jit
def accuracy_fn(model, batch):
    """
    Compute classification accuracy for a batch.

    Parameters
    ----------
    model : nnx.Module
        Neural network model mapping input features to raw logits.
    batch : tuple of jax.Array
        Tuple ``(x_batch, labels)`` containing feature inputs and binary
        targets.

    Returns
    -------
    accuracy : jax.Array
        Fraction of correctly classified instances in the batch.
    """
    x_batch, labels = batch
    logits = model(x_batch)
    preds = jax.nn.sigmoid(logits) > 0.5
    return (preds == (labels > 0.5)).mean()

@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch):
    """
    Perform a single optimization step using model gradients.

    Parameters
    ----------
    model : nnx.Module
        Flax NNX model to optimize.
    optimizer : nnx.Optimizer
        Flax NNX optimizer instance managing model parameters.
    batch : tuple of jax.Array
        Tuple ``(x_batch, labels)`` containing feature inputs and binary
        targets.
    """
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
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
        Raw model logits for the input batch (pre-sigmoid).
    """
  
    logits = model(x_batch)
    return logits

# ===============================================================
# Config
# ===============================================================
def default_cfg():
    """
    Return the default hyperparameter configuration for neural network training.

    Returns
    -------
    dict
        Default configuration dictionary containing optimizer parameters,
        batch size, learning rate schedule, early stopping thresholds,
        and dataset split ratios.
    """
    return dict(
        epochs=500, patience=10, lr=1e-4, batch_size=500,
        wd=1e-3, b1=0.9, decay_steps=1000, decay_rate=0.9,
        val_frac=0.15, test_frac=0.15,
        min_delta=1e-4, min_acc_for_stopping=0.7,
        plot=True, plot_step=1, swa_k=10,
    )



# ===============================================================
# 2. Train / val / test split  ->  {"train": {...}, "val": {...}, "test": {...}}
# ===============================================================
def split_dataset(arrays, key, cfg):
    """
    Split dataset arrays randomly into train, validation, and test subsets.

    Parameters
    ----------
    arrays : dict of jax.Array
        Dictionary mapping array names (such as ``"data"``, ``"mask"``,
        ``"m_norm"``, ``"true_params"``) to tensors with matching leading
        dimension.
    key : jax.Array
        JAX PRNG key used to randomly permute sample indices.
    cfg : dict
        Configuration dictionary containing split proportions ``"val_frac"``
        and ``"test_frac"``.

    Returns
    -------
    dict of dict
        Nested dictionary with keys ``"train"``, ``"val"``, and ``"test"``,
        each holding the corresponding partition of the input arrays.
    """
    N = arrays["data"].shape[0]
    perm = jax.random.permutation(key, N)
    n_test = int(cfg["test_frac"] * N)
    n_val = int(cfg["val_frac"] * N)

    idx = {
        "test": perm[:n_test],
        "val": perm[n_test:n_test + n_val],
        "train": perm[n_test + n_val:],
    }
    out = {}
    for name, ix in idx.items():
        out[name] = {k: v[ix] for k, v in arrays.items()}
        out[name]["m_norm"] = out[name]["m_norm"][:, None]
    return out


# ===============================================================
# 3. Group indices
# ===============================================================
def build_group_indices(param_groups, param_names):
    """
    Build parameter index pairs for autoregressive inference groups.

    Parameters
    ----------
    param_groups : list
        List of parameter groups defining the autoregressive ordering.
    param_names : list of str
        Global ordered list of parameter names.

    Returns
    -------
    list of tuple of (jax.Array, jax.Array)
        List of ``(current_idx, visible_idx)`` index pairs for each group.
    """
    visible_indices, group_indices = nninf.preprocess_groups(param_groups, param_names)
    return list(zip(group_indices, visible_indices))

# ===============================================================
# 4. Inputs / labels for one group
# ===============================================================
def add_noise(data, scale, key):
    """
    Add zero-mean Gaussian noise scaled by standard deviation to data.

    Parameters
    ----------
    data : jax.Array
        Input data array.
    scale : float or jax.Array
        Standard deviation of the Gaussian noise.
    key : jax.Array
        JAX PRNG key.

    Returns
    -------
    jax.Array
        Data perturbed by additive Gaussian noise.
    """
    return data + scale * jax.random.normal(key, data.shape, data.dtype)


def freeze_noise(dataset, key, names=("val", "test")):
    """
    Apply a single fixed noise realization to validation and test splits.

    Parameters
    ----------
    dataset : dict of dict
        Dataset containing split dictionaries with ``"data"`` and
        ``"noise_scale"`` entries.
    key : jax.Array
        JAX PRNG key.
    names : tuple of str, optional
        Names of splits on which to apply fixed noise. Default is
        ``("val", "test")``.

    Returns
    -------
    dict of dict
        Dataset with perturbed ``"data"`` arrays and removed
        ``"noise_scale"`` entries in the specified splits.
    """
    for name in names:
        key, k = jax.random.split(key)
        s = dataset[name]
        s["data"] = add_noise(s["data"], s["noise_scale"], k)
        del s["noise_scale"]
    return dataset


def make_base_inputs(split, key=None):
    """
    Concatenate observational data, object validity mask, and normalized count.

    Parameters
    ----------
    split : dict
        Dataset split dictionary containing ``"data"``, ``"mask"``,
        ``"m_norm"``, and optionally ``"noise_scale"``.
    key : jax.Array, optional
        JAX PRNG key. When supplied, noise is freshly drawn and added to the
        data.

    Returns
    -------
    jax.Array
        Concatenated base feature tensor along the last dimension.
    """
    data = split["data"]
    if key is not None:
        data = add_noise(data, split["noise_scale"], key)
    return jnp.concatenate([data, split["mask"], split["m_norm"]], axis=-1)


def make_resampler(split, group_idx, device):
    """
    Create a resampling function that generates fresh noisy batches and labels.

    Parameters
    ----------
    split : dict
        Dataset split dictionary containing base features and true parameters.
    group_idx : tuple of (jax.Array, jax.Array)
        Tuple of ``(current_idx, visible_idx)`` parameter index arrays.
    device : jax.Device
        JAX device onto which generated batches are moved.

    Returns
    -------
    callable
        Function taking a JAX PRNG key and returning a new ``(x, y)`` batch
        on ``device``.
    """
    def resample(key):
        key_noise, key_xy = jax.random.split(key)
        base = make_base_inputs(split, key_noise)          # fresh noise every epoch
        return make_group_xy(split, base, group_idx, key_xy, device)
    return resample


def make_group_xy(split, base, group_idx, key, device):
    """
    Generate inputs and binary classification labels for ratio estimation.

    Randomly partitions samples into positive instances (holding true
    parameters, label 1) and negative instances (whose current group parameters
    are swapped with donor samples, label 0).

    Parameters
    ----------
    split : dict
        Dataset split dictionary containing ``"true_params"``.
    base : jax.Array
        Base feature tensor for all samples.
    group_idx : tuple of (jax.Array, jax.Array)
        Tuple ``(cur, vis)`` of current parameter indices and all visible
        parameter indices up to this group.
    key : jax.Array
        JAX PRNG key.
    device : jax.Device
        Target JAX device for placing output tensors.

    Returns
    -------
    x : jax.Array
        Concatenated feature tensor containing base features and visible
        parameters.
    y : jax.Array
        Binary target tensor of shape ``(N, 1)``.
    """
    cur, vis = group_idx
    n = base.shape[0]
    assert n > 1, "need at least 2 samples so that donors are never self"
    key_split, key_donor = jax.random.split(key)

    # random positive/negative assignment
    perm = jax.random.permutation(key_split, n)
    is_pos = jnp.zeros(n, dtype=bool).at[perm[: n // 2]].set(True)

    # donor for each sample: a random cyclic shift, so no sample is its own donor
    p = jax.random.permutation(key_donor, n)
    src = jnp.zeros(n, dtype=int).at[p].set(jnp.roll(p, -1))   # src[p[i]] = p[(i+1) % n]

    true_p = split["true_params"]
    swapped = true_p.at[:, cur].set(true_p[src][:, cur])
    params = jnp.where(is_pos[:, None], true_p, swapped)

    x = jnp.concatenate([base, params[:, vis]], axis=-1)
    y = is_pos.astype(jnp.int32)[:, None]
    return jax.device_put(x, device), jax.device_put(y, device)



def make_group_data(dataset, group_idx, key, device):
    """
    Construct training resampler and fixed validation/test datasets for a group.

    Parameters
    ----------
    dataset : dict of dict
        Full dataset dictionary containing ``"train"``, ``"val"``, and
        ``"test"`` splits.
    group_idx : tuple of (jax.Array, jax.Array)
        Tuple of ``(current_idx, visible_idx)`` for the current group.
    key : jax.Array
        JAX PRNG key.
    device : jax.Device
        Target JAX device for tensor placement.

    Returns
    -------
    dict
        Dictionary containing ``"resample_train"`` callable alongside fixed
        validation and test pairs ``"val"`` and ``"test"``.
    """
    k_val, k_test = jax.random.split(key)
    fixed = {}
    for name, k in [("val", k_val), ("test", k_test)]:
        base = make_base_inputs(dataset[name])
        fixed[name] = make_group_xy(dataset[name], base, group_idx, k, device)

    return {
        "resample_train": make_resampler(dataset["train"], group_idx, device),
        "val": fixed["val"],
        "test": fixed["test"],
    }


# ===============================================================
# 5. Optimiser
# ===============================================================
def make_optimizer(model, cfg):
    """
    Instantiate an AdamW optimizer with exponential learning rate decay.

    Parameters
    ----------
    model : nnx.Module
        Neural network model whose parameters will be optimized.
    cfg : dict
        Configuration dictionary containing ``"lr"``, ``"decay_steps"``,
        ``"decay_rate"``, ``"b1"``, and ``"wd"``.

    Returns
    -------
    nnx.Optimizer
        Configured Flax NNX optimizer.
    """
    schedule = optax.exponential_decay(
        init_value=cfg["lr"],
        transition_steps=cfg["decay_steps"],
        decay_rate=cfg["decay_rate"],
    )
    return nnx.Optimizer(
        model,
        optax.adamw(learning_rate=schedule, b1=cfg["b1"],
                    weight_decay=cfg["wd"]),
    )


# ===============================================================
# 6. Training loop helpers
# ===============================================================
def run_epoch(model, x, y, batch_size, gpu, optimizer=None):
    """
    Execute one training or evaluation epoch over mini-batches.

    Parameters
    ----------
    model : nnx.Module
        Neural network model.
    x : jax.Array
        Input features tensor.
    y : jax.Array
        Target labels tensor.
    batch_size : int
        Mini-batch size.
    gpu : jax.Device
        Target JAX device for computation.
    optimizer : nnx.Optimizer, optional
        Flax NNX optimizer. If provided, gradient updates are performed;
        if ``None``, only loss and accuracy are evaluated.

    Returns
    -------
    mean_loss : float
        Mean binary cross-entropy loss over all batches.
    mean_acc : float
        Mean classification accuracy over all batches.
    """
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for i in range(0, len(x), batch_size):
        bx = jax.device_put(x[i:i + batch_size], gpu)
        by = jax.device_put(y[i:i + batch_size], gpu)

        loss, _ = loss_fn(model, (bx, by))
        acc = accuracy_fn(model, (bx, by))
        n = len(bx)
        total_loss += loss * n
        total_acc += acc * n
        total_n += n

        if optimizer is not None:
            train_step(model, optimizer, (bx, by))

    return total_loss / total_n, total_acc / total_n


def plot_history(history, epoch, M, N, patience, group_id, group_params):
    """
    Display dynamic plots of training and validation loss and accuracy.

    Parameters
    ----------
    history : dict of list
        Dictionary recording loss and accuracy trajectories per epoch.
    epoch : int
        Current epoch index (0-indexed).
    M : int
        Number of objects per simulation.
    N : int
        Number of training simulations.
    patience : int
        Early stopping patience in epochs.
    group_id : int
        Index of the active parameter group.
    group_params : list or str
        Parameter names in the active group.
    """
    clear_output(wait=True)
    print(f"=== Training model for group {group_id}: {group_params} ===")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs_axis = np.arange(1, epoch + 2)

    ax1.set_title(f"Loss for M:{M} and N:{N} with patience:{patience}")
    for name in ("train", "val"):
        ax1.plot(epochs_axis, history[f"{name}_loss"], label=f"{name}_loss")
    ax1.legend()
    ax1.set_yscale("log")

    ax2.set_title(f'Accuracy : {history["val_accuracy"][-1]:.3f}')
    for name in ("train", "val"):
        ax2.plot(epochs_axis, history[f"{name}_accuracy"], label=f"{name}_accuracy")
    ax2.legend()

    plt.show()
    plt.close(fig)


def should_stop(val_loss, val_acc, best_val_loss, strikes, cfg):
    """
    Evaluate early stopping criteria based on validation metrics.

    Parameters
    ----------
    val_loss : float
        Current epoch validation loss.
    val_acc : float
        Current epoch validation accuracy.
    best_val_loss : float
        Lowest validation loss observed across earlier epochs.
    strikes : int
        Number of consecutive epochs without sufficient loss improvement.
    cfg : dict
        Configuration containing ``"min_delta"``,
        ``"min_acc_for_stopping"``, and ``"patience"``.

    Returns
    -------
    stop : bool
        Whether early stopping conditions are met.
    new_best_val_loss : float
        Updated minimum validation loss.
    new_strikes : int
        Updated count of consecutive non-improving epochs.
    """
    if np.isinf(best_val_loss):
        improved = True                       # first epoch is never a strike
    else:
        improved = (best_val_loss - val_loss) / abs(best_val_loss) > cfg["min_delta"]

    strikes = 0 if improved else strikes + 1
    stop = val_acc >= cfg["min_acc_for_stopping"] and strikes >= cfg["patience"]
    return stop, min(best_val_loss, val_loss), strikes

def average_states(states):
    """
    Compute the uniform element-wise mean across NNX parameter state pytrees.

    Parameters
    ----------
    states : list of nnx.State
        Sequence of model parameter states collected across epochs.

    Returns
    -------
    nnx.State
        Averaged parameter state pytree.
    """
    return jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *states)


# ===============================================================
# 7. Training loop for one group
# ===============================================================
def train_loop(model, optimizer, resample_fn, val_xy, key, cfg, gpu, plot_info):
    """
    Train a single parameter group model over multiple epochs.

    Performs dynamic batch resampling, tracks loss and accuracy metrics,
    applies stochastic weight averaging (SWA) over the final checkpoints,
    and terminates upon early stopping or reaching maximum epochs.

    Parameters
    ----------
    model : nnx.Module
        Neural network model to train.
    optimizer : nnx.Optimizer
        Flax NNX optimizer.
    resample_fn : callable
        Function taking a JAX PRNG key and returning fresh ``(train_x, train_y)``
        batches for each epoch.
    val_xy : tuple of (jax.Array, jax.Array)
        Fixed validation features and labels ``(val_x, val_y)``.
    key : jax.Array
        JAX PRNG key.
    cfg : dict
        Training configuration dictionary.
    gpu : jax.Device
        Target computation device.
    plot_info : dict
        Dictionary containing ``M``, ``N``, ``group_id``, and
        ``group_params`` used for logging and real-time visualization.

    Returns
    -------
    model : nnx.Module
        Trained model with averaged weights updated and set to evaluation mode.
    history : dict of list
        Dictionary recording loss and accuracy trajectories over epochs.
    key : jax.Array
        Updated JAX PRNG key.
    """
    val_x, val_y = val_xy
    history = {"train_loss": [], "train_accuracy": [],
               "val_loss": [], "val_accuracy": []}
    best_val_loss, strikes = np.inf, 0

    snapshots = deque(maxlen=cfg.get("swa_k", 10))

    for epoch in range(cfg["epochs"]):
        key, resample_key = jax.random.split(key)

        # fresh labels (random pos/neg split, donors, noise); rows are already
        # randomly ordered with respect to the labels, so no extra shuffle needed
        train_x, train_y = resample_fn(resample_key)

        model.train()
        tr_loss, tr_acc = run_epoch(model, train_x, train_y, cfg["batch_size"], gpu, optimizer)

        model.eval()
        va_loss, va_acc = run_epoch(model, val_x, val_y, cfg["batch_size"], gpu)

        # snapshot the raw weights at the end of every epoch
        snapshots.append(jax.tree.map(jnp.copy, nnx.state(model, nnx.Param)))

        history["train_loss"].append(tr_loss)
        history["train_accuracy"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_accuracy"].append(va_acc)

        stop, best_val_loss, strikes = should_stop(va_loss, va_acc, best_val_loss, strikes, cfg)
        if stop:
            print(f"\n Early stopping at epoch {epoch + 1} "
                  f"(accuracy >= {cfg['min_acc_for_stopping']} and {cfg['patience']} strikes) \n")
            break

        if cfg["plot"] and epoch % cfg["plot_step"] == 0:
            plot_history(history, epoch, plot_info["M"], plot_info["N"],
                         cfg["patience"], plot_info["group_id"], plot_info["group_params"])

    else:
        print(f"\n Reached maximum epochs: {cfg['epochs']} \n")

    # raw_state = nnx.state(model, nnx.Param)
    swa_state = average_states(list(snapshots))
    nnx.update(model, swa_state)
    model.eval()
    return model, history, key


# ===============================================================
# 8. Loop over groups
# ===============================================================
def train_all_groups(models, param_groups, dataset, key, cfg, ctx):
    """
    Train an autoregressive sequence of models across all parameter groups.

    Generates data lazily for each group to minimize peak memory, trains each
    model using :func:`train_loop`, and aggregates metrics and test sets.

    Parameters
    ----------
    models : list of nnx.Module
        List of neural network models, one for each parameter group.
    param_groups : list
        List of parameter group definitions specifying the autoregressive order.
    dataset : dict of dict
        Full dataset containing train, validation, and test partitions.
    key : jax.Array
        JAX PRNG key.
    cfg : dict
        Training configuration dictionary.
    ctx : dict
        Context dictionary containing ``"param_names"``, ``"cpu"``,
        ``"gpu"``, ``"M"``, and ``"N"``.

    Returns
    -------
    models : list of nnx.Module
        Trained neural network models.
    histories : list of dict
        Training and validation histories for each parameter group.
    test_sets : list of tuple
        Test feature and label pairs ``(x, y)`` for each group.
    key : jax.Array
        Updated JAX PRNG key.
    """
    group_indices = build_group_indices(param_groups, ctx["param_names"])
    histories, test_sets = [], []

    for g, (group, gidx) in enumerate(zip(param_groups, group_indices)):
        print(f"\n=== Training model for group {g}: {group} ===")

        key, data_key = jax.random.split(key)
        data = make_group_data(dataset, gidx, data_key, ctx["cpu"])
        
        n_train = int(ctx["N"] * (1 - cfg["val_frac"] - cfg["test_frac"]))
        
        models[g], hist, key = train_loop(
            model=models[g],
            optimizer=make_optimizer(models[g], cfg),
            resample_fn=data["resample_train"],
            val_xy=data["val"],
            key=key,
            cfg=cfg,
            gpu=ctx["gpu"],
            plot_info=dict(M=ctx["M"], N=n_train, group_id=g, group_params=group),
        )
        histories.append(hist)
        test_sets.append(data["test"])

    return models, histories, test_sets, key
# Standard and scientific
import matplotlib.pyplot as plt
from IPython.display import clear_output

# JAX and Flax (new NNX API)
import jax  # Automatic differentiation library
import jax.numpy as jnp  # Numpy for JAX
from flax import nnx  # The Flax NNX API
import numpy as np

# Optimization
import optax  # Optimisers for JAX

def print_jax_memory():
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

# ===============================================================
# Config
# ===============================================================
def default_cfg():
    return dict(
        epochs=500, patience=10, lr=1e-4, batch_size=500,
        weight_decay=1e-3, b1=0.9, decay_steps=1000, decay_rate=0.9,
        val_frac=0.15, test_frac=0.15,
        min_delta=1e-4, min_acc_for_stopping=0.7,
        plot=True, plot_step=1,
    )


# ===============================================================
# 2. Train / val / test split  ->  {"train": {...}, "val": {...}, "test": {...}}
# ===============================================================
def split_dataset(arrays, key, cfg):
    """arrays: dict with data, mask, m_norm, true_params."""
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
    """List of (current_idx, visible_idx), one per group."""
    out, seen = [], []
    for group in param_groups:
        names = [group] if isinstance(group, str) else list(group)
        cur = jnp.array([param_names.index(n) for n in names], dtype=int)
        vis = jnp.array([param_names.index(n) for n in seen + names], dtype=int)
        out.append((cur, vis))
        seen += names
    return out

# ===============================================================
# 4. Inputs / labels for one group
# ===============================================================
def make_base_inputs(split):
    """Constant part of the input: [data, mask, m_norm]."""
    return jnp.concatenate([split["data"], split["mask"], split["m_norm"]], axis=-1)


def make_group_xy(split, base, group_idx, key, device):
    """Draw labels; negatives = current group's params taken from another simulation."""
    cur, vis = group_idx
    n = base.shape[0]
    k_lab, k_perm = jax.random.split(key)
    labels = jax.random.uniform(k_lab, (n,)) > 0.5      # True -> real params

    # cyclic shift along a random ordering -> every sample gets another sample's θ
    p = jax.random.permutation(k_perm, n)
    src = jnp.zeros(n, dtype=int).at[p].set(jnp.roll(p, 1))

    true_p = split["true_params"]
    shuffled_cur = true_p[src][:, cur]                  # same permutation for all columns of the group

    params = true_p.at[:, cur].set(
        jnp.where(labels[:, None], true_p[:, cur], shuffled_cur)
    )
    x = jnp.concatenate([base, params[:, vis]], axis=-1)
    y = labels.astype(jnp.int32)[:, None]
    return jax.device_put(x, device), jax.device_put(y, device)


def make_resampler(split, group_idx, device):
    """Returns resample(key) -> (x, y); new labels at every call."""
    base = make_base_inputs(split)

    def resample(key):
        return make_group_xy(split, base, group_idx, key, device)

    return resample


def make_group_data(dataset, group_idx, key, device):
    """
    train : resampler (labels redrawn each epoch)
    val / test : fixed (x, y), drawn once
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
    schedule = optax.exponential_decay(
        init_value=cfg["lr"],
        transition_steps=cfg["decay_steps"],
        decay_rate=cfg["decay_rate"],
    )
    return nnx.Optimizer(
        model,
        optax.adamw(learning_rate=schedule, b1=cfg["b1"],
                    weight_decay=cfg["weight_decay"]),
    )


# ===============================================================
# 6. Training loop helpers
# ===============================================================
def run_epoch(model, x, y, batch_size, gpu, optimizer=None):
    """
    One pass over (x, y) in batches. If optimizer is given, also updates the model.
    Returns (mean_loss, mean_accuracy).
    """
    total_loss, total_acc = 0.0, 0.0
    for i in range(0, len(x), batch_size):
        bx = jax.device_put(x[i:i + batch_size], gpu)
        by = jax.device_put(y[i:i + batch_size], gpu)

        loss, _ = loss_fn(model, (bx, by))
        acc = accuracy_fn(model, (bx, by))
        total_loss += loss * len(bx)
        total_acc += acc * len(bx)

        if optimizer is not None:
            train_step(model, optimizer, (bx, by))

    return total_loss / len(x), total_acc / len(x)


def plot_history(history, epoch, M, N, patience, group_id, group_params):
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
    """Returns (stop, new_best_val_loss, new_strikes)."""
    if np.isinf(best_val_loss):
        improved = True                       # first epoch is never a strike
    else:
        improved = (best_val_loss - val_loss) / abs(best_val_loss) > cfg["min_delta"]

    strikes = 0 if improved else strikes + 1
    stop = val_acc >= cfg["min_acc_for_stopping"] and strikes >= cfg["patience"]
    return stop, min(best_val_loss, val_loss), strikes


# ===============================================================
# 7. Training loop for one group
# ===============================================================
def train_loop(model, optimizer, resample_fn, val_xy, key, cfg, gpu, plot_info):
    """
    resample_fn : key -> (train_x, train_y), called at the start of every epoch.
    val_xy      : fixed (val_x, val_y).
    plot_info   : dict(M, N, group_id, group_params), only used for plotting.
    Returns (model, metrics_history, key).
    """
    val_x, val_y = val_xy
    history = {"train_loss": [], "train_accuracy": [],
               "val_loss": [], "val_accuracy": []}
    best_val_loss, strikes = np.inf, 0

    for epoch in range(cfg["epochs"]):
        key, resample_key, perm_key = jax.random.split(key, 3)

        # fresh labels + shuffle
        train_x, train_y = resample_fn(resample_key)
        perm = jax.random.permutation(perm_key, train_x.shape[0])
        train_x, train_y = train_x[perm], train_y[perm]

        model.train()
        tr_loss, tr_acc = run_epoch(model, train_x, train_y, cfg["batch_size"], gpu, optimizer)

        model.eval()
        va_loss, va_acc = run_epoch(model, val_x, val_y, cfg["batch_size"], gpu)

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

    return model, history, key


# ===============================================================
# 8. Loop over groups
# ===============================================================
def train_all_groups(models, param_groups, dataset, key, cfg, ctx):
    """
    Builds each group's data lazily (lower peak memory) and trains it.
    Returns (models, histories, test_sets, key).
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
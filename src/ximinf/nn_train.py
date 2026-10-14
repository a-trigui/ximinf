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
def loss_fn(model, batch, pos_weight):
    x_batch, labels = batch
    logits = model(x_batch)
    bce = optax.sigmoid_binary_cross_entropy(logits, labels)
    w = jnp.where(labels > 0.5, pos_weight, 1.0)
    loss = (w * bce).sum() / w.sum()
    return loss, logits

@nnx.jit
def accuracy_fn(model, batch, pos_weight):
    x_batch, labels = batch
    logits = model(x_batch)
    preds = jax.nn.sigmoid(logits) > 0.5
    comp = labels > 0.5
    w = jnp.where(comp, pos_weight, 1.0)
    return (w * (preds == comp)).sum() / w.sum()

@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch, pos_weight):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch, pos_weight)
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
        wd=1e-3, b1=0.9, decay_steps=1000, decay_rate=0.9,
        val_frac=0.15, test_frac=0.15,
        min_delta=1e-4, min_acc_for_stopping=0.7,
        plot=True, plot_step=1, swa_k=10,
        n_neg=5,          # K negatives per positive
        balance=True,     # weight positives by K so classes count equally
    )
def get_pos_weight(cfg):
    return float(cfg["n_neg"]) if cfg["balance"] else 1.0



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
def add_noise(data, scale, key):
    return data + scale * jax.random.normal(key, data.shape, data.dtype)


def freeze_noise(dataset, key, names=("val", "test")):
    """One fixed noisy realisation for val/test, shared by all groups."""
    for name in names:
        key, k = jax.random.split(key)
        s = dataset[name]
        s["data"] = add_noise(s["data"], s["noise_scale"], k)
        del s["noise_scale"]
    return dataset


def make_base_inputs(split, key=None):
    """[data (noisy if key given), mask, m_norm]."""
    data = split["data"]
    if key is not None:
        data = add_noise(data, split["noise_scale"], key)
    return jnp.concatenate([data, split["mask"], split["m_norm"]], axis=-1)


def make_resampler(split, group_idx, device, K):
    def resample(key):
        key_noise, key_neg = jax.random.split(key)
        base = make_base_inputs(split, key_noise)          # fresh noise every epoch
        return make_group_xy(split, base, group_idx, key_neg, device, K)
    return resample


def make_group_xy(split, base, group_idx, key, device, K):
    """
    Returns N positives followed by K*N negatives (N*(1+K) rows).
    Negative k (k=1..K) replaces the current group's params of sample i with those
    of sample p[(pos(i)+k) % N] for one random ordering p, so each sample gets K
    distinct donors, none of which is itself.
    """
    cur, vis = group_idx
    n = base.shape[0]
    assert 1 <= K < n, "need 1 <= K < N so that donors are distinct and never self"

    p = jax.random.permutation(key, n)
    idx = jnp.arange(n)

    def donor_map(k):
        # src[p[i]] = p[(i + k) % n]
        return jnp.zeros(n, dtype=int).at[p].set(p[(idx + k) % n])

    src = jax.vmap(donor_map)(jnp.arange(1, K + 1)).reshape(-1)   # (K*n,), k-major

    true_p = split["true_params"]
    neg_params = jnp.tile(true_p, (K, 1)).at[:, cur].set(true_p[src][:, cur])

    params = jnp.concatenate([true_p, neg_params], axis=0)         # (n*(1+K), P)
    base_all = jnp.concatenate([base, jnp.tile(base, (K, 1))], axis=0)

    x = jnp.concatenate([base_all, params[:, vis]], axis=-1)
    y = jnp.concatenate([jnp.ones(n, dtype=jnp.int32),
                         jnp.zeros(K * n, dtype=jnp.int32)])[:, None]
    return jax.device_put(x, device), jax.device_put(y, device)



def make_group_data(dataset, group_idx, key, device, K):
    k_val, k_test = jax.random.split(key)
    fixed = {}
    for name, k in [("val", k_val), ("test", k_test)]:
        base = make_base_inputs(dataset[name])
        fixed[name] = make_group_xy(dataset[name], base, group_idx, k, device, K)

    return {
        "resample_train": make_resampler(dataset["train"], group_idx, device, K),
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
                    weight_decay=cfg["wd"]),
    )


# ===============================================================
# 6. Training loop helpers
# ===============================================================
def run_epoch(model, x, y, batch_size, gpu, pos_weight, optimizer=None):
    total_loss, total_acc, total_w = 0.0, 0.0, 0.0
    for i in range(0, len(x), batch_size):
        bx = jax.device_put(x[i:i + batch_size], gpu)
        by = jax.device_put(y[i:i + batch_size], gpu)

        loss, _ = loss_fn(model, (bx, by), pos_weight)
        acc = accuracy_fn(model, (bx, by), pos_weight)
        bw = jnp.where(by > 0.5, pos_weight, 1.0).sum()   # batch weight
        total_loss += loss * bw
        total_acc += acc * bw
        total_w += bw

        if optimizer is not None:
            train_step(model, optimizer, (bx, by), pos_weight)

    return total_loss / total_w, total_acc / total_w


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

def average_states(states):
    """Uniform mean of a list of nnx.State pytrees."""
    return jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *states)


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

    snapshots = deque(maxlen=cfg.get("swa_k", 10))

    for epoch in range(cfg["epochs"]):
        key, resample_key, perm_key = jax.random.split(key, 3)

        # fresh labels + shuffle
        train_x, train_y = resample_fn(resample_key)
        perm = jax.random.permutation(perm_key, train_x.shape[0])
        train_x, train_y = train_x[perm], train_y[perm]

        pos_weight = get_pos_weight(cfg)
        model.train()
        tr_loss, tr_acc = run_epoch(model, train_x, train_y, cfg["batch_size"], gpu, pos_weight, optimizer)

        model.eval()
        va_loss, va_acc = run_epoch(model, val_x, val_y, cfg["batch_size"], gpu, pos_weight)

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
    Builds each group's data lazily (lower peak memory) and trains it.
    Returns (models, histories, test_sets, key).
    """
    group_indices = build_group_indices(param_groups, ctx["param_names"])
    histories, test_sets = [], []

    for g, (group, gidx) in enumerate(zip(param_groups, group_indices)):
        print(f"\n=== Training model for group {g}: {group} ===")

        key, data_key = jax.random.split(key)
        data = make_group_data(dataset, gidx, data_key, ctx["cpu"], cfg["n_neg"])
        
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
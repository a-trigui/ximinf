import os
import pickle
from flax import nnx

# Checkpointing
import orbax.checkpoint as ocp  # Checkpointing library
ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')

import pathlib  # File path handling library

import ximinf.nn_train as nntr

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

def load_autoregressive_nn(path):
    """
    Load an autoregressive stack of NNX models.

    Parameters
    ----------
    path : str
        Checkpoint directory.

    Returns
    -------
    models_per_group : list[nnx.Module]
        Reconstructed models, one per group.
    model_config : dict
        Loaded configuration dictionary.
    """
    ckpt_dir = pathlib.Path(path).resolve()
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory {ckpt_dir} does not exist.")

    config_path = ckpt_dir / "config.pkl"
    if not config_path.exists():
        raise ValueError("Model config file not found.")

    with open(config_path, "rb") as f:
        model_config = pickle.load(f)

    shared = model_config["shared"]
    group_configs = model_config["groups"]

    values_idx = [i for i, col in enumerate(shared["columns"]) if not col.endswith("_err")]
    errors_idx = [i for i, col in enumerate(shared["columns"]) if col.endswith("_err")]

    checkpointer = ocp.StandardCheckpointer()
    models_per_group = []

    for gconf in group_configs:
        n_params_visible = gconf["n_params_visible"]

        # Recreate abstract model (shape-only)
        abstract_model = nnx.eval_shape(
            lambda: nntr.DeepSetClassifier(
                rho_drop_rate=0.0,
                Nsize_p=shared["Nsize_p"],
                Nsize_r=shared["Nsize_r"],
                n_cols=len(shared["columns"]), #shared["n_cols"]
                n_params=n_params_visible,
                val_idx = values_idx,
                err_idx = errors_idx,
                rngs=nnx.Rngs(0),
            )
        )

        graphdef, rngkey, rngcount, _ = nnx.split(
            abstract_model, nnx.RngKey, nnx.RngCount, ...
        )

        # Restore parameters
        state = checkpointer.restore(
            ckpt_dir / f"state_group_{gconf['group_id']}"
        )

        model = nnx.merge(graphdef, rngkey, rngcount, state)
        models_per_group.append(model)

    return models_per_group, model_config
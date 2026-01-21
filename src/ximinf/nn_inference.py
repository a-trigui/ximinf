# Standard
import os
import json

# Jax
from flax import nnx

# Checkpointing
import orbax.checkpoint as ocp  # Checkpointing library
ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')
import pathlib  # File path handling library

# Modules
import ximinf.nn_train as nntr

# def load_nn(path):
#     """
#     Load a neural network model from a checkpoint.

#     Parameters
#     ----------
#     path : str
#         Path to the checkpoint directory.

#     Returns
#     -------
#     model : nnx.Module
#         The loaded neural network model.

#     Raises
#     ------
#     ValueError
#         If the checkpoint directory or config file does not exist.
#     """
#     # Define the checkpoint directory
#     ckpt_dir = os.path.abspath(path)
#     ckpt_dir = pathlib.Path(ckpt_dir).resolve()

#     # Ensure the folder is removed before saving
#     if ckpt_dir.exists()==False:
#         # Make an error
#         raise ValueError(f"Checkpoint directory {ckpt_dir} does not exist. Please check the path.")
    
#     # Load model configuration
#     config_path = ckpt_dir / 'config.json'
#     if not config_path.exists():
#         raise ValueError("Model config file not found in checkpoint directory.")
    
#     with open(config_path, 'r') as f:
#         model_config = json.load(f)

#     Nsize_p = model_config['Nsize_p']
#     Nsize_r = model_config['Nsize_r']
#     n_cols = model_config['n_cols']
#     n_params = model_config['n_params']
#     N_size_embed = model_config['N_size_embed']

#     # 1. Re-create the checkpointer
#     checkpointer = ocp.StandardCheckpointer()

#     # Split the model into GraphDef (structure) and State (parameters + buffers)
#     abstract_model = nnx.eval_shape(lambda: nntr.DeepSetClassifier(0.0, Nsize_p, Nsize_r, N_size_embed, n_cols, n_params, rngs=nnx.Rngs(0)))
#     abs_graphdef, abs_rngkey, abs_rngcount, _ = nnx.split(abstract_model, nnx.RngKey, nnx.RngCount, ...)

#     # 3. Restore
#     state_restored = checkpointer.restore(ckpt_dir / 'state')
#     print('NNX State restored: ')

#     model = nnx.merge(abs_graphdef, abs_rngkey, abs_rngcount, state_restored)

#     nnx.display(model)

#     return model

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

    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        raise ValueError("Model config file not found.")

    with open(config_path, "r") as f:
        model_config = json.load(f)

    shared = model_config["shared"]
    group_configs = model_config["groups"]

    checkpointer = ocp.StandardCheckpointer()
    models_per_group = []

    for gconf in group_configs:
        n_params_visible = gconf["n_params_visible"]

        # Recreate abstract model (shape-only)
        abstract_model = nnx.eval_shape(
            lambda: nntr.DeepSetClassifier(
                dropout_rate=0.0,
                Nsize_p=shared["Nsize_p"],
                Nsize_r=shared["Nsize_r"],
                n_cols=shared["n_cols"],
                n_params=n_params_visible,
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

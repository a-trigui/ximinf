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

def load_nn(path):
    """
    Load a neural network model from a checkpoint.

    Parameters
    ----------
    path : str
        Path to the checkpoint directory.

    Returns
    -------
    model : nnx.Module
        The loaded neural network model.

    Raises
    ------
    ValueError
        If the checkpoint directory or config file does not exist.
    """
    # Define the checkpoint directory
    ckpt_dir = os.path.abspath(path)
    ckpt_dir = pathlib.Path(ckpt_dir).resolve()

    # Ensure the folder is removed before saving
    if ckpt_dir.exists()==False:
        # Make an error
        raise ValueError(f"Checkpoint directory {ckpt_dir} does not exist. Please check the path.")
    
    # Load model configuration
    config_path = ckpt_dir / 'config.json'
    if not config_path.exists():
        raise ValueError("Model config file not found in checkpoint directory.")
    
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    Nsize_p = model_config['Nsize_p']
    Nsize_r = model_config['Nsize_r']
    n_cols = model_config['n_cols']
    n_params = model_config['n_params']

    # 1. Re-create the checkpointer
    checkpointer = ocp.StandardCheckpointer()

    # Split the model into GraphDef (structure) and State (parameters + buffers)
    abstract_model = nnx.eval_shape(lambda: nntr.DeepSetClassifier(0.05, Nsize_p, Nsize_r, n_cols, n_params, rngs=nnx.Rngs(0)))
    abs_graphdef, abs_rngkey, abs_rngcount, _ = nnx.split(abstract_model, nnx.RngKey, nnx.RngCount, ...)

    # 3. Restore
    state_restored = checkpointer.restore(ckpt_dir / 'state')
    #jax.tree.map(np.testing.assert_array_equal, abstract_state, state_restored)
    print('NNX State restored: ')
    # nnx.display(state_restored)

    model = nnx.merge(abs_graphdef, abs_rngkey, abs_rngcount, state_restored)

    nnx.display(model)

    return model
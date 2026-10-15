import os
import pickle
from flax import nnx

import ximinf.data_helper as dh

# Checkpointing
import orbax.checkpoint as ocp  # Checkpointing library
ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')

from pathlib import Path
import shutil


def save_autoregressive_nn(models_per_group, path, model_config):
    """
    Save an autoregressive stack of NNX models and its configuration.

    Each model is split with ``nnx.split`` so that its graph-independent state
    can be checkpointed independently of the model graph definition. The state
    of group ``g`` is stored as ``state_group_g``. The model configuration is
    stored in ``config.pkl`` using Python ``pickle`` serialization.

    Parameters
    ----------
    models_per_group : list of nnx.Module
        Neural network models corresponding to the autoregressive parameter
        groups. One model is saved for each group.
    path : str or path-like
        Directory in which the model checkpoints and configuration files are
        stored. If the directory already exists, its contents are removed
        before creating the new checkpoint.
    model_config : dict
        Full model configuration containing the shared configuration and the
        configuration for each autoregressive group.

    Returns
    -------
    None
        The model states and configuration are written to disk.
    """
    ckpt_dir = os.path.abspath(path)
    ckpt_dir = ocp.test_utils.erase_and_create_empty(ckpt_dir)

    checkpointer = ocp.StandardCheckpointer()

    for g, model in enumerate(models_per_group):
        # Split model into graph-independent state
        _, _, _, state = nnx.split(model, nnx.RngKey, nnx.RngCount, ...)
        checkpointer.save(ckpt_dir / f"state_group_{g}", state)

    # checkpointer.wait_until_finished()
    checkpointer.close()

    # Save configuration
    with open(ckpt_dir / "config.pkl", "wb") as f:
        pickle.dump(model_config, f)

def load_autoregressive_nn(path, model_cls):
    """
    Load an autoregressive stack of NNX models from a checkpoint directory.

    For each group, an abstract model is instantiated from the stored
    architecture configuration. Its graph definition and RNG state are extracted
    with ``nnx.split`` and used to define the target checkpoint structure. The
    saved parameter state is restored with the Orbax ``StandardCheckpointer``
    and merged back with the graph definition using ``nnx.merge``.

    Parameters
    ----------
    path : str or path-like
        Directory containing the saved model states and configuration.
    model_cls : type
        NNX model class used to reconstruct each autoregressive model.
        The class must accept the architecture parameters stored in the
        model configuration.

    Returns
    -------
    models_per_group : list of nnx.Module
        Reconstructed neural network models, one for each autoregressive
        parameter group.
    model_config : dict
        Configuration dictionary loaded from ``config.pkl``.

    Raises
    ------
    ValueError
        If the checkpoint directory does not exist or if ``config.pkl`` is
        missing.
    """
    ckpt_dir = Path(path).resolve()
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory {ckpt_dir} does not exist.")

    config_path = ckpt_dir / "config.pkl"
    if not config_path.exists():
        raise ValueError("Model config file not found.")

    with open(config_path, "rb") as f:
        model_config = pickle.load(f)

    shared = model_config["shared"]
    group_configs = model_config["groups"]

    checkpointer = ocp.StandardCheckpointer()
    models_per_group = []

    for gconf in group_configs:
        n_params_visible = gconf["n_params_visible"]

        abstract_model = model_cls(
            phi_drop_rate=shared["phi_dropout_rate"],
            rho_drop_rate=shared["rho_dropout_rate"],
            Nsize_p=shared["Nsize_p"],
            Nsize_r=shared["Nsize_r"],
            depth_r=shared["depth_r"],
            depth_p=shared["depth_p"],
            n_cols=len(shared["columns"]),
            n_params=n_params_visible,
            rngs=nnx.Rngs(0),
        )
        
        graphdef, rngkey, rngcount, abstract_state = nnx.split(
            abstract_model, nnx.RngKey, nnx.RngCount, ...
        )
        
        state = checkpointer.restore(
            ckpt_dir / f"state_group_{gconf['group_id']}",
            target=abstract_state,
        )

        model = nnx.merge(graphdef, rngkey, rngcount, state)
        models_per_group.append(model)

    return models_per_group, model_config




def build_run_name(cfg, ctx, param_groups, arch, n_realisations, sim_idx):
    """
    Construct a standardized identifier string for a neural network training run.

    Parameters
    ----------
    cfg : dict
        Training configuration containing ``"batch_size"``, ``"lr"``,
        ``"wd"``, and ``"patience"``.
    ctx : dict
        Training context containing sample counts ``"N"`` and ``"M"``.
    param_groups : list
        List of parameter groups defining the model structure.
    arch : dict
        Architecture specification containing ``"phi_dropout"``,
        ``"rho_dropout"``, ``"width_phi"``, ``"width_rho"``,
        ``"depth_phi"``, and ``"depth_rho"``.
    n_realisations : int
        Number of noise or simulation realizations per parameter point.
    sim_idx : int or str
        Index or identifier of the underlying simulation dataset.

    Returns
    -------
    str
        Formatted run name string incorporating training, architecture,
        and simulation metadata.
    """
    return (
        f"NN_M_N_{ctx['N']}_{n_realisations}_M_{ctx['M']}"
        f"_batch_{cfg['batch_size']}_lr_{cfg['lr']}_wd_{cfg['wd']}"
        f"_params_{dh.format_groups(param_groups)}"
        f"_dropout_phi_{arch['phi_dropout']}_rho_{arch['rho_dropout']}"
        f"_width_phi_{arch['width_phi']}_rho_{arch['width_rho']}"
        f"_depth_phi_{arch['depth_phi']}_rho_{arch['depth_rho']}"
        f"_patience_{cfg['patience']}_sim_{sim_idx}"
    )


def save_run(models, model_config, run_name, sim_path, base_dir="../data/NNs"):
    """
    Save trained model checkpoints and archive associated configuration files.

    Serializes the autoregressive model state and configuration, then copies
    ``nn_config.py`` and ``sim_config.py`` to the run checkpoint directory.

    Parameters
    ----------
    models : list of nnx.Module
        List of trained NNX model instances across parameter groups.
    model_config : dict
        Complete model architecture and training configuration dictionary.
    run_name : str
        Name of the run directory to create under ``base_dir``.
    sim_path : str or pathlib.Path
        Directory containing the simulation configuration file ``sim_config.py``.
    base_dir : str or pathlib.Path, optional
        Base directory in which run folders are stored. Default is
        ``"../data/NNs"``.

    Returns
    -------
    pathlib.Path
        Path to the created run directory.
    """
    nn_path = Path(base_dir) / run_name
    save_autoregressive_nn(models, nn_path, model_config)
    shutil.copy("nn_config.py", nn_path / "nn_config.py")
    shutil.copy(Path(sim_path) / "sim_config.py", nn_path / "sim_config.py")
    print(f"NNs saved to {nn_path}")
    return nn_path
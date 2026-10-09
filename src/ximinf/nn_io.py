import os
import pickle
from flax import nnx

# Checkpointing
import orbax.checkpoint as ocp  # Checkpointing library
ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')

from pathlib import Path
import shutil


def save_autoregressive_nn(models_per_group, path, model_config):
    """
    Save an autoregressive stack of NNX models and its configuration.

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

    Notes
    -----
    Each model is split with ``nnx.split`` so that its graph-independent
    state can be checkpointed independently of the model graph definition.
    The state of group ``g`` is stored as ``state_group_g``.

    The model configuration is stored in ``config.pkl`` using Python
    ``pickle`` serialization. The ``nn_config.py`` file is also copied into
    the checkpoint directory for reference.
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

    Notes
    -----
    For each group, an abstract model is first instantiated from the stored
    architecture configuration. Its graph definition and RNG state are
    extracted with ``nnx.split`` and used to define the target checkpoint
    structure. The saved parameter state is then restored with the
    Orbax ``StandardCheckpointer`` and merged back with the graph definition
    using ``nnx.merge``.

    The number of visible parameters for each group is read from
    ``group_configs`` in the saved configuration.
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

def _fmt_groups(groups):
    """['mabs', 'beta', 'alpha'] -> 'mabs-beta-alpha'; nested groups joined with '+'."""
    return "+".join(
        "-".join(g) if isinstance(g, (list, tuple)) else str(g) for g in groups
    )


def build_run_name(cfg, ctx, param_groups, arch, n_realisations, sim_idx):
    """
    cfg  : training config (batch_size, lr, patience are read from it)
    ctx  : dict with N and M
    arch : dict(phi_dropout, rho_dropout, width_phi, width_rho, depth_phi, depth_rho)
    """
    return (
        f"NN_M_N_{ctx['N']}_{n_realisations}_M_{ctx['M']}"
        f"_batch_{cfg['batch_size']}_lr_{cfg['lr']}"
        f"_params_{_fmt_groups(param_groups)}"
        f"_dropout_phi_{arch['phi_dropout']}_rho_{arch['rho_dropout']}"
        f"_width_phi_{arch['width_phi']}_rho_{arch['width_rho']}"
        f"_depth_phi_{arch['depth_phi']}_rho_{arch['depth_rho']}"
        f"_patience_{cfg['patience']}_sim_{sim_idx}"
    )


def save_run(models, model_config, run_name, sim_path, base_dir="../data/NNs"):
    """Save the networks and copy the config files next to them. Returns the path."""
    nn_path = Path(base_dir) / run_name
    save_autoregressive_nn(models, nn_path, model_config)
    shutil.copy("nn_config.py", nn_path / "nn_config.py")
    shutil.copy(Path(sim_path) / "sim_config.py", nn_path / "sim_config.py")
    print(f"NNs saved to {nn_path}")
    return nn_path
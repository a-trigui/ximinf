import ximinf.data_helper as dh

import pickle
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def save_autoregressive_nn(models_per_group, path, model_config):
    """
    Save an autoregressive stack of NNX models and its configuration.

    Each model is split with ``nnx.split`` so that its graph-independent state
    (everything except RNG keys and counts) can be stored without the model
    graph definition. The state of group ``g`` is converted to a pure nested
    dict of numpy arrays and pickled as ``state_group_{group_id}.pkl``, where
    ``group_id`` is read from ``model_config["groups"][g]``. The model
    configuration is stored in ``config.pkl``.

    Parameters
    ----------
    models_per_group : list of nnx.Module
        One model per autoregressive parameter group.
    path : str or path-like
        Checkpoint directory. If it already exists, its contents are removed
        before the new checkpoint is written.
    model_config : dict
        Full model configuration with keys ``"shared"`` and ``"groups"``.

    Returns
    -------
    None
    """
    group_configs = model_config["groups"]
    if len(models_per_group) != len(group_configs):
        raise ValueError(
            f"{len(models_per_group)} models but {len(group_configs)} group configs."
        )

    ckpt_dir = Path(path).resolve()
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True)

    for model, gconf in zip(models_per_group, group_configs):
        _, _, _, state = nnx.split(model, nnx.RngKey, nnx.RngCount, ...)
        pure = jax.tree.map(np.asarray, nnx.to_pure_dict(state))  # host numpy
        with open(ckpt_dir / f"state_group_{gconf['group_id']}.pkl", "wb") as f:
            pickle.dump(pure, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(ckpt_dir / "config.pkl", "wb") as f:
        pickle.dump(model_config, f)


def load_autoregressive_nn(path, model_cls):
    """
    Load an autoregressive stack of NNX models from a checkpoint directory.

    For each group, a model is instantiated from the stored architecture
    configuration. Its graph definition and RNG state are extracted with
    ``nnx.split``. The saved parameter state is loaded, checked against the
    freshly built state (structure, shapes, dtypes), written into it with
    ``nnx.replace_by_pure_dict`` and merged back with ``nnx.merge``.

    Parameters
    ----------
    path : str or path-like
        Directory containing the saved model states and configuration.
    model_cls : type
        NNX model class used to reconstruct each autoregressive model.

    Returns
    -------
    models_per_group : list of nnx.Module
    model_config : dict

    Raises
    ------
    ValueError
        If the directory, ``config.pkl`` or a state file is missing, or if a
        saved state does not match the architecture in the configuration.
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

    models_per_group = []
    for gconf in group_configs:
        state_path = ckpt_dir / f"state_group_{gconf['group_id']}.pkl"
        if not state_path.exists():
            raise ValueError(f"State file {state_path} not found.")

        model = model_cls(
            phi_drop_rate=shared["phi_dropout_rate"],
            rho_drop_rate=shared["rho_dropout_rate"],
            Nsize_p=shared["Nsize_p"],
            Nsize_r=shared["Nsize_r"],
            depth_r=shared["depth_r"],
            depth_p=shared["depth_p"],
            n_cols=len(shared["columns"]),
            n_params=gconf["n_params_visible"],
            rngs=nnx.Rngs(0),
        )
        graphdef, rngkey, rngcount, state = nnx.split(
            model, nnx.RngKey, nnx.RngCount, ...
        )

        with open(state_path, "rb") as f:
            pure = pickle.load(f)

        # Same role as `target=abstract_state` in Orbax: refuse a mismatch.
        def _check(expected, loaded):
            if expected.shape != loaded.shape or expected.dtype != loaded.dtype:
                raise ValueError(
                    f"Group {gconf['group_id']}: saved {loaded.dtype}{loaded.shape} "
                    f"!= model {expected.dtype}{expected.shape}."
                )
            return loaded

        try:
            pure = jax.tree.map(_check, nnx.to_pure_dict(state), pure)
        except ValueError as e:
            raise ValueError(
                f"State in {state_path} does not match the model: {e}"
            ) from e

        pure = jax.tree.map(jnp.asarray, pure)
        nnx.replace_by_pure_dict(state, pure)
        models_per_group.append(nnx.merge(graphdef, rngkey, rngcount, state))

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
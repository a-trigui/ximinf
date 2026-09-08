import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path
import shutil
import h5py

import ximinf.nn_train as nntr

import jax.scipy as jsp
from ximinf import cosmo_helper as ch

import h5py

def normalize(data_dict, stats_dict):
    normed = {}
    for k, v in data_dict.items():
        if k in stats_dict:
            mu = stats_dict[k]['mu']
            sigma = stats_dict[k]['sigma']
            normed[k] = (v - mu) / sigma
        else:
            normed[k] = v  # leave untouched
    return normed

def unnormalize(normed_params, param_stats):
    unnormed = {}
    for k, v in normed_params.items():
        if k in param_stats:
            mu = param_stats[k]['mu']
            sigma = param_stats[k]['sigma']
            unnormed[k] = v * sigma + mu  # inverse of normalization
        else:
            unnormed[k] = v  # leave untouched
    return unnormed

# --------------------------------------------------------------------------
# 0. Save the simulation
# --------------------------------------------------------------------------
def save_simulation(
    params_dict,
    dict_arrays,
    priors,
    base_dir=Path("../data/SIM"),
    sim_config_path="sim_config.py",
    nn_config_path="nn_config.py",
):
    # Base directory
    base_dir.mkdir(parents=True, exist_ok=True)

    # Find next simulation number
    existing = sorted(
        [
            int(p.name.split("_")[1])
            for p in base_dir.glob("sim_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()
        ]
    )

    sim_id = max(existing, default=0) + 1

    # Create simulation directory
    sim_dir = base_dir / f"sim_{sim_id:04d}"
    sim_dir.mkdir()

    # Copy configuration file
    shutil.copy(sim_config_path, sim_dir / "sim_config.py")
    shutil.copy(nn_config_path, sim_dir / "nn_config.py")

    # HDF5 file path
    save_path = sim_dir / "simulations.h5"

    with h5py.File(save_path, "w") as f:
        # Save parameters
        for key, arr in params_dict.items():
            f.create_dataset(f"params/{key}", data=arr, dtype=np.float32)

        # Save data columns
        for col, arr in dict_arrays.items():
            f.create_dataset(f"data/{col}", data=arr, dtype=np.float32)

        # Save priors
        priors_grp = f.create_group("priors")
        for name, prior in priors.items():
            param_grp = priors_grp.create_group(name)
            param_grp.create_dataset("range", data=prior["range"])
            param_grp.attrs["type"] = prior["type"]

    return sim_dir, save_path

def load_simulation(
    sim_dir,
):
    sim_dir = Path(sim_dir)

    # HDF5 file path
    save_path = sim_dir / "simulations.h5"

    params_dict = {}
    dict_arrays = {}
    priors = {}

    with h5py.File(save_path, "r") as f:
        # Load parameters
        for key in f["params"]:
            params_dict[key] = f[f"params/{key}"][:]

        # Load data columns
        for col in f["data"]:
            dict_arrays[col] = f[f"data/{col}"][:]

        # Load priors
        for name in f["priors"]:
            param_grp = f[f"priors/{name}"]

            priors[name] = {
                "range": param_grp["range"][:],
                "type": param_grp.attrs["type"],
            }

    return params_dict, dict_arrays, priors

# --------------------------------------------------------------------------
# 0. Load the simulation
# --------------------------------------------------------------------------
def save_hdf5(file_path, params, data):
    with h5py.File(file_path, "w") as f:

        # Save parameters
        params_grp = f.create_group("params")
        for k, value in params.items():
            params_grp.create_dataset(k, data=np.asarray(value))

        # Save data
        data_grp = f.create_group("data")
        for k, value in data.items():
            data_grp.create_dataset(k, data=np.asarray(value))

def _load_group(group):
    return {
        k: jnp.array(
            dataset[()] if dataset.ndim == 0 else dataset[:],
            dtype=jnp.float32,
        )
        for k, dataset in group.items()
    }

def load_hdf5(file_path):
    with h5py.File(file_path, "r") as f:

        # Load parameters
        params = _load_group(f["params"])

        # Load data
        data = _load_group(f["data"])
        
        return params, data

# --------------------------------------------------------------------------
# 1. Normalize input: accept either a list of sim dicts (parallel case)
#    or a single sim dict (one-off case), always return (list, N_total).
# --------------------------------------------------------------------------
def _as_result_list(results):
    if isinstance(results, dict):
        return [results], 1
    return list(results), len(results)


def filter_and_pad(results, columns, get_quality_mask_fn):
    """
    results : dict (single sim) or list of dict (multiple sims)
    Returns: data_dict (padded, float32), max_size
    """
    results_list, N_total = _as_result_list(results)

    filtered_results = []
    sizes = []
    for sim_data in results_list:
        quality_mask = get_quality_mask_fn(sim_data)
        filtered_data = {
            key: np.asarray(value)[quality_mask]
            for key, value in sim_data.items()
        }
        filtered_results.append(filtered_data)
        sizes.append(len(filtered_data["magobs"]))

    max_size = max(sizes)

    data_dict = {
        col: np.zeros((N_total, max_size), dtype=np.float32)
        for col in columns
    }

    for i, sim_data in enumerate(filtered_results):
        for col, arr in sim_data.items():
            arr = np.asarray(arr, dtype=np.float32)
            data_dict[col][i, :arr.shape[0]] = arr

    return data_dict, max_size

# --------------------------------------------------------------------------
# 3. Cosmology removal + isup shift (mutates/returns data_dict, mask).
# --------------------------------------------------------------------------
def remove_cosmology(
    data_dict,
    cosmo,
    ref_mag=19.3,
    package="cosmologix",
    disable_x64_after=True,
):
    """Remove the cosmological distance modulus from observed magnitudes.

    Parameters
    ----------
    data_dict : dict
        Data dictionary containing ``z``, ``magobs``, and ``isup``.
    cosmo : dict
        Cosmology used to compute the distance modulus.
    ref_mag : float
        Reference magnitude added after removing the distance modulus.
    package : {'astropy', 'cosmologix'}
        Package used to compute the distance modulus.
    disable_x64_after : bool
        Disable JAX x64 precision after the correction.
    """

    z = data_dict["z"]
    magobs = data_dict["magobs"]  # (N, M)

    mask = magobs != 0  # (N, M)

    data_dict["isup"] = data_dict["isup"] - 0.5

    # Careful not to run the correction twice
    mu = ch.distmod(z, cosmo, package=package)

    magobs_corr = jnp.where(
        mask,
        magobs - mu + ref_mag,
        0.0,
    )

    if disable_x64_after:
        jax.config.update("jax_enable_x64", False)

    data_dict["magobs"] = magobs_corr

    return data_dict, mask

# --------------------------------------------------------------------------
# 4. Normalize columns (z-score), special-casing magobs's zero-mask.
# --------------------------------------------------------------------------
def normalize_data(data_dict, columns, mask, data_stats):
    infer_columns = [col for col in columns if col != 'z']
    data_norm = {k: data_dict[k].copy() for k in infer_columns}

    for col, arr in data_norm.items():
        mu = data_stats[col]['mu']
        sigma = data_stats[col]['sigma']
        if col == 'magobs':
            mag_mask = arr != 0.0
            data_norm[col] = arr.at[mag_mask].set((arr[mag_mask] - mu) / sigma)
        else:
            data_norm[col] = (arr - mu) / sigma

    M_ = jnp.sum(mask, axis=1)
    M_norm = (M_ - data_stats['M']['mu']) / data_stats['M']['sigma']

    return data_norm, M_norm, infer_columns

# --------------------------------------------------------------------------
# 5. Stack / flatten / concatenate the data into the inference array.
# --------------------------------------------------------------------------
def build_data_concat(data_norm, infer_columns, N_total, max_size):
    data_arrays = [data_norm[col] for col in infer_columns]  # list of (N, M)
    n_cols = len(infer_columns)
    data_stacked = jnp.stack(jnp.asarray(data_arrays), axis=-1)   # (N, M, n_cols)
    data_concat_infer = data_stacked.reshape(N_total, max_size * n_cols)

    return data_concat_infer

# --------------------------------------------------------------------------
# 6. Build the final inference input.
# --------------------------------------------------------------------------
def build_inputs_infer(data_concat_infer, mask, M_norm, N_total, max_size):
    inputs_infer = jnp.concatenate(
        [
            data_concat_infer,
            jnp.asarray(mask).reshape(N_total, max_size),
            jnp.asarray(M_norm[:, None]),
        ],
        axis=-1,
    )

    return inputs_infer

# --------------------------------------------------------------------------
# 7. Orchestrator
# --------------------------------------------------------------------------
def process_simulation_results(
    results,
    columns,
    models_config,
    get_quality_mask_fn,
    cosmo,
    disable_x64_after=True,
):
    results_list, N_total = _as_result_list(results)

    data_dict, max_size = filter_and_pad(results_list, columns, get_quality_mask_fn)
    data_dict, mask = remove_cosmology(data_dict, cosmo, disable_x64_after=disable_x64_after)

    data_stats = models_config['shared']['data_stats']
    data_norm, M_norm, infer_columns = normalize_data(data_dict, columns, mask, data_stats)

    data_concat_infer = build_data_concat(
        data_norm, infer_columns, N_total, max_size
    )

    inputs_infer = build_inputs_infer(
        data_concat_infer, mask, M_norm, N_total, max_size
    )

    return inputs_infer, data_dict, mask, max_size
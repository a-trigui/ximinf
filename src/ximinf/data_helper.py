import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path
import shutil
import h5py

from ximinf import cosmo_helper as ch

import h5py

def normalize(data_dict, stats_dict):
    """
    Normalize data using parameter-specific mean and standard deviation.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping parameter or data names to numerical values.

    stats_dict : dict
        Dictionary containing normalization statistics for selected entries
        in ``data_dict``. For each normalized key, ``stats_dict[key]`` must
        contain ``"mu"`` and ``"sigma"`` entries.

    Returns
    -------
    dict
        Dictionary containing the normalized values. Entries present in
        ``stats_dict`` are transformed according to

        .. math::

            x_{\\mathrm{norm}} = \\frac{x - \\mu}{\\sigma}.

        Entries without corresponding statistics are returned unchanged.

    Notes
    -----
    The input dictionary is not modified. A new dictionary containing the
    normalized or unchanged values is returned.
    """
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
    """
    Transform normalized parameters back to their original parameterization.

    Parameters
    ----------
    normed_params : dict
        Dictionary containing normalized parameter values.

    param_stats : dict
        Dictionary containing the normalization statistics for the
        parameters. For each parameter to be unnormalized,
        ``param_stats[key]`` must contain ``"mu"`` and ``"sigma"``.

    Returns
    -------
    dict
        Dictionary containing the parameters in their original
        parameterization. Normalized values are transformed according to

        .. math::

            x = x_{\\mathrm{norm}} \\sigma + \\mu.

        Entries without corresponding statistics are returned unchanged.

    Notes
    -----
    The input dictionary is not modified. A new dictionary is returned.
    """
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
):
    """
    Save a simulation and its configuration to an HDF5 file.

    A new simulation directory is created under ``base_dir`` using the next
    available simulation identifier. The simulation parameters, simulated
    data, and prior definitions are stored in ``simulations.h5``. The
    simulation configuration file is also copied into the simulation
    directory.

    Parameters
    ----------
    params_dict : dict
        Dictionary mapping parameter names to arrays containing the
        parameter values used for the simulation.

    dict_arrays : dict
        Dictionary mapping data-column names to arrays containing the
        simulated data.

    priors : dict
        Dictionary containing the prior definitions for each parameter.
        Each prior must contain a ``"range"`` entry and a ``"type"`` entry.

    base_dir : pathlib.Path or str, optional
        Base directory in which simulation directories are created.
        Default is ``Path("../data/SIM")``.

    sim_config_path : pathlib.Path or str, optional
        Path to the simulation configuration file to copy into the new
        simulation directory. Default is ``"sim_config.py"``.

    Returns
    -------
    sim_dir : pathlib.Path
        Path to the newly created simulation directory.

    save_path : pathlib.Path
        Path to the generated ``simulations.h5`` file.

    Notes
    -----
    Simulation directories are named using the format ``sim_XXXX``, where
    ``XXXX`` is a zero-padded simulation identifier.

    Parameters are stored under the ``params`` HDF5 group, simulated data
    under ``data``, and prior definitions under ``priors``.
    Parameter and data arrays are stored using ``float32`` precision.
    """
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

def load_simulation(sim_dir):
    """
    Load a saved simulation from an HDF5 file.

    Parameters
    ----------
    sim_dir : pathlib.Path or str
        Directory containing the ``simulations.h5`` file produced by
        :func:`save_simulation`.

    Returns
    -------
    params_dict : dict
        Dictionary mapping parameter names to NumPy arrays loaded from the
        ``params`` HDF5 group.

    dict_arrays : dict
        Dictionary mapping data-column names to NumPy arrays loaded from the
        ``data`` HDF5 group.

    priors : dict
        Dictionary containing the prior definitions loaded from the
        ``priors`` HDF5 group. Each entry contains ``"range"`` and
        ``"type"``.

    Raises
    ------
    FileNotFoundError
        If ``simulations.h5`` does not exist in ``sim_dir``.

    Notes
    -----
    The function expects the HDF5 file to follow the directory structure
    produced by :func:`save_simulation`.
    """
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
    """
    Save parameter and data dictionaries to an HDF5 file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path of the HDF5 file to create. An existing file at this path is
        overwritten.

    params : dict
        Dictionary mapping parameter names to array-like values. Parameters
        are stored in the ``params`` HDF5 group.

    data : dict
        Dictionary mapping data-column names to array-like values. Data are
        stored in the ``data`` HDF5 group.

    Returns
    -------
    None

    Notes
    -----
    Values are converted to NumPy arrays using ``np.asarray`` before being
    written to the HDF5 file.
    """
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
    """
    Load all datasets in an HDF5 group as JAX float32 arrays.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group containing the datasets to load.

    Returns
    -------
    dict
        Dictionary mapping dataset names to JAX arrays with ``float32``
        dtype. Scalar datasets are converted using ``dataset[()]`` while
        non-scalar datasets are loaded using ``dataset[:]``.
    """
    return {
        k: jnp.array(
            dataset[()] if dataset.ndim == 0 else dataset[:],
            dtype=jnp.float32,
        )
        for k, dataset in group.items()
    }

def load_hdf5(file_path):
    """
    Load parameter and data arrays from an HDF5 file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to an HDF5 file containing ``params`` and ``data`` groups,
        typically produced by :func:`save_hdf5`.

    Returns
    -------
    params : dict
        Dictionary mapping parameter names to JAX arrays with ``float32``
        dtype.

    data : dict
        Dictionary mapping data-column names to JAX arrays with ``float32``
        dtype.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.

    Notes
    -----
    All loaded arrays are converted to JAX arrays with ``float32`` dtype.
    """
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
    """
    Normalize single or multiple simulation results to a list representation.

    Parameters
    ----------
    results : dict or sequence of dict
        Simulation result represented either by a single dictionary or by a
        sequence of dictionaries.

    Returns
    -------
    results_list : list of dict
        List containing the simulation result dictionaries.

    N_total : int
        Number of simulation results in ``results_list``.
    """
    if isinstance(results, dict):
        return [results], 1
    return list(results), len(results)


def filter_and_pad(results, columns, get_quality_mask_fn=None):
    """
    Apply a quality selection to simulation results and pad them to a common size.

    Each simulation is filtered using ``get_quality_mask_fn``. Since the
    resulting simulations may contain different numbers of objects, the
    selected data are padded with zeros to the maximum number of objects
    across all simulations.

    Parameters
    ----------
    results : dict or sequence of dict
        Simulation result represented either by a single dictionary or by a
        sequence of dictionaries.

    columns : sequence of str
        Names of the data columns to include in the output.

    get_quality_mask_fn : callable
        Function that accepts one simulation dictionary and returns a
        boolean mask selecting the objects that satisfy the required
        quality criteria.

    Returns
    -------
    data_dict : dict
        Dictionary mapping each data-column name to a NumPy array of shape
        ``(N_total, max_size)`` and ``float32`` dtype. Simulations shorter
        than ``max_size`` are padded with zeros.

    max_size : int
        Maximum number of selected objects among all simulations.

    Notes
    -----
    The number of objects in each simulation is determined from the
    ``"magobs"`` column after applying the quality mask.

    The function preserves the ordering of the selected objects within each
    simulation.
    """
    results_list, N_total = _as_result_list(results)

    filtered_results = []
    sizes = []
    for sim_data in results_list:
        if get_quality_mask_fn is not None:
            quality_mask = get_quality_mask_fn(sim_data)
            filtered_data = {
                key: np.asarray(value)[quality_mask]
                for key, value in sim_data.items()
            }
        else:
            filtered_data = sim_data

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
    """
    Remove the cosmological distance modulus from observed magnitudes.

    The distance modulus is computed at each redshift using the supplied
    cosmology and subtracted from the observed magnitudes. A reference
    magnitude is then added to keep the corrected magnitudes on a convenient
    numerical scale.

    The ``isup`` indicator is shifted by ``-0.5`` as part of the same
    preprocessing step.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing at least the ``"z"``, ``"magobs"``, and
        ``"isup"`` entries. ``magobs`` is expected to have shape
        ``(N, M)``.

    cosmo : dict
        Canonical cosmology dictionary used to compute the distance modulus.

    ref_mag : float, optional
        Reference magnitude added after subtracting the distance modulus.
        Default is 19.3.

    package : {"astropy", "cosmologix"}, optional
        Cosmology package used to calculate the distance modulus. Default is
        ``"cosmologix"``.

    disable_x64_after : bool, optional
        If ``True``, disable JAX 64-bit precision after the cosmological
        correction. Default is ``True``.

    Returns
    -------
    data_dict : dict
        Updated data dictionary containing the corrected ``"magobs"`` and
        shifted ``"isup"`` values.

    mask : jax.Array
        Boolean mask identifying non-zero entries in the original
        ``"magobs"`` array.

    Notes
    -----
    For valid entries, the corrected magnitude is computed as

    .. math::

        m_{\\mathrm{corr}} =
        m_{\\mathrm{obs}} - \\mu(z) + m_{\\mathrm{ref}}.

    Zero-valued entries in ``magobs`` are treated as padding and remain zero
    after the correction.

    The function modifies ``data_dict`` in place and also returns it.
    """

    z = data_dict["z"]
    magobs = data_dict["magobs"]  # (N, M)

    mask = magobs != 0  # (N, M)

    if "isup" in data_dict:
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
    """
    Normalize simulation data for neural-network inference.

    The selected inference columns are standardized using the supplied
    dataset statistics. The redshift column ``"z"`` is excluded from the
    normalized inference data. Zero-valued padding entries in ``"magobs"``
    are preserved and are not normalized.

    The number of valid objects in each simulation is computed from ``mask``
    and normalized separately using the statistics associated with ``"M"``.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the preprocessed simulation data.

    columns : sequence of str
        Data columns available for inference. The ``"z"`` column is
        explicitly excluded from the returned normalized inference columns.

    mask : jax.Array
        Boolean mask identifying valid, non-padded objects. Expected to have
        shape ``(N, M)``.

    data_stats : dict
        Dictionary containing the mean and standard deviation for each
        normalized data column. Each entry must contain ``"mu"`` and
        ``"sigma"``. An ``"M"`` entry must also be provided for the number
        of valid objects.

    Returns
    -------
    data_norm : dict
        Dictionary containing the normalized inference columns. For
        ``"magobs"``, zero-valued padding entries remain zero.

    M_norm : jax.Array
        Normalized number of valid objects for each simulation.

    infer_columns : list of str
        Names of the columns included in ``data_norm`` and subsequently used
        to construct the inference input.

    Notes
    -----
    Non-zero values are normalized according to

    .. math::

        x_{\\mathrm{norm}} = \\frac{x - \\mu}{\\sigma}.

    The number of valid objects is computed as

    .. math::

        M = \\sum_j \\mathrm{mask}_j,

    and is normalized using ``data_stats["M"]``.
    """
    data_norm = {k: data_dict[k].copy() for k in columns}

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

    return data_norm, M_norm

# --------------------------------------------------------------------------
# 5. Stack / flatten / concatenate the data into the inference array.
# --------------------------------------------------------------------------
def build_data_concat(data_norm, infer_columns, N_total, max_size):
    """
    Stack and flatten normalized simulation data into inference features.

    The normalized data columns are stacked along a final feature axis and
    then flattened over the object and feature dimensions for each
    simulation.

    Parameters
    ----------
    data_norm : dict
        Dictionary containing normalized data arrays. Each array is expected
        to have shape ``(N_total, max_size)``.

    infer_columns : sequence of str
        Names of the data columns to include, in the order in which they
        should appear in the flattened input.

    N_total : int
        Number of simulations.

    max_size : int
        Maximum number of objects per simulation after padding.

    Returns
    -------
    data_concat_infer : jax.Array
        Flattened inference data with shape

        ``(N_total, max_size * n_cols)``

        where ``n_cols = len(infer_columns)``.
    """
    n_cols = len(infer_columns)

    data_concat_infer = jnp.concatenate(
        [data_norm[col][..., None] for col in infer_columns],
        axis=-1,
    ).reshape(N_total, max_size * n_cols)

    return data_concat_infer

# --------------------------------------------------------------------------
# 6. Build the final inference input.
# --------------------------------------------------------------------------
def build_inputs_infer(data_concat_infer, mask, M_norm, N_total, max_size):
    """
    Construct the final neural-network inference input array.

    The flattened normalized data, object-validity mask, and normalized
    number of objects are concatenated into a single feature array.

    Parameters
    ----------
    data_concat_infer : jax.Array
        Flattened normalized simulation data with shape
        ``(N_total, max_size * n_cols)``.

    mask : array-like
        Boolean or numerical mask identifying valid objects, with shape
        ``(N_total, max_size)``.

    M_norm : jax.Array
        Normalized number of valid objects for each simulation. Expected to
        have shape ``(N_total,)``.

    N_total : int
        Number of simulations.

    max_size : int
        Maximum number of objects per simulation.

    Returns
    -------
    inputs_infer : jax.Array
        Final inference input array containing the flattened normalized
        observations, validity mask, and normalized object count.
        Its shape is

        ``(N_total, max_size * n_cols + max_size + 1)``.
    """
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
    """
    Process simulated supernova data into neural-network inference inputs.

    This function orchestrates the complete preprocessing pipeline:

    1. Normalize single or multiple simulation results to a common list
       representation.
    2. Apply the quality-selection mask and zero-pad simulations to a common
       size.
    3. Remove the cosmological distance modulus from the observed
       magnitudes.
    4. Normalize the inference data using the statistics stored in
       ``models_config``.
    5. Stack and flatten the normalized data.
    6. Concatenate the flattened data, object-validity mask, and normalized
       number of objects into the final inference input.

    Parameters
    ----------
    results : dict or sequence of dict
        Single simulation result or collection of simulation results.

    columns : sequence of str
        Names of the data columns to retain during preprocessing.

    models_config : dict
        Model configuration containing the normalization statistics under
        ``models_config["shared"]["data_stats"]``.

    get_quality_mask_fn : callable
        Function that accepts one simulation dictionary and returns a
        boolean mask identifying objects that pass the required quality
        criteria.

    cosmo : dict
        Canonical cosmology dictionary used to remove the cosmological
        distance modulus.

    disable_x64_after : bool, optional
        If ``True``, disable JAX 64-bit precision after the cosmological
        correction. Default is ``True``.

    Returns
    -------
    inputs_infer : jax.Array
        Final neural-network inference input array containing the flattened
        normalized observations, validity mask, and normalized object
        count.

    data_dict : dict
        Preprocessed but non-normalized data dictionary after quality
        filtering and cosmological correction.

    mask : jax.Array
        Boolean mask identifying valid, non-padded objects.

    max_size : int
        Maximum number of valid objects among the processed simulations.

    Notes
    -----
    The function applies the same preprocessing operations to all supplied
    simulations, ensuring that the resulting arrays have compatible shapes
    for batched neural-network inference.

    The normalization statistics are taken from
    ``models_config["shared"]["data_stats"]`` and therefore should be the
    same statistics used when training the corresponding inference model.
    """
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
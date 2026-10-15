import sys

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ximinf.cosmo_helper import get_canonical
from ximinf.data_helper import (
    group_names_as_lists,
    format_groups,
    normalize_array,
    unnormalize_array,
    normalize_priors,
    get_next_simulation_index,
    save_simulation,
    load_simulation,
    save_hdf5,
    load_hdf5,
    load_module_from_run,
    filter_and_pad,
    remove_cosmology,
    normalize_data,
    build_data_concat,
    _as_result_list,
    normalize,
    unnormalize,
    build_inputs_infer,
)

STATS = {"a": {"mu": 1.0, "sigma": 2.0}, "b": {"mu": -1.0, "sigma": 0.5}}

# ---------------------------------------------------------------- groups
def test_group_names_as_lists():
    out = group_names_as_lists(["H0", ("Om", "w0"), ["a"]])
    assert out == [["H0"], ["Om", "w0"], ["a"]]


def test_format_groups():
    assert format_groups(["H0", ["Om", "w0"], ("a", "b")]) == "H0+Om-w0+a-b"


# ---------------------------------------------------- array normalisation
def test_normalize_unnormalize_array_roundtrip_and_order():
    theta = np.array([[3.0, 0.0], [1.0, -1.0]])
    normed = normalize_array(theta, ["a", "b"], STATS)
    np.testing.assert_allclose(normed, [[1.0, 2.0], [0.0, 0.0]])
    np.testing.assert_allclose(unnormalize_array(normed, ["a", "b"], STATS), theta)

    # order of `names` must follow the trailing axis
    swapped = normalize_array(theta[:, ::-1], ["b", "a"], STATS)
    np.testing.assert_allclose(swapped, normed[:, ::-1])


def test_normalize_priors():
    priors = {"a": {"range": (0.0, 4.0), "type": "uniform"}}
    out = normalize_priors(priors, STATS)
    np.testing.assert_allclose(out["a"]["range"], [-0.5, 1.5])
    assert out["a"]["type"] == "uniform"
    assert priors["a"]["range"] == (0.0, 4.0)  # input untouched


# ------------------------------------------------------------- file I/O
def test_next_simulation_index(tmp_path):
    assert get_next_simulation_index(tmp_path / "missing") == 1
    for n in ("sim_0001", "sim_0002", "sim_0004", "sim_abc"):
        (tmp_path / n).mkdir()
    (tmp_path / "sim_0003").touch()  # a file, not a directory -> ignored
    assert get_next_simulation_index(tmp_path) == 3
    assert get_next_simulation_index(tmp_path, prefix="other") == 1


def test_save_load_simulation_roundtrip(tmp_path):
    params = {"H0": np.array([70.0, 71.0])}
    data = {"magobs": np.array([[18.0, 19.0], [20.0, 21.0]])}
    priors = {"H0": {"range": (60.0, 80.0), "type": "uniform"}}
    cfg = tmp_path / "cfg.py"
    cfg.write_text("X = 1\n")
    folder = tmp_path / "run" / "nested"

    path = save_simulation(folder, "simulation.h5", params, data,
                           priors=priors, sim_config_path=cfg)
    assert path.exists()
    assert (folder / "cfg.py").exists()

    # default file_name of load_simulation is "simulation.h5"
    p, d, pr = load_simulation(folder)
    assert p["H0"].dtype == np.float32
    np.testing.assert_allclose(d["magobs"], data["magobs"])
    np.testing.assert_allclose(pr["H0"]["range"], [60.0, 80.0])
    assert str(pr["H0"]["type"]) == "uniform"


def test_save_simulation_without_priors(tmp_path):
    save_simulation(tmp_path, "s.h5", {"a": np.zeros(2)}, {"x": np.zeros((2, 2))})
    _, _, priors = load_simulation(tmp_path, "s.h5")
    assert priors is None


def test_load_simulation_missing_file(tmp_path):
    with pytest.raises((FileNotFoundError, OSError)):
        load_simulation(tmp_path, "nope.h5")


def test_save_hdf5_returns_path_and_load_is_float32_jax(tmp_path):
    f = tmp_path / "a.h5"
    out = save_hdf5(f, {"H0": np.array([70.0])}, {"x": np.ones((1, 2))})
    assert out == f
    params, data = load_hdf5(f)
    assert isinstance(params["H0"], jax.Array)
    assert params["H0"].dtype == jnp.float32
    assert data["x"].dtype == jnp.float32


def test_load_hdf5_scalar_dataset(tmp_path):
    f = tmp_path / "scalar.h5"
    with h5py.File(f, "w") as h:
        h["params/s"] = 3.0
        h["data/x"] = np.arange(3)
    params, data = load_hdf5(f)
    assert params["s"].shape == ()
    assert params["s"].dtype == jnp.float32
    assert data["x"].dtype == jnp.float32


def test_load_module_from_run(tmp_path):
    run = tmp_path / "run_a"
    run.mkdir()
    (run / "cfg.py").write_text("VALUE = 42\n")
    mod = load_module_from_run(run, "cfg.py")
    assert mod.VALUE == 42
    assert "cfg_run_a" not in sys.modules  # not registered globally

    with pytest.raises(FileNotFoundError):
        load_module_from_run(run, "missing.py")


def test_normalize_and_unnormalize():
    data = {"H0": 70.0, "Om": 0.3, "extra": 42.0}
    stats = {
        "H0": {"mu": 67.0, "sigma": 2.0},
        "Om": {"mu": 0.3, "sigma": 0.05},
    }

    normed = normalize(data, stats)
    assert np.isclose(normed["H0"], (70.0 - 67.0) / 2.0)
    assert np.isclose(normed["Om"], 0.0)
    assert normed["extra"] == 42.0  # Unmodified

    restored = unnormalize(normed, stats)
    assert np.isclose(restored["H0"], data["H0"])
    assert np.isclose(restored["Om"], data["Om"])
    assert restored["extra"] == data["extra"]


def test_as_result_list():
    single = {"a": [1, 2]}
    res_list, count = _as_result_list(single)
    assert count == 1
    assert res_list == [single]

    multiple = [{"a": [1]}, {"a": [2]}]
    res_list, count = _as_result_list(multiple)
    assert count == 2
    assert res_list == multiple


def test_filter_and_pad():
    results = [
        {"magobs": [18.0, 19.0], "x1": [0.1, 0.2]},
        {"magobs": [18.5], "x1": [-0.5]},
    ]

    data_dict, max_size = filter_and_pad(results)
    assert max_size == 2
    assert data_dict["magobs"].shape == (2, 2)
    assert data_dict["x1"].shape == (2, 2)

    # Second simulation should be zero-padded at index 1
    assert data_dict["magobs"][1, 1] == 0.0
    assert data_dict["x1"][1, 1] == 0.0

    # Test with custom quality mask
    mask_fn = lambda d: np.array(d["magobs"]) < 18.6
    filtered_dict, filtered_max_size = filter_and_pad(results, get_quality_mask_fn=mask_fn)
    assert filtered_max_size == 1
    assert filtered_dict["magobs"].shape == (2, 1)


def test_remove_cosmology_astropy():
    cosmo = get_canonical()
    data_dict = {
        "z": np.array([[0.1, 0.2]]),
        "magobs": np.array([[20.0, 0.0]]),  # second element is zero-padded
        "isup": np.array([[1.0, 0.0]]),
    }

    corrected_dict, mask = remove_cosmology(
        data_dict, cosmo, ref_mag=19.3, package="astropy"
    )

    assert mask.shape == (1, 2)
    assert mask[0, 0] == True
    assert mask[0, 1] == False

    # Zero padding in magobs should remain 0
    assert corrected_dict["magobs"][0, 1] == 0.0
    # Valid element should be adjusted
    assert corrected_dict["magobs"][0, 0] != 20.0
    # isup should be shifted by -0.5
    assert np.isclose(corrected_dict["isup"][0, 0], 0.5)


def test_normalize_data():
    data_dict = {
        "magobs": jnp.array([[20.0, 0.0]]),
        "x1": jnp.array([[1.0, 0.0]]),
    }
    mask = jnp.array([[True, False]])
    data_stats = {
        "magobs": {"mu": 20.0, "sigma": 2.0},
        "x1": {"mu": 0.0, "sigma": 1.0},
        "M": {"mu": 1.0, "sigma": 1.0},
    }

    data_norm, M_norm = normalize_data(data_dict, ["magobs", "x1"], mask, data_stats)

    # Padding remains 0
    assert data_norm["magobs"][0, 1] == 0.0
    # Normalized value: (20 - 20) / 2 = 0.0
    assert data_norm["magobs"][0, 0] == 0.0
    assert data_norm["x1"][0, 0] == 1.0
    # M_norm: sum of mask is 1, (1 - 1)/1 = 0
    assert np.isclose(M_norm[0], 0.0)


def test_build_data_concat():
    data_norm = {
        "magobs": jnp.zeros((2, 3)),
        "x1": jnp.ones((2, 3)),
    }
    concat = build_data_concat(data_norm, ["magobs", "x1"], N_total=2, max_size=3)
    assert concat.shape == (2, 6)


def test_build_inputs_infer():
    data_concat = jnp.zeros((2, 6))
    mask = jnp.ones((2, 3))
    M_norm = jnp.array([0.5, -0.5])

    inputs = build_inputs_infer(data_concat, mask, M_norm, N_total=2, max_size=3)
    # Shape should be (N_total, max_size * n_cols + max_size + 1) -> (2, 6 + 3 + 1) = (2, 10)
    assert inputs.shape == (2, 10)


def test_save_and_load_hdf5(tmp_path):
    h5_file = tmp_path / "test_sim.h5"
    params = {"H0": np.array([70.0, 71.0]), "Om": np.array([0.3, 0.28])}
    data = {"magobs": np.array([[18.0, 19.0], [20.0, 21.0]])}

    save_hdf5(h5_file, params, data)
    assert h5_file.exists()

    loaded_params, loaded_data = load_hdf5(h5_file)
    for k in params:
        assert k in loaded_params
        assert np.allclose(loaded_params[k], params[k])
    for k in data:
        assert k in loaded_data
        assert np.allclose(loaded_data[k], data[k])

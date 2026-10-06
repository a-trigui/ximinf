import numpy as np
import pytest

from ximinf.selection_effects import apply_malmquist_bias


def test_apply_malmquist_bias_basic():
    results = [
        {
            "magobs": [15.0, 16.0, 24.0, 25.0],
            "z": [0.05, 0.1, 0.8, 0.9],
        }
    ]
    rng = np.random.default_rng(42)
    biased_results, masks = apply_malmquist_bias(results, loc=18.8, scale=4.5, rng=rng)

    assert len(biased_results) == 1
    assert len(masks) == 1
    assert "magobs" in biased_results[0]
    assert "z" in biased_results[0]

    mask = masks[0]
    assert len(mask) == 4
    # Length of filtered values in biased_dict should match number of True in mask
    n_detected = np.sum(mask)
    assert len(biased_results[0]["magobs"]) == n_detected
    assert len(biased_results[0]["z"]) == n_detected


def test_apply_malmquist_bias_reproducibility():
    results = [
        {"magobs": [17.0, 18.0, 19.0, 20.0]},
    ]
    biased_1, masks_1 = apply_malmquist_bias(results, loc=18.8, scale=4.5, rng=np.random.default_rng(123))
    biased_2, masks_2 = apply_malmquist_bias(results, loc=18.8, scale=4.5, rng=np.random.default_rng(123))

    assert np.array_equal(masks_1[0], masks_2[0])
    assert biased_1[0]["magobs"] == biased_2[0]["magobs"]


def test_apply_malmquist_bias_all_bright():
    # Far brighter than loc -> detection probability is virtually 1.0
    results = [{"magobs": [10.0, 11.0, 12.0]}]
    biased_results, masks = apply_malmquist_bias(results, loc=18.8, scale=4.5)
    assert np.all(masks[0])
    assert len(biased_results[0]["magobs"]) == 3


def test_apply_malmquist_bias_all_faint():
    # Far fainter than loc -> detection probability is virtually 0.0
    results = [{"magobs": [28.0, 29.0, 30.0]}]
    biased_results, masks = apply_malmquist_bias(results, loc=18.8, scale=4.5)
    assert not np.any(masks[0])
    assert len(biased_results[0]["magobs"]) == 0


def test_apply_malmquist_bias_multiple_sims():
    results = [
        {"magobs": [12.0, 28.0]},
        {"magobs": [11.0, 13.0]},
    ]
    biased_results, masks = apply_malmquist_bias(results, loc=18.8, scale=4.5)
    assert len(biased_results) == 2
    assert len(masks) == 2

import numpy as np
import jax.numpy as jnp
import pytest

from ximinf.nn_inference import (
    preprocess_groups,
    log_group_prior,
)


def test_preprocess_groups():
    param_groups = [["H0"], ["Om", "w0"]]
    global_param_names = ["H0", "Om", "w0"]

    visible_indices, group_indices = preprocess_groups(param_groups, global_param_names)

    assert len(visible_indices) == 2
    assert len(group_indices) == 2

    # Group 0: visible is [H0] -> index [0]
    assert np.array_equal(visible_indices[0], [0])
    assert np.array_equal(group_indices[0], [0])

    # Group 1: visible is [H0, Om, w0] -> indices [0, 1, 2], group is [Om, w0] -> indices [1, 2]
    assert np.array_equal(visible_indices[1], [0, 1, 2])
    assert np.array_equal(group_indices[1], [1, 2])


def test_preprocess_groups_string_names():
    # Individual group passed as string instead of list
    param_groups = ["H0", ["Om"]]
    global_param_names = ["H0", "Om"]

    visible_indices, group_indices = preprocess_groups(param_groups, global_param_names)
    assert np.array_equal(visible_indices[0], [0])
    assert np.array_equal(group_indices[0], [0])
    assert np.array_equal(visible_indices[1], [0, 1])
    assert np.array_equal(group_indices[1], [1])


def test_log_group_prior_uniform():
    priors = {
        "H0": {"range": (60.0, 80.0), "type": "uniform"},
    }
    # Within bounds
    theta = jnp.array([70.0])
    logp = log_group_prior(theta, priors, ["H0"], [0])
    assert np.isfinite(logp)
    assert np.isclose(logp, -np.log(20.0))

    # Below bounds -> -inf
    theta_low = jnp.array([55.0])
    assert log_group_prior(theta_low, priors, ["H0"], [0]) == -jnp.inf

    # Above bounds -> -inf
    theta_high = jnp.array([85.0])
    assert log_group_prior(theta_high, priors, ["H0"], [0]) == -jnp.inf


def test_log_group_prior_gaussian():
    priors = {
        "Om": {"range": (0.2, 0.4), "type": "gaussian"},
    }
    theta = jnp.array([0.3])  # exactly at the mean
    logp = log_group_prior(theta, priors, ["Om"], [0])
    assert np.isfinite(logp)


def test_log_group_prior_half_gaussian():
    priors = {
        "sigma": {"range": (0.0, 2.0), "type": "half-gaussian"},
    }
    theta_pos = jnp.array([1.0])
    assert np.isfinite(log_group_prior(theta_pos, priors, ["sigma"], [0]))

    theta_neg = jnp.array([-0.5])
    assert log_group_prior(theta_neg, priors, ["sigma"], [0]) == -jnp.inf

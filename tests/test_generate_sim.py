import numpy as np
import pandas as pd
import pytest

from ximinf import generate_sim as gs
# ------------------------------------------------------------ simulate_one
class _FakeSNe:
    def __init__(self, df):
        self.data = df


@pytest.fixture
def draw(monkeypatch):
    """Replace skysurvey.SNeIa.from_draw and record what it receives."""
    captured = {}
    df = pd.DataFrame({
        "magobs": np.linspace(18.0, 22.0, 50),
        "z": np.linspace(0.01, 0.3, 50),
        "x1": np.zeros(50),
    })

    def fake_from_draw(size, zmax, model, **kwargs):
        captured.update(size=size, zmax=zmax, model=model)
        return _FakeSNe(df.copy())

    monkeypatch.setattr(gs.skysurvey.SNeIa, "from_draw", staticmethod(fake_from_draw))
    return captured


@pytest.fixture
def defaults():
    return {"sigma_int": 0.1, "mabs": -19.3, "alpha": 0.14, "beta": 3.1,
            "gamma": 0.05, "x1_ref": 0.5,
            "cut_loc_X": 100.0, "cut_scale_X": 10.0}


def _model(**extra_kwargs):
    return {"magabs": {"kwargs": dict(extra_kwargs)}}

def test_simulate_one_output_formats(draw, defaults):
    out = gs.simulate_one({}, 0.3, 50, defaults, _model(), cols=["magobs", "nope"])
    assert set(out) == {"magobs"}                     # unknown columns ignored
    assert isinstance(out["magobs"], np.ndarray)

    assert set(gs.simulate_one({}, 0.3, 50, defaults, _model())) == {"magobs", "z", "x1"}

    df = gs.simulate_one({}, 0.3, 50, defaults, _model(), cols=["z"], out_df=True)
    assert list(df.columns) == ["z"]
    assert isinstance(gs.simulate_one({}, 0.3, 50, defaults, _model(), out_df=True),
                      pd.DataFrame)

def test_get_stretch_mode_simple():
    x1 = np.array([1.5, -0.5, 0.0])
    x1ref = np.array([0.0, 0.0, 0.0])
    mode = gs.get_stretch_mode_simple(x1, x1ref)
    assert np.array_equal(mode, [True, False, False])


def test_scan_params_uniform():
    priors = {"H0": {"range": (60.0, 80.0), "type": "uniform"}}
    params = gs.scan_params(priors, N=20, n_realisation=1)

    assert "H0" in params
    assert params["H0"].shape == (20,)
    assert np.all(params["H0"] >= 60.0)
    assert np.all(params["H0"] <= 80.0)


def test_scan_params_multiple_realisations():
    priors = {"H0": {"range": (60.0, 80.0), "type": "uniform"}}
    params = gs.scan_params(priors, N=5, n_realisation=3)

    assert params["H0"].shape == (15,)


def test_scan_params_gaussian():
    priors = {"Om": {"range": (0.2, 0.4), "type": "gaussian"}}
    params = gs.scan_params(priors, N=30)
    assert params["Om"].shape == (30,)
    assert np.all(np.isfinite(params["Om"]))


def test_scan_params_half_gaussian():
    priors_valid = {"sigma": {"range": (0.0, 1.5), "type": "half-gaussian"}}
    params = gs.scan_params(priors_valid, N=20)
    assert np.all(params["sigma"] >= 0.0)

    priors_invalid = {"sigma": {"range": (0.5, 1.5), "type": "half-gaussian"}}
    with pytest.raises(ValueError, match="Half-Gaussian prior requires low=0"):
        gs.scan_params(priors_invalid, N=10)


def test_scan_params_exponential():
    priors_valid = {"tau": {"range": (0.0, 5.0), "type": "exponential"}}
    params = gs.scan_params(priors_valid, N=20)
    assert np.all(params["tau"] >= 0.0)

    priors_invalid = {"tau": {"range": (1.0, 5.0), "type": "exponential"}}
    with pytest.raises(ValueError, match="Exponential prior requires low=0"):
        gs.scan_params(priors_invalid, N=10)


def test_scan_params_log_uniform():
    priors_valid = {"alpha": {"range": (0.01, 100.0), "type": "log-uniform"}}
    params = gs.scan_params(priors_valid, N=20)
    assert np.all(params["alpha"] >= 0.01)
    assert np.all(params["alpha"] <= 100.0)

    priors_invalid = {"alpha": {"range": (0.0, 100.0), "type": "log-uniform"}}
    with pytest.raises(ValueError, match="requires low>0"):
        gs.scan_params(priors_invalid, N=10)


def test_scan_params_unknown_prior():
    priors = {"x": {"range": (0.0, 1.0), "type": "unknown_distribution"}}
    with pytest.raises(ValueError, match="Unknown prior type"):
        gs.scan_params(priors, N=10)


# def test_evolving_rate():
#     # At z = 0, rate should equal r0
#     assert np.isclose(evolving_rate(0.0, r0=2.3e4, alpha=1.70), 2.3e4)

#     # Array input
#     z = np.array([0.0, 1.0])
#     rate = evolving_rate(z, r0=100.0, alpha=2.0)
#     assert np.isclose(rate[0], 100.0)
#     assert np.isclose(rate[1], 100.0 * (2.0 ** 2.0))

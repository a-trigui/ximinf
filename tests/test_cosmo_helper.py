import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM, LambdaCDM

from ximinf.cosmo_helper import (
    PRESETS,
    REQUIRED_KEYS,
    get_canonical,
    to_cosmologix,
    to_astropy,
    distmod,
)


def test_get_canonical_default():
    cosmo = get_canonical()
    assert isinstance(cosmo, dict)
    assert REQUIRED_KEYS.issubset(cosmo.keys())
    assert cosmo["H0"] == PRESETS["PlanckBAO18"]["H0"]
    assert cosmo["w"] == -1.0
    assert cosmo["wa"] == 0.0


def test_get_canonical_override():
    cosmo = get_canonical(H0=72.0)
    assert cosmo["H0"] == 72.0
    # Original preset should remain unchanged
    assert PRESETS["PlanckBAO18"]["H0"] != 72.0


def test_get_canonical_invalid_preset():
    with pytest.raises(ValueError, match="Unknown cosmology preset"):
        get_canonical("NonExistentPreset")


def test_get_canonical_non_lcdm_raises():
    with pytest.raises(ValueError, match="only supports LambdaCDM"):
        get_canonical(w=-0.8)

    with pytest.raises(ValueError, match="only supports LambdaCDM"):
        get_canonical(wa=0.1)


def test_to_cosmologix():
    cosmo = get_canonical()
    cx_dict = to_cosmologix(cosmo)

    for key in ["H0", "Omega_bc", "Omega_b_h2", "Omega_k", "w", "wa", "m_nu", "Tcmb", "Neff"]:
        assert key in cx_dict
        assert hasattr(cx_dict[key], "shape")
        assert np.isclose(float(cx_dict[key]), cosmo[key])


def test_to_astropy_flat():
    cosmo = get_canonical(Omega_k=0.0)
    ap_cosmo = to_astropy(cosmo)
    assert isinstance(ap_cosmo, FlatLambdaCDM)
    assert np.isclose(ap_cosmo.H0.value, cosmo["H0"])
    assert np.isclose(ap_cosmo.Om0, cosmo["Omega_bc"])


def test_to_astropy_non_flat():
    cosmo = get_canonical(Omega_k=0.05)
    ap_cosmo = to_astropy(cosmo)
    assert isinstance(ap_cosmo, LambdaCDM)
    assert not isinstance(ap_cosmo, FlatLambdaCDM)
    assert np.isclose(ap_cosmo.Ok0, 0.05)


def test_distmod_astropy():
    cosmo = get_canonical()
    z = [0.1, 0.5, 1.0]
    mu = distmod(z, cosmo, package="astropy")
    assert hasattr(mu, "shape")
    assert mu.shape == (3,)
    # Distance modulus should increase monotonically with redshift
    assert mu[0] < mu[1] < mu[2]


def test_distmod_invalid_package():
    cosmo = get_canonical()
    with pytest.raises(ValueError, match="package must be 'astropy' or 'cosmologix'"):
        distmod([0.1], cosmo, package="invalid_pkg")

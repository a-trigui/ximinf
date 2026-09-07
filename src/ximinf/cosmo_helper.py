import numpy as np

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, LambdaCDM

"""
Unified cosmology interface for cosmologix and astropy — LambdaCDM only.

cosmologix format (from parameters.get_cosmo_params, e.g. Planck18):
    {'Tcmb': 2.7255, 'Omega_bc': 0.31315017687047186, 'H0': 67.37,
     'Omega_b_h2': 0.02233, 'Omega_k': 0.0, 'w': -1.0, 'wa': 0.0,
     'm_nu': 0.06, 'Neff': 3.046}

astropy format (from Cosmology.parameters, e.g. Planck18):
    {'H0': <Quantity 67.66 km/(Mpc s)>, 'Om0': 0.30966,
     'Tcmb0': <Quantity 2.7255 K>, 'Neff': 3.046,
     'm_nu': <Quantity [0., 0., 0.06] eV>, 'Ob0': 0.04897}
"""

# ---------------------------------------------------------------------------
# 1. Single source of truth: plain dicts, keyed like cosmologix
# ---------------------------------------------------------------------------

PRESETS = {
    "PlanckBAO18": {
        "H0": 67.66,
        "Omega_bc": 0.30964144154550644,
        "Omega_b_h2": 0.02242,
        "Omega_k": 0.0,
        "w": -1.0,
        "wa": 0.0,
        "m_nu": 0.06,
        "Tcmb": 2.7255,
        "Neff": 3.046,
    },
}

REQUIRED_KEYS = {"H0", "Omega_bc", "Omega_b_h2", "Omega_k", "w", "wa",
                  "m_nu", "Tcmb", "Neff"}


def get_canonical(name: str = "PlanckBAO18", **overrides) -> dict:
    """Fetch a named preset as a dict, optionally overriding fields.

    This is the ONLY function that should be called to obtain cosmological
    parameters anywhere in the pipeline (simulation, fitting, rm_cosmo).
    Enforces LambdaCDM (w=-1, wa=0) unless explicitly overridden — but
    overriding them will raise in to_astropy(), since this module only
    supports LambdaCDM/FlatLambdaCDM.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown cosmology preset '{name}'. "
                          f"Available presets: {list(PRESETS)}")
    cosmo = dict(PRESETS[name])   # copy, don't mutate the preset
    cosmo.update(overrides)
    missing = REQUIRED_KEYS - cosmo.keys()
    if missing:
        raise ValueError(f"Canonical cosmology missing keys: {missing}")
    if not (np.isclose(cosmo["w"], -1.0) and np.isclose(cosmo["wa"], 0.0)):
        raise ValueError(
            "This module only supports LambdaCDM (w=-1, wa=0); "
            f"got w={cosmo['w']}, wa={cosmo['wa']}."
        )
    return cosmo


# ---------------------------------------------------------------------------
# 2. Translators
# ---------------------------------------------------------------------------

def to_cosmologix(cosmo: dict) -> dict:
    """Build the cosmologix parameter dict matching `cosmo` exactly."""
    import jax.numpy as jnp
    return {
        "Tcmb": jnp.asarray(cosmo["Tcmb"]),
        "Omega_bc": jnp.asarray(cosmo["Omega_bc"]),
        "H0": jnp.asarray(cosmo["H0"]),
        "Omega_b_h2": jnp.asarray(cosmo["Omega_b_h2"]),
        "Omega_k": jnp.asarray(cosmo["Omega_k"]),
        "w": jnp.asarray(cosmo["w"]),
        "wa": jnp.asarray(cosmo["wa"]),
        "m_nu": jnp.asarray(cosmo["m_nu"]),
        "Neff": jnp.asarray(cosmo["Neff"]),
    }


def to_astropy(cosmo: dict):
    """Build the astropy LambdaCDM / FlatLambdaCDM object matching
    `cosmo` exactly."""
    h = cosmo["H0"] / 100.0
    Ob0 = cosmo["Omega_b_h2"] / h**2
    m_nu = u.Quantity([0.0, 0.0, cosmo["m_nu"]], u.eV)

    common = dict(
        H0=cosmo["H0"] * u.km / u.s / u.Mpc,
        Om0=cosmo["Omega_bc"],
        Ob0=Ob0,
        Tcmb0=cosmo["Tcmb"] * u.K,
        Neff=cosmo["Neff"],
        m_nu=m_nu,
    )

    if np.isclose(cosmo["Omega_k"], 0.0):
        return FlatLambdaCDM(**common)
    return LambdaCDM(Ode0=1.0 - cosmo["Omega_bc"] - cosmo["Omega_k"],
                      **common)


# ---------------------------------------------------------------------------
# 3. Unified distance modulus + rm_cosmo
# ---------------------------------------------------------------------------

def distmod(z, cosmo: dict, package: str = "cosmologix"):
    """Distance modulus at redshifts `z`, guaranteed consistent across
    packages because both branches read from the same `cosmo` dict."""
    import jax.numpy as jnp
    if package == "astropy":
        z_np = np.asarray(z)
        ap_cosmo = to_astropy(cosmo)
        return jnp.array(ap_cosmo.distmod(z_np).value, dtype=jnp.float32)
    elif package == "cosmologix":
        from cosmologix import distances as cosmologix_distances
        cx_params = to_cosmologix(cosmo)
        return cosmologix_distances.mu(cx_params, z).astype(jnp.float32)
    else:
        raise ValueError("package must be 'astropy' or 'cosmologix'")
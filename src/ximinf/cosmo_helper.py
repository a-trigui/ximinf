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
    """
    Return a canonical cosmology parameter dictionary.

    The canonical dictionary uses the parameter naming convention adopted by
    ``cosmologix`` and serves as the common representation for cosmological
    parameters throughout the pipeline. A named preset is copied before any
    overrides are applied, so the global preset is never modified.

    Parameters
    ----------
    name : str, optional
        Name of the cosmology preset to use. Must be a key in ``PRESETS``.
        Default is ``"PlanckBAO18"``.

    **overrides
        Cosmological parameters that replace the corresponding values in the
        selected preset. All keys required by ``REQUIRED_KEYS`` must be
        present after applying the overrides.

    Returns
    -------
    dict
        Canonical cosmology parameter dictionary containing the required
        parameters:

        ``"H0"``
            Hubble constant in km / s / Mpc.

        ``"Omega_bc"``
            Total baryon-plus-cold-dark-matter matter density parameter.

        ``"Omega_b_h2"``
            Physical baryon density parameter.

        ``"Omega_k"``
            Curvature density parameter.

        ``"w"``
            Dark-energy equation-of-state parameter.

        ``"wa"``
            Evolution parameter of the dark-energy equation of state.

        ``"m_nu"``
            Neutrino mass parameter in eV.

        ``"Tcmb"``
            CMB temperature in K.

        ``"Neff"``
            Effective number of relativistic neutrino species.

    Raises
    ------
    ValueError
        If ``name`` is not a registered preset.

    ValueError
        If the resulting cosmology is missing one or more required
        parameters.

    ValueError
        If ``w`` and ``wa`` do not satisfy ``w = -1`` and ``wa = 0``.
        This module only supports LambdaCDM cosmologies.

    Notes
    -----
    The returned dictionary is a new object and can therefore be modified
    without changing the corresponding entry in ``PRESETS``.

    This function is intended to be the single source of cosmological
    parameter values used throughout the simulation and inference pipeline.
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
    """
    Convert a canonical cosmology dictionary to the ``cosmologix`` format.

    The returned dictionary preserves the parameter names and values of the
    canonical cosmology while converting the numerical values to JAX arrays.
    This representation can be passed directly to functions in
    ``cosmologix`` that expect cosmological parameters.

    Parameters
    ----------
    cosmo : dict
        Canonical cosmology parameter dictionary. It must contain the keys
        required by the cosmology interface, including ``H0``, ``Omega_bc``,
        ``Omega_b_h2``, ``Omega_k``, ``w``, ``wa``, ``m_nu``, ``Tcmb``, and
        ``Neff``.

    Returns
    -------
    dict
        Dictionary containing the same cosmological parameters as ``cosmo``,
        with each value converted to a JAX array.

    Notes
    -----
    No cosmological parameter conversion is performed beyond conversion to
    JAX arrays. In particular, the parameter names and numerical
    conventions are those of the canonical ``cosmologix`` representation.
    """
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
    """
    Convert a canonical cosmology dictionary to an Astropy cosmology object.

    The canonical cosmological parameters are converted to the conventions
    and units expected by Astropy. A flat ``FlatLambdaCDM`` instance is
    returned when ``Omega_k`` is consistent with zero; otherwise a
    ``LambdaCDM`` instance is constructed with the corresponding curvature
    contribution.

    Parameters
    ----------
    cosmo : dict
        Canonical cosmology parameter dictionary containing the cosmological
        parameters required by this module.

    Returns
    -------
    astropy.cosmology.FlatLambdaCDM or astropy.cosmology.LambdaCDM
        Astropy cosmology object corresponding to ``cosmo``.

        ``H0`` is converted to km / s / Mpc, ``Tcmb`` to K, and ``m_nu`` to
        eV. The baryon density parameter ``Ob0`` is derived from
        ``Omega_b_h2`` and ``H0``.

    Notes
    -----
    The baryon density parameter is calculated as

    .. math::

        \\Omega_b = \\frac{\\Omega_b h^2}{h^2},

    where

    .. math::

        h = H_0 / 100.

    The neutrino mass is represented as three neutrino species with masses
    ``[0, 0, m_nu]`` eV.

    For a spatially flat cosmology, ``FlatLambdaCDM`` is used. For non-zero
    curvature, ``LambdaCDM`` is used with

    .. math::

        \\Omega_\\Lambda = 1 - \\Omega_m - \\Omega_k.
    """
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
    """
    Compute the distance modulus for a canonical cosmology.

    The calculation can be performed using either ``astropy`` or
    ``cosmologix``. Both branches use the same canonical cosmology
    dictionary, providing a common parameter definition between the two
    implementations.

    Parameters
    ----------
    z : array-like
        Redshift or array of redshifts at which to evaluate the distance
        modulus.

    cosmo : dict
        Canonical cosmology parameter dictionary used to construct the
        cosmology in the selected package.

    package : {"cosmologix", "astropy"}, optional
        Package used to calculate the distance modulus. Default is
        ``"cosmologix"``.

    Returns
    -------
    jax.Array
        Distance modulus evaluated at ``z`` as a JAX array with
        ``float32`` dtype.

    Raises
    ------
    ValueError
        If ``package`` is neither ``"astropy"`` nor ``"cosmologix"``.

    Notes
    -----
    For ``package="astropy"``, the calculation is performed using the
    Astropy cosmology constructed by :func:`to_astropy`.

    For ``package="cosmologix"``, the canonical parameters are converted
    using :func:`to_cosmologix` and the distance modulus is computed with
    ``cosmologix.distances.mu`` using ``nstep=10000``.

    The two branches therefore use the same cosmological parameter values,
    but their numerical implementations may differ slightly.
    """
    import jax.numpy as jnp
    if package == "astropy":
        z_np = np.asarray(z)
        ap_cosmo = to_astropy(cosmo)
        return jnp.array(ap_cosmo.distmod(z_np).value, dtype=jnp.float32)
    elif package == "cosmologix":
        from cosmologix import distances as cosmologix_distances
        cx_params = to_cosmologix(cosmo)
        return cosmologix_distances.mu(cx_params, z, nstep=10_000).astype(jnp.float32)
    else:
        raise ValueError("package must be 'astropy' or 'cosmologix'")
# Simulation libraries
import skysurvey
import numpy as np
from pyDOE import lhs  # LHS sampler
from scipy.special import erfinv, erf
from astropy.cosmology import Planck18
from modeldag.tools import apply_gaussian_noise
from copy import deepcopy

def get_stretch_mode_simple(x1, x1ref):
    return x1>x1ref

def scan_params(priors, N, n_realisation=1, dtype=np.float32):
    """
    Generate sampled parameter sets using Latin Hypercube Sampling (LHS),
    using the per-parameter priors defined in `priors`.

    Parameters
    ----------
    priors : dict
        Mapping param name -> {'range': (low, high), 'type': str}.
        Supported types: 'uniform', 'gaussian', 'half-gaussian', 'log-uniform'.
    N : int
        Number of distinct parameter tuples.
    n_realisation : int, optional
        Number of realizations per parameter tuple.
    dtype : data-type, optional
        Numeric type for the sampled arrays (default is np.float32).

    Returns
    -------
    params_dict : dict
        Dictionary of parameter arrays of shape (N * n_realisation,).
    """
    param_names = list(priors.keys())
    n_params = len(param_names)

    # LHS unit samples in [0,1]
    unit_samples = lhs(n_params, samples=N)  # shape (N, n_params)
    samples = np.zeros_like(unit_samples)

    for i, p in enumerate(param_names):
        u = unit_samples[:, i]
        info = priors[p]
        low, high = info["range"]
        ptype = info["type"]

        if ptype == 'uniform':
            samples[:, i] = u * (high - low) + low
        elif ptype == 'gaussian':
            mu = 0.5 * (low + high)
            sigma = (high - low) / (2.0 * 1.96)
            gaussian = np.sqrt(2.0) * erfinv(2.0 * u - 1.0)
            samples[:, i] = mu + sigma * gaussian
        elif ptype == 'half-gaussian':
            if low != 0:
                raise ValueError(f"Half-Gaussian prior requires low=0, got {low}")
            sigma = high / 1.96
            gaussian = np.sqrt(2.0) * erfinv(2.0 * u - 1.0)
            samples[:, i] = np.abs(gaussian) * sigma
        elif ptype == 'positive-gaussian':
            mu = 0.5 * (low + high)
            sigma = (high - low) / (2.0 * 1.96)

            alpha = (0.0 - mu) / sigma
            Phi_alpha = 0.5 * (1.0 + erf(alpha / np.sqrt(2.0)))

            # Transform uniform samples into truncated normal
            truncated_u = Phi_alpha + u * (1.0 - Phi_alpha)
            gaussian = np.sqrt(2.0) * erfinv(2.0 * truncated_u - 1.0)

            samples[:, i] = mu + sigma * gaussian
        elif ptype == 'exponential':
            if low != 0:
                raise ValueError(f"Exponential prior requires low=0, got {low}")

            # 95% mass below `high`
            lam = -np.log(1.0 - 0.95) / high

            # Standard exponential inverse CDF
            samples[:, i] = -np.log(1.0 - u) / lam

        elif ptype == 'truncated-exponential':
            if low != 0:
                raise ValueError(f"Exponential prior requires low=0, got {low}")

            # Truncate the exponential distribution at `high`
            lam = -np.log(1.0 - 0.95) / high  # Lambda for 95% mass below `high`

            # Inverse CDF for truncated exponential: F^{-1}(u) = -log(1 - u*(1 - exp(-lam*high))) / lam
            # This ensures samples are in [0, high]
            samples[:, i] = -np.log(1.0 - u * (1.0 - np.exp(-lam * high))) / lam
        elif ptype == 'log-uniform':
            if low <= 0:
                raise ValueError(f"log-uniform prior for '{p}' requires low>0")
            samples[:, i] = low * (high / low) ** u
        else:
            raise ValueError(f"Unknown prior type '{ptype}' for parameter '{p}'")

    # Repeat for multiple realizations if needed
    params_dict = {p: np.repeat(samples[:, i], n_realisation).astype(dtype)
                   for i, p in enumerate(param_names)}

    return params_dict

def simulate_one(params_dict, z_max, M, cols, default_params, SIMULATION_MODEL,
                 errormodel=None, rng=None, N=None, i=None, out_df=False):
    """
    Simulate a single dataset of SNe Ia.
    """

    # Print progress
    if N is not None and i is not None:
        if (i + 1) % max(1, N // 10) == 0 or i == N - 1:
            print(f"Simulation {i + 1}/{N}", end="\r", flush=True)

    # Merge defaults with provided params
    params = {**default_params, **params_dict}
    
    alpha_ = float(params["alpha"])
    beta_ = float(params["beta"])
    mabs_ = float(params["mabs"])
    gamma_ = float(params["gamma"])
    sigma_int_ = float(params["sigma_int"])

    model = deepcopy(SIMULATION_MODEL)

    model["magabs"]["kwargs"]["sigmaint"] = sigma_int_
    model["magabs"]["kwargs"]["mabs"] = mabs_
    model["magabs"]["kwargs"]["alpha"] = alpha_
    model["magabs"]["kwargs"]["beta"] = beta_
    model["magabs"]["kwargs"]["gamma"] = gamma_

    if rng is None:
        rng = np.random.default_rng()
    
    model["isup"]["func"] = rng.binomial

    # Draw
    snia = skysurvey.SNeIa.from_draw(
        size=M,
        zmax=z_max,
        model=model,
    )

    # Noise
    if errormodel is None:
        df = snia.data
    else:
        if rng is None:
            rng = np.random.default_rng()
        df = apply_gaussian_noise(errormodel, data=snia.data, rng=rng)

    if out_df == True:
        return df
    else:
        return {col: list(df[col]) for col in cols if col in df}
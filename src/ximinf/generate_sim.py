# Simulation libraries
import skysurvey
import numpy as np
from pyDOE import lhs  # LHS sampler
import ztfidr.simulation as sim
import skysurvey_sniapop
from scipy.special import erfinv

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


def simulate_one(params_dict, z_max, M, cols, N=None, i=None):
    """
    Simulate a single dataset of SNe Ia.

    Parameters
    ----------
    params_dict : dict
        Dictionary of model parameters (alpha, beta, mabs, gamma, sigma_int, etc.).
    z_max : float
        Maximum redshift.
    M : int
        Number of SNe to simulate.
    cols : list of str
        List of columns to include in the output.
    N : int, optional
        Total number of simulations (for progress printing).
    i : int, optional
        Current simulation index (for progress printing).

    Returns
    -------
    data_dict : dict
        Dictionary of lists (one per column) containing the simulated data.
    """

    # Print progress
    if N is not None and i is not None:
        if (i+1) % max(1, N//10) == 0 or i == N-1:
            print(f"Simulation {i+1}/{N}", end="\r", flush=True)

    # Define default parameters including sigma_int
    default_params = {
        "alpha_low": 0.0,
        "alpha_high": 0.0,
        "beta": 0.0,
        "mabs": -19.3,
        "gamma": 0.0,
        "sigma_int": 0.0,  # default intrinsic scatter
        "x1_ref": -0.5
    }

    # Merge defaults with provided params (params_dict takes priority)
    params = {**default_params, **params_dict}

    # If a single alpha is provided, enforce alpha_low = alpha_high = alpha
    if "alpha" in params:
        params["alpha_low"] = params["alpha"]
        params["alpha_high"] = params["alpha"]

    # Ensure all are floats
    alpha_low_ = float(params["alpha_low"])
    alpha_high_ = float(params["alpha_high"])
    beta_  = float(params["beta"])
    mabs_  = float(params["mabs"])
    gamma_ = float(params["gamma"])
    sigma_int_ = float(params["sigma_int"])
    x1_ref_ = float(params["x1_ref"])

    brokenalpha_model = skysurvey_sniapop.brokenalpha_model

    # Generate SNe sample
    snia = skysurvey.SNeIa.from_draw(
        size=M,
        zmax=z_max,
        model=brokenalpha_model,
        magabs={
            "x1": "@x1",
            "c": "@c",
            "mabs": mabs_,
            "sigmaint": sigma_int_,
            "alpha_low": alpha_low_,
            "alpha_high": alpha_high_,
            "beta": beta_,
            "gamma": gamma_,
            "x1ref": x1_ref_
        }
    )

    # Apply noise
    errormodel = sim.noise_model
    # errormodel.pop("localcolor", None)
    # errormodel.pop("c", None)
    errormodel["localcolor"]["kwargs"]["a"] = 2
    errormodel["localcolor"]["kwargs"]["loc"] = 0.005
    errormodel["localcolor"]["kwargs"]["scale"] = 0.05
    noisy_snia = snia.apply_gaussian_noise(errormodel)

    df = noisy_snia.data

    # Collect requested columns as lists
    data_dict = {col: list(df[col]) for col in cols if col in df}

    return data_dict
import numpy as np
from scipy.special import expit

def apply_malmquist_bias(results, loc=18.7, scale=4.5, rng=None):
    """
    Apply a stochastic magnitude-limit selection using a sigmoid function
    instead of a Gaussian threshold.

    Each SN i is selected with probability:
        P_detect = 1 - expit((mag_i - loc) * scale)

    Rejected SNe are padded to zero.

    Parameters
    ----------
    results : list of dict
        Each element is output of `simulate_one` (combined low/high z).
    loc : float
        Sigmoid midpoint (magnitude at ~50% completeness).
    scale : float
        Sigmoid steepness (higher = sharper cut).
    rng : np.random.Generator, optional

    Returns
    -------
    biased_results : list of dict
        Same structure as input.
    masks : list of np.ndarray
        Boolean selection masks per simulation.
    """

    if rng is None:
        rng = np.random.default_rng()

    biased_results = []
    masks = []

    for data in results:

        mag = np.asarray(data["magobs"], dtype=np.float32)

        # Probability of detection per SN
        p_detect = 1.0 - expit((mag - loc) * scale)

        # Draw Bernoulli trial for each SN
        mask = rng.uniform(size=mag.shape) < p_detect
        masks.append(mask)

        # Apply mask
        biased_dict = {}
        for key, values in data.items():
            arr = np.asarray(values)
            out = np.zeros_like(arr)
            out[mask] = arr[mask]
            biased_dict[key] = list(out)

        biased_results.append(biased_dict)

    return biased_results, masks

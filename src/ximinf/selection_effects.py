import numpy as np
from scipy.special import expit

def apply_malmquist_bias(results, loc=18.8, scale=4.5, rng=None):
    """
    Apply a stochastic magnitude-limit selection using a sigmoid function.

    Each SN i is selected with probability:
        P_detect = 1 - expit((mag_i - loc) * scale)

    Rejected SNe are removed (no zero padding).

    Parameters
    ----------
    results : list of dict
        Each element is output of `simulate_one`.
    loc : float
        Sigmoid midpoint.
    scale : float
        Sigmoid steepness.
    rng : np.random.Generator, optional

    Returns
    -------
    biased_results : list of dict
        Same structure as input but containing only detected SNe.
    masks : list of np.ndarray
        Boolean selection masks per simulation.
    """

    if rng is None:
        rng = np.random.default_rng()

    biased_results = []
    masks = []

    for data in results:

        mag = np.asarray(data["magobs"], dtype=np.float32)

        # Detection probability
        p_detect = 1.0 - expit((mag - loc) * scale)

        # Bernoulli draw
        mask = rng.uniform(size=mag.shape) < p_detect
        masks.append(mask)

        # Compress directly (no zero padding)
        biased_dict = {}
        for key, values in data.items():
            arr = np.asarray(values)
            biased_dict[key] = list(arr[mask])

        biased_results.append(biased_dict)

    return biased_results, masks

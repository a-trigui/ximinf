import numpy as np
from scipy.special import expit

def apply_malmquist_bias(results, loc=18.8, scale=4.5, rng=None):
    """
    Apply a stochastic magnitude-limit selection to simulated supernovae.

    Each supernova is independently detected with probability given by a
    sigmoid selection function. Rejected supernovae are removed rather than
    represented by zero padding.

    Parameters
    ----------
    results : list of dict
        Simulation results, where each dictionary contains arrays or
        array-like values for the simulated supernova properties. The
        ``"magobs"`` entry is used to compute the detection probability.
    loc : float, optional
        Midpoint of the magnitude-selection sigmoid. At ``magobs == loc``,
        the detection probability is 0.5. Default is ``18.8``.
    scale : float, optional
        Steepness of the magnitude-selection sigmoid. Larger values produce
        a sharper transition around `loc`. Default is ``4.5``.
    rng : numpy.random.Generator, optional
        Random number generator used for the Bernoulli selection. If ``None``,
        a new default generator is created.

    Returns
    -------
    biased_results : list of dict
        Simulation results after applying the magnitude selection. Each
        dictionary has the same keys as the corresponding input dictionary,
        but only detected supernovae are retained.
    masks : list of numpy.ndarray
        Boolean selection masks for each simulation. ``True`` entries
        correspond to detected supernovae.

    Notes
    -----
    The detection probability for a supernova with observed magnitude
    ``mag`` is

    ``1 - expit((mag - loc) * scale)``.
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

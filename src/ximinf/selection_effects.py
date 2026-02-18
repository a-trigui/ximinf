import numpy as np

# def malmquist_bias(sim_data, m_lim: float, M: int, columns=None):
#     """
#     Apply magnitude-limited selection on a single simulation's data.
#     Only returns data arrays, no params.

#     Parameters
#     ----------
#     sim_data : dict
#         Dictionary of columns: {col_name: array of length M}.
#     m_lim : float
#         Limiting magnitude.
#     M : int
#         Target number of SNe after selection.
#     columns : list of str
#         Columns to select/apply magnitude cut. If None, use all columns.

#     Returns
#     -------
#     dict
#         Dictionary of selected/padded/truncated data arrays.
#     """
#     if columns is None:
#         columns = list(sim_data.keys())

#     magobs = np.asarray(sim_data['magobs'])
#     mask = magobs < m_lim
#     n_selected = mask.sum()

#     new_data = {}
#     for col in columns:
#         col_values = np.asarray(sim_data[col])
#         selected = col_values[mask]
#         if n_selected < M:
#             pad = np.zeros(M - n_selected, dtype=col_values.dtype)
#             selected = np.concatenate([selected, pad])
#         else:
#             selected = selected[:M]
#         new_data[col] = selected

#     return new_data




# def malmquist_bias_batch(simulations, m_lim: float, M: int, columns=None):
#     """
#     Apply Malmquist selection to a batch of simulations in the new dict-of-lists format.

#     Parameters
#     ----------
#     simulations : list of dict
#         List of simulations in {"data": ..., "params": ...} format
#     m_lim : float
#         Magnitude limit
#     M : int
#         Target number of SNe per simulation
#     columns : list of str
#         Columns to include. If None, use all columns in each simulation.

#     Returns
#     -------
#     list of dict
#         List of magnitude-limited simulations in the same format
#     """
#     return [malmquist_bias(sim, m_lim, M, columns) for sim in simulations]

def apply_malmquist_bias(results, mu_cuts, sigma_cut=0.1, rng=None):
    """
    Apply a stochastic magnitude-limit selection to a list of simulations.

    Each simulation i uses:
        m < mu_cut_i + N(0, sigma_cut)

    Rejected SNe are padded to zero.

    Parameters
    ----------
    results : list of dict
        Each element is output of `simulate_one` (combined low/high z).
    mu_cuts : array-like, shape (N_total,)
        One cut value per simulation.
    sigma_cut : float
        Scatter of magnitude threshold.
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

    N_total = len(results)

    # --- Handle scalar or vector mu_cuts ---
    if np.isscalar(mu_cuts):
        mu_cuts = np.full(N_total, float(mu_cuts), dtype=np.float32)
    else:
        mu_cuts = np.asarray(mu_cuts, dtype=np.float32)
        if mu_cuts.shape[0] != N_total:
            raise ValueError("mu_cuts must be scalar or have length len(results)")

    biased_results = []
    masks = []

    for i in range(N_total):

        data = results[i]
        mu_cut_i = float(mu_cuts[i])

        # Convert once
        mag = np.asarray(data["magobs"], dtype=np.float32)

        # Vectorized noisy threshold
        noisy_cut = mu_cut_i + rng.normal(
            loc=0.0,
            scale=sigma_cut,
            size=mag.shape
        )

        mask = mag < noisy_cut
        masks.append(mask)

        # Allocate output dict
        biased_dict = {}

        for key, values in data.items():
            arr = np.asarray(values)
            out = np.zeros_like(arr)
            out[mask] = arr[mask]
            biased_dict[key] = list(out)

        biased_results.append(biased_dict)

    return biased_results, masks

import numpy as np

import numpy as np

def malmquist_bias(sim_data, m_lim: float, M: int, columns=None):
    """
    Apply magnitude-limited selection on a single simulation's data.
    Only returns data arrays, no params.

    Parameters
    ----------
    sim_data : dict
        Dictionary of columns: {col_name: array of length M}.
    m_lim : float
        Limiting magnitude.
    M : int
        Target number of SNe after selection.
    columns : list of str
        Columns to select/apply magnitude cut. If None, use all columns.

    Returns
    -------
    dict
        Dictionary of selected/padded/truncated data arrays.
    """
    if columns is None:
        columns = list(sim_data.keys())

    magobs = np.asarray(sim_data['magobs'])
    mask = magobs < m_lim
    n_selected = mask.sum()

    new_data = {}
    for col in columns:
        col_values = np.asarray(sim_data[col])
        selected = col_values[mask]
        if n_selected < M:
            pad = np.zeros(M - n_selected, dtype=col_values.dtype)
            selected = np.concatenate([selected, pad])
        else:
            selected = selected[:M]
        new_data[col] = selected

    return new_data




def malmquist_bias_batch(simulations, m_lim: float, M: int, columns=None):
    """
    Apply Malmquist selection to a batch of simulations in the new dict-of-lists format.

    Parameters
    ----------
    simulations : list of dict
        List of simulations in {"data": ..., "params": ...} format
    m_lim : float
        Magnitude limit
    M : int
        Target number of SNe per simulation
    columns : list of str
        Columns to include. If None, use all columns in each simulation.

    Returns
    -------
    list of dict
        List of magnitude-limited simulations in the same format
    """
    return [malmquist_bias(sim, m_lim, M, columns) for sim in simulations]

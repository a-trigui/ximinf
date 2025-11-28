import numpy as np

def malmquist_bias(sim, m_lim: float, M: int, columns=None):
    """
    Apply magnitude-limited selection on a single simulation in the new dict-of-lists format.

    Parameters
    ----------
    sim : dict
        Single simulation: {"data": {col_name: list}, "params": {param_name: value}}
    m_lim : float
        Limiting magnitude for selection.
    M : int
        Target number of SNe after selection (padding/truncation).
    columns : list of str
        Which columns to select/apply magnitude cut on. If None, use all columns in sim['data'].

    Returns
    -------
    dict
        New simulation dict with magnitude-limited "data" and same "params".
        Data lists are padded/truncated to length M.
    """
    if columns is None:
        columns = list(sim["data"].keys())

    # Convert magobs to numpy array for masking
    magobs = np.array(sim["data"]["magobs"])
    mask = magobs < m_lim
    n_selected = mask.sum()

    new_data = {}
    for col in columns:
        col_values = np.array(sim["data"][col])
        selected = col_values[mask]
        if n_selected < M:
            pad = np.zeros(M - n_selected, dtype=col_values.dtype)
            selected = np.concatenate([selected, pad])
        else:
            selected = selected[:M]
        new_data[col] = selected.tolist()

    return {"data": new_data, "params": sim["params"].copy()}



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

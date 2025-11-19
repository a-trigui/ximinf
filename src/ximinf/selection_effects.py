import numpy as np

def malmquist_bias(flat: np.ndarray, m_lim: float, M: int, n_columns: int = 4) -> np.ndarray:
    """
    Apply magnitude-limited selection on a flattened array of a simulation.

    Parameters
    ----------
    flat : np.ndarray
        1D array containing [params..., col1..., col2..., ..., colN...] flattened.
        Each column has length equal to the number of simulated SNe.
    m_lim : float
        Limiting magnitude for selection.
    M : int
        Target number of SNe after selection (for padding/truncation).
    n_columns : int
        Number of data columns per SN (default 4: ['magobs','x1','c','z']).

    Returns
    -------
    np.ndarray
        1D array: [params..., flattened selected columns padded/truncated to M entries]
        Length = params_len + M * n_columns
    """
    # Deduce params length from the input size
    n_sne = flat.size // n_columns
    params_len = flat.size - n_columns * n_sne

    params = flat[:params_len]                   # Extract parameters
    data = flat[params_len:]                     # Remaining data
    n_sne = data.size // n_columns              # Number of SNe
    data = data.reshape((n_columns, n_sne))     # Shape: (n_columns, n_sne)

    # Apply magnitude limit (magobs is first row)
    mask = data[0, :] < m_lim
    selected = data[:, mask]                     # Shape: (n_columns, n_selected)

    n_selected = selected.shape[1]

    # Pad or truncate to exactly M
    if n_selected < M:
        pad = np.zeros((n_columns, M - n_selected), dtype=flat.dtype)
        selected = np.hstack([selected, pad])
    else:
        selected = selected[:, :M]

    # Flatten and prepend params
    return np.concatenate([params, selected.flatten()])


def malmquist_bias_batch(simulations, m_lim, M, n_columns: int = 4):
    """
    Batch version of Malmquist selection on flattened arrays.

    Parameters
    ----------
    simulations : list of np.ndarray
        Each element is a flattened array [params..., data...]
    m_lim : float
        Magnitude limit
    M : int
        Target number of SNe per simulation
    n_columns : int
        Number of data columns per SN

    Returns
    -------
    np.ndarray
        2D array: each row is a flattened, magnitude-limited simulation of length params_len + M*n_columns
    """
    result = [malmquist_bias(flat, m_lim, M, n_columns) for flat in simulations]
    return np.vstack(result).astype(np.float32)

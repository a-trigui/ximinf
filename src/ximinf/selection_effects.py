import numpy as np

def malmquist_bias(simulation, m_lim, M):
    """
    Apply a magnitude-limited selection to a single simulation vector and zero-pad.

    Parameters
    ----------
    simulation : ndarray, shape (4*M + 3,)
        Single simulation vector: [α, β, Mabs, 4*M SN entries]
        Each SN entry ordered as [magobs, x1, c, z].
    m_lim : float
        Limiting magnitude; SNe with magobs >= m_lim are removed.
    M : int
        Target number of SNe after selection (fixed padding size).

    Returns
    -------
    out : ndarray, shape (4*M + 3,)
        Same format as input, but with selection enforced and zero-padding applied.
    """
    params = simulation[:3]
    data   = simulation[3:].reshape(4, M)
    
    magobs = data[0]
    x1     = data[1]
    c      = data[2]
    z      = data[3]

    mask = magobs < m_lim

    n = mask.sum()
    mag_out = np.zeros(M, dtype=np.float32)
    x1_out  = np.zeros(M, dtype=np.float32)
    c_out   = np.zeros(M, dtype=np.float32)
    z_out   = np.zeros(M, dtype=np.float32)

    if n >= M:
        mag_out[:] = magobs[mask][:M]
        x1_out[:]  = x1[mask][:M]
        c_out[:]   = c[mask][:M]
        z_out[:]   = z[mask][:M]
    else:
        mag_out[:n] = magobs[mask]
        x1_out[:n]  = x1[mask]
        c_out[:n]   = c[mask]
        z_out[:n]   = z[mask]

    data_out = np.concatenate([mag_out, x1_out, c_out, z_out]).astype(np.float32)

    return np.concatenate([params, data_out])


def malmquist_bias_batch(simulations, m_lim, M):
    """
    Apply Malmquist bias to a batch of simulations using malmquist_bias().

    Parameters
    ----------
    simulations : ndarray, shape (N, 4*M + 3)
        Batch of simulations.
    m_lim : float
        Limiting magnitude.
    M : int
        Target number of SNe.

    Returns
    -------
    out : ndarray, shape (N, 4*M + 3)
        Batch with Malmquist selection applied.
    """
    N = simulations.shape[0]
    out = np.zeros_like(simulations, dtype=np.float32)

    for i in range(N):
        out[i] = malmquist_bias(simulations[i], m_lim, M)

    return out

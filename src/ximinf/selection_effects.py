
import numpy as np

# -------- numpy-only Malmquist selection (batch applied after simulations) --------

def malmquist_bias_batch(simulations, m_lim, M):
    """
    Apply a magnitude-limited selection to a full simulation tensor and zero-pad.

    Parameters
    ----------
    simulations : ndarray, shape (N, 4*M + 3)
        Output of the simulation pool. Each row has parameters (α,β,Mabs) then
        4*M flattened SN entries ordered as [magobs, x1, c, z].
    m_lim : float
        Limiting magnitude; SNe with magobs >= m_lim are removed.
    M : int
        Target number of SNe after selection (fixed padding size).

    Returns
    -------
    out : ndarray, shape (N, 4*M + 3)
        Same format as input, but with selection enforced and zero-padding applied.
    """

    N = simulations.shape[0]

    # split parameters and SN data
    params = simulations[:, :3]
    data   = simulations[:, 3:]                         # shape (N, 4*M)
    data   = data.reshape(N, 4, M)                      # (N, 4, M)
    magobs = data[:, 0, :]                              # (N, M)
    x1     = data[:, 1, :]
    c      = data[:, 2, :]
    z      = data[:, 3, :]

    # selection mask
    mask = magobs < m_lim                               # (N, M)

    # preallocate outputs
    mag_out = np.zeros((N, M), dtype=np.float32)
    x1_out  = np.zeros((N, M), dtype=np.float32)
    c_out   = np.zeros((N, M), dtype=np.float32)
    z_out   = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        sel = mask[i]
        # selected rows
        mo = magobs[i, sel]
        xo = x1[i, sel]
        co = c[i, sel]
        zo = z[i, sel]

        n = mo.size
        if n >= M:
            # truncate
            mag_out[i] = mo[:M]
            x1_out[i]  = xo[:M]
            c_out[i]   = co[:M]
            z_out[i]   = zo[:M]
        else:
            # fill and pad with zeros
            mag_out[i, :n] = mo
            x1_out[i, :n]  = xo
            c_out[i, :n]   = co
            z_out[i, :n]   = zo

    # flatten back
    data_out = np.concatenate(
        [mag_out, x1_out, c_out, z_out], axis=1
    ).reshape(N, 4*M)

    return np.concatenate([params, data_out], axis=1).astype(np.float32)

import numpy as np
import pandas as pd

# Simulation libraries
import skysurvey
import skysurvey_sniapop
import ztfidr.simulation as sim


def flatten_df(df: pd.DataFrame, columns: list, params: list = None) -> np.ndarray:
    """
    Flatten selected columns from a DataFrame into a single 1D numpy array.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the data.
    columns : list of str
        Column names to extract and flatten.
    prepend_params : list or None
        Optional list of parameters to prepend to the flattened array.
        
    Returns
    -------
    np.ndarray
        1D array containing [prepend_params..., col1..., col2..., ...]
    """
    arrays = [df[col].to_numpy(dtype=np.float32) for col in columns]
    flat = np.concatenate(arrays)
    
    if params is not None:
        flat = np.concatenate([np.array(params, dtype=np.float32), flat])
        
    return flat

def unflatten_array(flat_array: np.ndarray, columns: list, n_params: int = 0):
    """
    Convert a flattened array back into its original columns and optional prepended parameters.

    Parameters
    ----------
    flat_array : np.ndarray
        1D array containing the prepended parameters (optional) and column data.
    columns : list of str
        Original column names in the same order as they were flattened.
    n_prepend : int
        Number of elements at the start of the array that were prepended (e.g., α, β, mabs).

    Returns
    -------
    tuple
        If n_prepend > 0: (prepended_params, df)  
        Else: df
    """
    flat_array = flat_array.astype(np.float32)
    
    if n_params > 0:
        prepended_params = flat_array[:n_params]
        data_array = flat_array[n_params:]
    else:
        prepended_params = None
        data_array = flat_array

    n_rows = data_array.size // len(columns)
    if n_rows * len(columns) != data_array.size:
        raise ValueError("Flat array size is not compatible with number of columns")
    
    # Split array into columns
    split_arrays = np.split(data_array, len(columns))
    df = pd.DataFrame({col: arr for col, arr in zip(columns, split_arrays)})

    if n_params > 0:
        return prepended_params, df
    else:
        return df
    
def simulate_one(i, alpha_, beta_, mabs_, sigma_int, z_max, M, N=None):
    # Print progress
    if N!=None:
        if (i+1) % (N//10) == 0 or i == N-1:  # also print on last iteration
            print(f"Simulation {i+1}/{N}", end="\r", flush=True)

    brokenalpha_model = skysurvey_sniapop.brokenalpha_model

    # Model the SNe Ia
    snia = skysurvey.SNeIa.from_draw(
        size=M,
        zmax=z_max,
        model=brokenalpha_model,
        magabs={
            "x1": "@x1",
            "c": "@c",
            "mabs": float(mabs_),
            "sigmaint": sigma_int, #0.15
            "alpha_low": float(alpha_),
            "alpha_high": float(alpha_),
            "beta":  float(beta_),
        }
    )

    # Apply realistic noise
    errormodel = sim.noise_model
    errormodel.pop("localcolor", None)
    noisy_snia = snia.apply_gaussian_noise(errormodel)

    # Create dataframe with only needed columns
    df = noisy_snia.data
    flat = flatten_df(df, ['magobs', 'x1', 'c', 'z'],[alpha_, beta_, mabs_])
    return i, flat
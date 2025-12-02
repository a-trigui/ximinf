import numpy as np
import pandas as pd

# Simulation libraries
import skysurvey
import skysurvey_sniapop
import ztfidr.simulation as sim


# def flatten_df(df: pd.DataFrame, columns: list, params: list = None) -> np.ndarray:
#     """
#     Flatten selected columns from a DataFrame into a single 1D numpy array.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input dataframe containing the data.
#     columns : list of str
#         Column names to extract and flatten.
#     prepend_params : list or None
#         Optional list of parameters to prepend to the flattened array.
        
#     Returns
#     -------
#     np.ndarray
#         1D array containing [prepend_params..., col1..., col2..., ...]
#     """
#     arrays = [df[col].to_numpy(dtype=np.float32) for col in columns]
#     flat = np.concatenate(arrays)
    
#     if params is not None:
#         flat = np.concatenate([np.array(params, dtype=np.float32), flat])
        
#     return flat

# def unflatten_array(flat_array: np.ndarray, columns: list, n_points: int = 0):
#     """
#     Convert a flattened array back into its original columns and optional prepended parameters.

#     Parameters
#     ----------
#     flat_array : np.ndarray
#         1D array containing the prepended parameters (optional) and column data.
#     columns : list of str
#         Original column names in the same order as they were flattened.
#     n_points : int
#         Number of rows (SNe) in the data. If > 0, the function will deduce
#         the number of prepended parameters automatically.

#     Returns
#     -------
#     tuple
#         If prepended_params exist: (prepended_params, df)  
#         Else: df
#     """
#     flat_array = flat_array.astype(np.float32)
    
#     if n_points > 0:
#         # Deduce number of prepended parameters
#         n_params = flat_array.size - n_points * len(columns)
#         if n_params < 0:
#             raise ValueError("Number of points incompatible with flat array size")
#         prepended_params = flat_array[:n_params] if n_params > 0 else None
#         data_array = flat_array[n_params:]
#     else:
#         prepended_params = None
#         data_array = flat_array

#     n_rows = data_array.size // len(columns)
#     if n_rows * len(columns) != data_array.size:
#         raise ValueError("Flat array size is not compatible with number of columns")
    
#     # Split array into columns
#     split_arrays = np.split(data_array, len(columns))
#     df = pd.DataFrame({col: arr for col, arr in zip(columns, split_arrays)})

#     if prepended_params is not None:
#         return prepended_params, df
#     else:
#         return df
    
def simulate_one(params_dict, sigma_int, z_max, M, cols, N=None, i=None):
    """
    params_dict: dict of model parameters (alpha, beta, mabs, gamma, etc.)
    cols: list of columns to include in the output
    Returns a dict with:
        'data': dict of lists (one per column)
        'params': dict of parameter values
    """
    # Print progress
    if N is not None and i is not None:
        if (i+1) % max(1, N//10) == 0 or i == N-1:
            print(f"Simulation {i+1}/{N}", end="\r", flush=True)

    # Unpack parameters
    alpha_ = float(params_dict.get("alpha", 0))
    beta_  = float(params_dict.get("beta", 0))
    mabs_  = float(params_dict.get("mabs", 0))
    gamma_ = float(params_dict.get("gamma", 0))

    brokenalpha_model = skysurvey_sniapop.brokenalpha_model

    # Generate SNe sample
    snia = skysurvey.SNeIa.from_draw(
        size=M,
        zmax=z_max,
        model=brokenalpha_model,
        magabs={
            "x1": "@x1",
            "c": "@c",
            "mabs": mabs_,
            "sigmaint": sigma_int,
            "alpha_low": alpha_,
            "alpha_high": alpha_,
            "beta": beta_,
            "gamma": gamma_
        }
    )

    # Apply noise
    errormodel = sim.noise_model
    errormodel["localcolor"]["kwargs"]["a"] = 2
    errormodel["localcolor"]["kwargs"]["loc"] = 0.005
    errormodel["localcolor"]["kwargs"]["scale"] = 0.05
    noisy_snia = snia.apply_gaussian_noise(errormodel)

    df = noisy_snia.data

    # Collect requested columns as lists
    data_dict = {col: list(df[col]) for col in cols if col in df}

    return data_dict

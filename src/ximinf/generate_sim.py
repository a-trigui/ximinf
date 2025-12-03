# Simulation libraries
import skysurvey
import skysurvey_sniapop
import ztfidr.simulation as sim
    
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

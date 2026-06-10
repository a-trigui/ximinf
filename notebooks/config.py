from scipy import stats
import numpy as np

# #1 - Constant gaussian noise
# noise_model = {
#     "x1": {
#         "func": "np.random.normal",
#         "kwargs": {"loc": 0.1, "scale": 0.0}
#     },
#     "c": {
#         "func": "np.random.normal",
#         "kwargs": {"loc": 0.02, "scale": 0.0}
#     },
#     "magobs": {
#         "func": "np.random.normal",
#         "kwargs": {"loc": 0.1, "scale": 0.0}
#     }
# }

# 2 - Gaussian distrib
# noise_model = {
#     "x1": {
#         "func": stats.norm.rvs,
#         "kwargs": {"loc": 0.1, "scale": 0.01} 
#     },
#     "c": {
#         "func": stats.norm.rvs,
#         "kwargs": {"loc": 0.02, "scale": 0.002} 
#     },
#     "magobs": {
#         "func": stats.norm.rvs,
#         "kwargs": {"loc": 0.1, "scale": 0.01}
#     },
#     # Environments
#     "mass": 
#         { "func": stats.norm.rvs,
#             "kwargs": {"loc": 0.02, "scale": 0.001} 
#           },

#     "localcolor": 
#         { "func": stats.norm.rvs,
#             "kwargs": {"loc": 0.1, "scale": 0.005} 
#           },
# }

# 3 - Realistic distrib
noise_model = {
    # SALT paramerers
    "x1": { "func": stats.beta.rvs,
            "kwargs": {"a":1.78, "b":793.7, "loc":0.03, "scale":66.4}
          }, 

    "c": 
        { "func": stats.alpha.rvs,
            "kwargs": {"a":3.27e+00, "loc":1.71e-02, "scale":5.03e-02}
          },

    # derived

    "magobs": { "func": stats.beta.rvs,
            "kwargs": { "a":3., "b":600., "loc":0.03 , "scale":2.}
            },
    # Environments
    "mass": 
        { "func": stats.alpha.rvs,
            "kwargs": { "a":3.58e+00, "loc":1.01e-01, "scale":1.55e-03}
            },

    "localcolor": 
        { "func": stats.alpha.rvs,
                    "kwargs":{"a":3e-7, "loc": 0.01, "scale": 0.017}
                }
}

# #4 - Realistic distrib fit correc
# noise_model = {
#     # SALT paramerers
#     "x1": { "func": stats.beta.rvs,
#             "kwargs": {"a":1.78, "b":793.7, "loc":0.03, "scale":66.4}
#           }, 
#     "c": { "func": stats.alpha.rvs,
#             "kwargs": {"a":3.13, "loc":-0.001686, "scale":0.1229}
#           },

#     # derived

#     "magobs": { "func": stats.beta.rvs,
#             "kwargs": { "a":3., "b":600., "loc":0.03 , "scale":2.}
#             },
#     # Environments
#     "mass": { "func": stats.alpha.rvs,
#             "kwargs": { "a":4.436e-08, "loc":0.09788, "scale":0.00634}
#             },
#     "localcolor": { "func": stats.alpha.rvs,
#                     "kwargs":{"a":2.83e-07, "loc": 0.00349, "scale": 0.027}
#                 }
# }


params = {
    'mabs': -19.3, # from Ginolin 2025
    # 'alpha_low': -0.271,
    # 'alpha_high': -0.083,
    'alpha': -0.14,
    'x1_ref': -0.5,
    'beta': 3.31,
    'gamma': 0.175,
    'sigma_int': 0.1,
    # 'Om0': 0.3,
    # 'cut_loc_ZTF': 18.8, # from Rigault 2025
    # 'cut_scale_ZTF': 4.5,
    # 'cut_loc_SNLS': 24.1, # Fitted on Bazin 2011
    # 'cut_scale_SNLS': 4,
}

ranges = {
    'mabs': (-19.6, -19),
    # 'alpha_low': (-0.34, -0.2),
    # 'alpha_high': (-0.2, 0.04),
    'alpha': (-0.25, -0.05),
    # 'x1_ref':(-1.5, 0.5),
    'beta': (2.5, 4.1),
    'gamma': (0.0, 0.35),
    # 'sigma_int': (0.0, 0.2),
    # 'cut_loc_ZTF': (18.6, 19),
    # 'cut_scale_ZTF': (3, 6),
}

types = {
    'mabs': 'uniform',         # Uniform in [-21, -18]
    # 'alpha_low': 'uniform',
    # 'alpha_high': 'uniform',
    'alpha': 'uniform',
    # 'x1_ref': 'uniform',
    'beta': 'uniform',
    'gamma': 'uniform',
    # 'sigma_int': 'uniform',  # Positive, includes zero
    # 'cut_loc_ZTF': 'uniform',
    # 'cut_scale_ZTF': 'uniform',
}

z_max = 0.06

colour_distrib = None
#{ "func": stats.alpha.rvs, "kwargs":{"a":2.3, "loc": -0.27, "scale": 0.66}} 
#None 
    
N = 10_000
n_realisation = 5
M = 1_000

def get_quality_mask(sim_data):
    return (
        (np.asarray(sim_data["c"]) >= -0.5)
        & (np.asarray(sim_data["c"]) <= 1.0)
        & (np.asarray(sim_data["x1"]) >= -4)
        & (np.asarray(sim_data["x1"]) <= 4)
        & (np.asarray(sim_data["x1_err"]) <= 1)
        & (np.asarray(sim_data["c_err"]) <= 0.1)
    )
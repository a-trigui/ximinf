from scipy import stats
import numpy as np
import skysurvey

import ximinf.cosmo_helper as ch
cosmo = ch.get_canonical("PlanckBAO18")
astropy_cosmo = ch.to_astropy(cosmo)

def uniform_distrib(xx, loc, scale):
    return xx, stats.uniform.pdf(xx, loc, scale)

def normal_distrib(xx, loc, scale):
    return xx, stats.norm.pdf(xx, loc, scale)

columns = ['magobs', 'magobs_err','x1', 'x1_err', 'c', 'c_err', 'isup', 'z'] #, 'isup' , 'prompt' , 'localcolor', 'localcolor_err', 'mass', 'mass_err', 

# 1 - Realistic distrib
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
    # "mass": 
    #     { "func": stats.alpha.rvs,
    #         "kwargs": { "a":3.58e+00, "loc":1.01e-01, "scale":1.55e-03}
    #         },

    # "localcolor": 
    #     { "func": stats.alpha.rvs,
    #                 "kwargs":{"a":3e-7, "loc": 0.01, "scale": 0.017}
    #             }
}

# # 2 - Small gaussian
# noise_model = {
#     # SALT paramerers
#     "x1": { "func": stats.norm.rvs,
#             "kwargs": {"loc":0.1, "scale":0.005}
#           }, 

#     # x1 = {"func": normal_distrib,
#     #         "kwargs": {"xx":"0:1:0.001", "loc":0.1 , "scale":0.01}
#     #          },

#     "c": 
#         { "func": stats.norm.rvs,
#             "kwargs": {"loc":0.03, "scale":0.002}
#           },

#     # derived

#     "magobs": { "func": stats.norm.rvs,
#             "kwargs": {"loc":0.03 , "scale":0.002}
#             },
# }

# # 3 - Small uniform
# noise_model = {
#     # SALT paramerers
#     "x1": { "func": stats.uniform.rvs,
#             "kwargs": {"loc":0.05, "scale":0.2} #{"loc":0.1, "scale":0.02}
#           }, 

#     "c": 
#         { "func": stats.uniform.rvs,
#             "kwargs": {"loc":0.005, "scale":0.03} #{"loc":0.03, "scale":0.01}
#           },

#     # derived

#     "magobs": { "func": stats.uniform.rvs,
#             "kwargs": {"loc":0.02 , "scale":0.02} #{"loc":0.03 , "scale":0.01}
#             },
# }

params = {
    'mabs': -19.3,
    'alpha': -0.161,
    'beta': 3.05 ,
    'gamma': 0.143,
    'sigma_int': 0.1, #0.1,
}

# Normal M=1000
ranges = {
    'mabs': np.array([-19.4, -19.2]),
    'alpha': np.array([-0.25, -0.07]),
    'beta': np.array([2.4, 3.6]),
    'gamma': np.array([-0.1, 0.3]),
}

# # Small M=2000
# ranges = {
#     'mabs': np.array([-19.4, -19.2]),
#     'alpha': np.array([-0.2, -0.13]),
#     'beta': np.array([2.8, 3.2]),
#     # 'gamma': np.array([0.05, 0.25]),
# }

# # Very small M=2000 sigma_int=0
# ranges = {
#     'mabs': np.array([-19.35, -19.25]),
#     'alpha': np.array([-0.18, -0.14]),
#     'beta': np.array([2.85, 3.15]),
#     # 'gamma': np.array([0.1, 0.2]),
# }

# BIG for TARP
# ranges = {
#     'mabs': np.array([-19.7, -19.0]),
#     'alpha': np.array([-0.5, -0.0]),
#     'beta': np.array([2.0, 4.0]),
#     # 'gamma': np.array([-0.1, 0.4]),
# }

# ranges = {
#     'mabs': (-19.38, -19.22),
#     'alpha': (-0.22, -0.10),
#     'beta': (2.6, 3.4),
#     'gamma': (0.0, 0.3),
# }

# Around Ginolin 2025
types = {
    'mabs': 'uniform',
    'alpha': 'uniform',
    'beta': 'uniform',
    'gamma': 'uniform',
}

z_max = 0.06

colour_distrib = None

N = 200_000
M = 1_000 #2_000

def get_quality_mask(sim_data):
    return (
        (np.asarray(sim_data["c"]) >= -0.5)
        & (np.asarray(sim_data["c"]) <= 1.0)
        & (np.asarray(sim_data["x1"]) >= -4)
        & (np.asarray(sim_data["x1"]) <= 4)
        & (np.asarray(sim_data["x1_err"]) <= 1)
        & (np.asarray(sim_data["c_err"]) <= 0.1)
    )

SIMULATION_MODEL = dict( redshift = {"kwargs": {"zmax":z_max}, "as":"z"},
                              
                   x1 = {"func": skysurvey.target.snia.SNeIaStretch.nicolas2021,
                        "kwargs": {"xx":"-4:4:0.005", "mu1":0.42, "sigma1":0.54, 
                     "mu2":-1.24, "sigma2":0.73, "a":0.2,
                     "fprompt":0.5}}, 

                    # x1 = {"func": stats.norm.rvs,
                    #     "kwargs": {"loc":0.0 , "scale":1}
                    #      },

                    # x1 = {"func": stats.uniform.pdf,
                    #     "kwargs": {"x":"-4:4:0.005"} #, "loc":-3 , "scale":6}
                    #      },

                    # x1 = {"func": uniform_distrib,
                    #     "kwargs": {"xx":"-3:3:0.001", "loc":-3 , "scale":6}
                    #      },
                   
                   c = {"func": skysurvey.target.snia.SNeIaColor.intrinsic_and_dust,
                       "kwargs": {"xx":"-0.3:1:0.001", "cint":-0.085, "sigmaint":0.05, "tau":0.155}},

                    # c = {"func": stats.norm.rvs,
                    #     "kwargs": {"loc":0.0 , "scale":0.2}
                    #      },

                    # c = {"func": stats.uniform.pdf,
                    #     "kwargs": {"x":"-1:1:0.005"}#, "loc":-0.3 , "scale":1}
                    #      },

                     # c = {"func": uniform_distrib,
                     #    "kwargs": {"xx":"-0.3:0.8:0.001", "loc":-0.3 , "scale":1}
                     #     },

                   isup = {"func": np.random.binomial, 
                         "kwargs": {"n":1, "p":0.5} },
                       
                   magabs = {"func": skysurvey.target.snia.SNeIaMagnitude.tripp_and_step,
                             "kwargs": {"x1": "@x1", "c": "@c", "isup": "@isup",
                                        "mabs":params['mabs'], 'alpha':params['alpha'], 'beta':params['beta'], 'gamma':params['gamma'], 'sigmaint':params['sigma_int']}
                            },

                    # magabs = {"func": skysurvey.target.snia.SNeIaMagnitude.tripp1998,
                    #          "kwargs": {"x1": "@x1", "c": "@c",
                    #                     "mabs":params['mabs'], 'alpha':params['alpha'], 'beta':params['beta'], 'sigmaint':params['sigma_int']}
                    #         },
                           
                   magobs = {"func": "magabs_to_magobs", # str-> method of the class
                             "kwargs": {"z":"@z", "magabs":"@magabs"},
                            'cosmology': astropy_cosmo,
                            },
                       )

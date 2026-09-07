from scipy import stats
import numpy as np
import skysurvey

import ximinf.cosmo_helper as ch
cosmo = ch.get_canonical("PlanckBAO18")
astropy_cosmo = ch.to_astropy(cosmo)

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

params = {
    'mabs': -19.3,
    'alpha': -0.161,
    'beta': 3.05 ,
    'gamma': 0.143,
    'sigma_int': 0.1,
}

ranges = {
    'mabs': (-19.4, -19.2),
    'alpha': (-0.25, -0.07),
    'beta': (2.4, 3.6),
    'gamma': (-0.1, 0.3),
}

# Around Ginolin 2025
types = {
    'mabs': 'uniform',
    'alpha': 'uniform',
    'beta': 'uniform',
    'gamma': 'uniform',
}

z_max = 0.06

colour_distrib = None

N = 20_000
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

SIMULATION_MODEL = dict( redshift = {"kwargs": {"zmax":z_max}, "as":"z"},
                              
                   x1 = {"func": skysurvey.target.snia.SNeIaStretch.nicolas2021,
                        "kwargs": {"xx":"-4:4:0.005", "mu1":0.42, "sigma1":0.54, 
                     "mu2":-1.24, "sigma2":0.73, "a":0.2,
                     "fprompt":0.5}}, 
                   
                   c = {"func": skysurvey.target.snia.SNeIaColor.intrinsic_and_dust,
                       "kwargs": {"xx":"-0.3:1:0.001", "cint":-0.085, "sigmaint":0.05, "tau":0.155}},

                   isup = {"func": np.random.binomial, 
                         "kwargs": {"n":1, "p":0.5} },
                       
                   magabs = {"func": skysurvey.target.snia.SNeIaMagnitude.tripp_and_step,
                             "kwargs": {"x1": "@x1", "c": "@c", "isup": "@isup",
                                        "mabs":params['mabs'], 'alpha':params['alpha'], 'beta':params['beta'], 'gamma':params['gamma'], 'sigmaint':params['sigma_int']}
                            },
                           
                   magobs = {"func": "magabs_to_magobs", # str-> method of the class
                             "kwargs": {"z":"@z", "magabs":"@magabs"},
                            'cosmology': astropy_cosmo,
                            },
                       )
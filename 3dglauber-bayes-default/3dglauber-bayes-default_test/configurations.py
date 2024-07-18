#!/usr/bin/env python3
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from bins_and_cuts import *
#from Read_calculations_with_flow import obs_indices_dict


#############################################################################
#############################################################################

## Configuration file with analysis settings and flags

#############################################################################
#############################################################################

logging.getLogger().setLevel(logging.INFO)
# scramble=True # to scramble the Sobol sequence or not

# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = "<f8"
int_t = "<i8"
complex_t = "<c16"

# fix the random seed for cross validation, that sets are deleted consistently
np.random.seed(1)

# Work, Design, and Exp directories
workdir = Path(os.getenv("WORKDIR", "."))

# Directory for storing/providing parameter sets for training and validation
design_dir = str(workdir / "production_designs/ProductionMaxPro-Mod")

# Directory housing experimental data (or pseudodata) in csv format to be read into bayes_dtype format
dir_obs_exp = "HIC_experimental_data"
#dir_obs_exp = "HIC_pseudodata_val_pt_0"
#dir_obs_exp = "HIC_pseudodata_val_pt_2"
#dir_obs_exp = "HIC_pseudodata_val_pt_10"
#dir_obs_exp = "HIC_pseudodata_val_pt_25"
#dir_obs_exp = "HIC_pseudodata_val_pt_41"

# Flag to run closure test and validation point number to use
run_closure = False
closure_val_pt = 0

####################################
### USEFUL LABELS / DICTIONARIES ###
####################################

# only using data from these experimental collabs
# (if the bayes_dtype is changed to include experiment this array is no longer needed - currently used in
# reading exp data into the current bayes_dtype format)
expt_and_obs_for_system = {



        "Au-Au-200": {

            # "BRAHMS" : {
                # "dNdeta_eta_cen_00_05_BRAH" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_05_10_BRAH" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_10_20_BRAH" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_20_30_BRAH" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_30_40_BRAH" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_40_50_BRAH" : BRAHMS_mult_rap_bins,

                # "dNdeta_eta_cen_00_05_BRAH_2" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_05_10_BRAH_2" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_10_20_BRAH_2" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_20_30_BRAH_2" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_30_40_BRAH_2" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_40_50_BRAH_2" : BRAHMS_mult_rap_bins,
                #
                # "dNdeta_eta_cen_00_05_BRAH_3" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_05_10_BRAH_3" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_10_20_BRAH_3" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_20_30_BRAH_3" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_30_40_BRAH_3" : BRAHMS_mult_rap_bins,
                # "dNdeta_eta_cen_40_50_BRAH_3" : BRAHMS_mult_rap_bins,


                # "dNdeta_eta_cen_00_05_frwd_BRAH" : BRAHMS_mult_large_rap_bins,
                # "dNdeta_eta_cen_05_10_frwd_BRAH" : BRAHMS_mult_large_rap_bins,
                # "dNdeta_eta_cen_10_20_frwd_BRAH" : BRAHMS_mult_large_rap_bins,
                # "dNdeta_eta_cen_20_30_frwd_BRAH" : BRAHMS_mult_large_rap_bins,
                # "dNdeta_eta_cen_30_40_frwd_BRAH" : BRAHMS_mult_large_rap_bins,
                # "dNdeta_eta_cen_40_50_frwd_BRAH" : BRAHMS_mult_large_rap_bins,
            # },

            "PHOBOS" : {
                "v22_eta_cen_03_15_PHOB" : PHOBOS_v2_cen,
                "v22_eta_cen_15_25_PHOB" : PHOBOS_v2_cen,
                "v22_eta_cen_25_50_PHOB" : PHOBOS_v2_cen,

                "dNdeta_eta_cen_00_03_PHOB" : PHOBOS_mult_rap_bins,
                "dNdeta_eta_cen_03_06_PHOB" : PHOBOS_mult_rap_bins,
                "dNdeta_eta_cen_06_10_PHOB" : PHOBOS_mult_rap_bins,
                "dNdeta_eta_cen_10_15_PHOB" : PHOBOS_mult_rap_bins,
                "dNdeta_eta_cen_15_20_PHOB" : PHOBOS_mult_rap_bins,
                "dNdeta_eta_cen_20_25_PHOB" : PHOBOS_mult_rap_bins,
                "dNdeta_eta_cen_25_30_PHOB" : PHOBOS_mult_rap_bins,
                "dNdeta_eta_cen_30_35_PHOB" : PHOBOS_mult_rap_bins,
                "dNdeta_eta_cen_35_40_PHOB" : PHOBOS_mult_rap_bins,
                "dNdeta_eta_cen_40_45_PHOB" : PHOBOS_mult_rap_bins,
                "dNdeta_eta_cen_45_50_PHOB" : PHOBOS_mult_rap_bins,
            },

            "STAR" : {
                "v22_eta_cen_20_70_STAR" : STAR_v22_rap_bins,

                "v22_int_STAR" : STAR_meanpT_cen[:6],
                "v32_int_STAR" : STAR_meanpT_cen[:6],

                #"meanpT_pi_STAR" : STAR_meanpT_cen[:7],
                #"meanpT_k_STAR" : STAR_meanpT_cen[:7],
                #"r2_eta_cen_10_40_STAR" : STAR_rn_eta_bins,
                #"r3_eta_cen_10_40_STAR" : STAR_rn_eta_bins,

            },

            "PHENIX" : {
                "v22_pt_cen_00_10_PHEN" : PHENIX_vn_pt[:5],
                "v22_pt_cen_10_20_PHEN" : PHENIX_vn_pt[:5],
                "v22_pt_cen_20_30_PHEN" : PHENIX_vn_pt[:5],
                "v22_pt_cen_30_40_PHEN" : PHENIX_vn_pt[:5],
                "v22_pt_cen_40_50_PHEN" : PHENIX_vn_pt[:5],
                "v22_pt_cen_50_60_PHEN" : PHENIX_vn_pt[:5],

                # "v32_pt_cen_00_10_PHEN" : PHENIX_vn_pt,
                # "v32_pt_cen_10_20_PHEN" : PHENIX_vn_pt,
                # "v32_pt_cen_20_30_PHEN" : PHENIX_vn_pt,
                # "v32_pt_cen_30_40_PHEN" : PHENIX_vn_pt,
                # "v32_pt_cen_40_50_PHEN" : PHENIX_vn_pt,
                # "v32_pt_cen_50_60_PHEN" : PHENIX_vn_pt,
                #
                # "v42_pt_cen_00_10_PHEN" : PHENIX_vn_pt,
                # "v42_pt_cen_10_20_PHEN" : PHENIX_vn_pt,
                # "v42_pt_cen_20_30_PHEN" : PHENIX_vn_pt,
                # "v42_pt_cen_30_40_PHEN" : PHENIX_vn_pt,
                # "v42_pt_cen_40_50_PHEN" : PHENIX_vn_pt,
                #"v42_pt_cen_50_60_PHEN" : PHENIX_vn_pt,

                "meanpT_pi_PHEN" : PHENIX_meanpT_cen[:8],
                #"meanpT_k_PHEN"  : PHENIX_meanpT_cen[:8],
            },
        },

        "d-Au-200": {

            "PHOBOS" : {
                'dNdeta_eta_cen_00_20_PHOB' : PHOBOS_mult_rap_bins,
                #'dNdeta_eta_cen_20_40_PHOB' : PHOBOS_mult_rap_bins,
                #'dNdeta_eta_cen_40_60_PHOB' : PHOBOS_mult_rap_bins,
            },

            "STAR" : {
                'v22_pt_cen_00_10_STAR'     : STAR_dAu_vn_pt[:5],
                # 'v32_pt_cen_00_10_STAR'     : STAR_dAu_vn_pt,
            },

            "PHENIX" : {
                'v22_eta_cen_00_05_PHEN'    : PHENIX_v22_rap_bins[10:],

                'v22_pt_cen_00_05_PHEN'     : PHENIX_dAu_vn_pt[:5],
                # 'v32_pt_cen_00_05_PHEN'     : PHENIX_dAu_vn_pt,
            }
        },

    }



# legacy df labels
idf_label = {0: "Grad", 1: "Chapman-Enskog R.T.A"}
idf_label_short = {0: "Grad", 1: "CE"}

####################################
### SWITCHES AND OPTIONS         ###
####################################

# how many versions of the model are run, for instance
number_of_models_per_run = 1

# default set this to zero, analysis run using Grad viscous corrections
idf = 0
logging.info(f"Using idf = {idf}: {idf_label[idf]}")

# the Collision systems
systems = [
     ('Au', 'Au', 200),
     ('d', 'Au', 200),
     # ("Pb", "Pb", 5020),
]

# create system string
system_strs = ["{:s}-{:s}-{:d}".format(*s) for s in systems]
num_systems = len(system_strs)

# these are problematic points (legacy from previous analysis)
nan_sets_by_deltaf = {0: set([65, 78, 207, 322, 210, 229, 256, 316])}
nan_design_pts_set = nan_sets_by_deltaf[idf]
unfinished_events_design_pts_set = set([])
strange_features_design_pts_set = set([])

delete_design_pts_set = nan_design_pts_set.union(
    unfinished_events_design_pts_set.union(strange_features_design_pts_set)
)

delete_design_pts_validation_set = []  # [10, 68, 93] # idf 0

# directory and file structure for storing the designs and model calculations
class systems_setting(dict):
    def __init__(self, A, B, sqrts):
        super().__setitem__("proj", A)
        super().__setitem__("targ", B)
        super().__setitem__("sqrts", sqrts)
        sysdir = "/design_pts_{:s}_{:s}_{:d}_production".format(A, B, sqrts)
        super().__setitem__(
            "main_design_file",
            design_dir
            + sysdir
            + "/design_points_main_{:s}{:s}-{:d}.dat".format(A, B, sqrts),
        )
        super().__setitem__(
            "main_range_file",
            design_dir
            + sysdir
            + "/design_ranges_main_{:s}{:s}-{:d}.dat".format(A, B, sqrts),
        )
        super().__setitem__(
            "validation_design_file",
            design_dir
            + sysdir
            + "/design_points_validation_{:s}{:s}-{:d}.dat".format(A, B, sqrts),
        )
        super().__setitem__(
            "validation_range_file",
            design_dir
            + sysdir
            + "/design_ranges_validation_{:s}{:s}-{:d}.dat".format(A, B, sqrts),
        )
        try:
            with open(
                design_dir
                + sysdir
                + "/design_labels_{:s}{:s}-{:d}.dat".format(A, B, sqrts),
                "r",
            ) as f:
                labels = [r"" + line[:-1] for line in f]
            super().__setitem__("labels", labels)
        except:
            pass
            #logging.info("can't load design point labels")

    def __setitem__(self, key, value):
        if key == "run_id":
            super().__setitem__(
                "main_events_dir",
                str(workdir / "model_calculations/{:s}/Events/main/".format(value)),
            )
            super().__setitem__(
                "validation_events_dir",
                str(
                    workdir / "model_calculations/{:s}/Events/validation/".format(value)
                ),
            )
            super().__setitem__(
                "main_obs_file",
                str(workdir / "model_calculations/{:s}/Obs/main.dat".format(value)),
            )
            super().__setitem__(
                "validation_obs_file",
                str(
                    workdir / "model_calculations/{:s}/Obs/validation.dat".format(value)
                ),
            )
        else:
            super().__setitem__(key, value)

# Some calculation and emulator details for each system (number of principal components, npc, set here)
SystemsInfo = {"{:s}-{:s}-{:d}".format(*s): systems_setting(*s) for s in systems}

if "d-Au-200" in system_strs:
    SystemsInfo["d-Au-200"]["run_id"] = "production_375pts_d_Au_200"
    SystemsInfo["d-Au-200"]["run_dir"] = "production_375pts_d_Au_200"
    SystemsInfo["d-Au-200"]["n_design"] = 414
    SystemsInfo["d-Au-200"]["n_validation"] = 5
    SystemsInfo["d-Au-200"]["design_remove_idx"] = [] #list(delete_design_pts_set)
    SystemsInfo["d-Au-200"]["npc"] = 9
    SystemsInfo["d-Au-200"]["MAP_obs_file"] =  "None"
    SystemsInfo["d-Au-200"]["labels"] =  "None"


if "Au-Au-200" in system_strs:
    SystemsInfo["Au-Au-200"]["run_id"] = "production_375pts_Au_Au_200"
    SystemsInfo["Au-Au-200"]["run_dir"] = "production_375pts_Au_Au_200"
    SystemsInfo["Au-Au-200"]["n_design"] = 414
    SystemsInfo["Au-Au-200"]["n_validation"] = 5
    SystemsInfo["Au-Au-200"]["design_remove_idx"] = [] #list(delete_design_pts_set)
    SystemsInfo["Au-Au-200"]["npc"] = 6
    SystemsInfo["Au-Au-200"]["MAP_obs_file"] =  "None"
    SystemsInfo["Au-Au-200"]["labels"] =  "None"

# legacy system
if "Pb-Pb-5020" in system_strs:
    SystemsInfo["Pb-Pb-5020"]["MAP_obs_file"] = (
        str(workdir / "model_calculations/MAP")
        + "/"
        + idf_label_short[idf]
        + "/Obs/obs_Pb-Pb-5020.dat"
    )

# print out systems info when running
#logging.info("SystemsInfo = ")
#logging.info(SystemsInfo)

###############################################################################
############### BAYES #########################################################

# if True, we will use the emcee Parallel Tempering Sampler to sample the posterior
# this allows the estimation of the Bayesian evidence
usePTSampler = False

# if True: perform emulator validation
# if False: use experimental data for parameter estimation
# unclear use? not used in analysis unless doing emulator validation
validation = False

# if true, we will validate emulator against points in the training set
pseudovalidation = False
# if true, we will omit 20% of the training design when training emulator
crossvalidation = False

# variable that can be used for validation, not used as default
fixed_validation_pt = 3

# if this switch is turned on, some parameters will be fixed
# to certain values in the parameter estimation. see bayes_mcmc.py
hold_parameters = False
# hold are pairs of parameter (index, value)
# count the index correctly when have multiple systems!
# e.g [(1, 10.5), (5, 0.3)] will hold parameter[1] at 10.5, and parameter[5] at 0.3
# hold_parameters_set = [(7, 0.0), (8, 0.154), (9, 0.0), (15, 0.0), (16, 5.0)] #these should hold the parameters to Jonah's prior for LHC+RHIC
# hold_parameters_set = [(6, 0.0), (7, 0.154), (8, 0.0), (14, 0.0), (15, 5.0)] #these should hold the parameters to Jonah's prior for LHC only

hold_parameters_set = []  # fix g_strong to 2 for data comparison

# logging information about emulator validation
if validation:
    logging.info("Performing emulator validation type ...")
    if pseudovalidation:
        logging.info("... pseudo-validation")
        pass
    elif crossvalidation:
        logging.info("... cross-validation")
        cross_validation_pts = np.random.choice(
            n_design_pts_main, n_design_pts_main // 5, replace=False
        )
        delete_design_pts_set = cross_validation_pts  # omit these points from training
    else:
        validation_pt = fixed_validation_pt
        logging.info(
            "... independent-validation, using validation_pt = " + str(validation_pt)
        )

# if this switch is True, all experimental errors will be set to zero
set_exp_error_to_zero = False

# if this switch is True, then when performing MCMC each experimental error
# will be multiplied by the corresponding factor.
# unclear if this remains working? need to be checked
change_exp_error = False
change_exp_error_vals = {
    "Au-Au-200": {},
    "Pb-Pb-2760": {"dN_dy_proton": 1.0e-1, "mean_pT_proton": 1.0e-1},
}


if hold_parameters:
    logging.info("Warning: holding parameters to fixed values: ")
    logging.info(hold_parameters_set)

change_parameters_range = False
if change_parameters_range:
    # first check if a file exists to change the ranges
    if os.path.exists("restricted_prior_ranges/prior_range.dat"):
        par_dtype = [("idx", int), ("min", float), ("max", float)]
        with open("restricted_prior_ranges/prior_range.dat", "r") as f:
            change_parameters_range_set = np.fromiter(
                (tuple(l.split()) for l in f if not l.startswith("#")), dtype=par_dtype
            )
    # if this file does not exist, use the hardcoded values
    else:
        # the set below will fix the ranges to be similar to those used in J. Bernhard's study
        #                                   p               w               tau_r       zeta/s max    zeta/s T_peak      zeta/s width
        change_parameters_range_set = [
            (1, -0.5, 0.5),
            (3, 0.5, 1.0),
            (5, 0.3, 1.5),
            (11, 0.01, 0.1),
            (12, 0.15, 0.2),
            (13, 0.025, 0.1),
        ]

    logging.info("Warning: changing parameter ranges: ")
    logging.info(change_parameters_range_set)


# VISCOSITY DISCRETIZATION
# if this switch is turned on, the emulator will be trained on the values of
# eta/s (T_i) and zeta/s (T_i), where T_i are a grid of temperatures, rather
# than the parameters such as slope, width, etc...
do_transform_design = True

""" Comment from Derek Everett:
BTW something that Jonah, Scott, Weiayo, Derek did *not* do was rescale the
parameters to [-1,1] before training the GP, but this would probably be a really good idea to do.
Say like the trento normalization [10,20], first rescale the design matrix to [-1, 1].
Then you will be able to interpret the GP correlation lengths straightforwardly
(they should be O(1))

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
"""

# TRANSFORM MULTIPLICITY
# if this switch is turned on, the emulator will be trained on log(1 + dY_dx) (or log(dY_dx) in default analysis)
# where dY_dx includes dET_deta, dNch_deta, dN_dy_pion, etc...
transform_multiplicities = True

# Experimental covariance correlations
# this switches on/off parameterized experimental covariance btw. centrality bins and groups
assume_corr_exp_error = False
cent_corr_length = 0.5  # this is the correlation length between centrality bins

# define groups of observables which are assumed to have correlated experimental error, if used
# not used in current analysis
expt_obs_corr_group = {
    "dNch_deta": "yields",
    "dET_deta": "yields",
    "dN_dy_pion": "yields",
    "dN_dy_kaon": "yields",
    "dN_dy_proton": "yields",
    "mean_pT_pion": "mean_pT",
    "mean_pT_kaon": "mean_pT",
    "mean_pT_proton": "mean_pT",
    "pT_fluct": "pT_fluct",
    "v22": "flows",
    "v32": "flows",
    "v42": "flows",
}


# The data format for emulator training and bayesian analysis
bayes_dtype = [
    (
        s,
        [
            (obs, [("mean", float_t, len(cent_list)), ("err", float_t, len(cent_list))])
            for obs, cent_list in obs_cent_list[s].items()
        ],
        number_of_models_per_run,
    )
    for s in system_strs
]


# The active observables used in Bayes analysis (MCMC) (unused in default analysis)
active_obs_list = {sys: list(obs_cent_list[sys].keys()) for sys in system_strs}

# excluding observables from analysis
# try exluding PHENIX dN/dy proton from fit
exclude_data = False
if exclude_data:
    for s in system_strs:
        #if s == "Au-Au-200":
            #active_obs_list[s].remove("dN_dy_proton")
            #active_obs_list[s].remove("mean_pT_proton")

        if s == "Pb-Pb-2760":
            active_obs_list[s].remove("dET_deta")
            # active_obs_list[s].remove('chi532')

logging.info(f"The active observable list for calibration: {active_obs_list}")

# Functions for calculating viscosities using parametrization
@np.vectorize
def zeta_over_s(T, zmax, T0, width, asym):
    DeltaT = T - T0
    sign = 1 if DeltaT > 0 else -1
    x = DeltaT / (width * (1.0 + asym * sign))
    return zmax / (1.0 + x**2)

@np.vectorize
def eta_over_s(T, T_k, alow, ahigh, etas_k):
    if T < T_k:
        y = etas_k + alow * (T - T_k)
    else:
        y = etas_k + ahigh * (T - T_k)
    if y > 0:
        return y
    else:
        return 0.0

@np.vectorize
def taupi(T, T_k, alow, ahigh, etas_k, bpi):
    return bpi * eta_over_s(T, T_k, alow, ahigh, etas_k) / T


# Load design
def load_design(system_str, pset="main"):  # or validation
    design_file = (
        SystemsInfo[system_str]["main_design_file"]
        if pset == "main"
        else SystemsInfo[system_str]["validation_design_file"]
    )
    range_file = (
        SystemsInfo[system_str]["main_range_file"]
        if pset == "main"
        else SystemsInfo[system_str]["validation_range_file"]
    )
    logging.info("Loading {:s} points from {:s}".format(pset, design_file))
    logging.info("Loading {:s} ranges from {:s}".format(pset, range_file))
    labels = SystemsInfo[system_str]["labels"]
    # design
    design = pd.read_csv(design_file)
    design = design.drop("idx", axis=1)
    #logging.info("Summary of design: ")
    design.describe()
    design_range = pd.read_csv(range_file)
    design_max = design_range["max"].values
    design_min = design_range["min"].values
    ## Here maybe add betas for the design
    return design, design_max, design_min, labels

# Function used for viscosity discretization
def transform_design(X):
    # pop out the viscous parameters
    indices = [0,1,2,3,4,5,6,7,8,9,10]  # IPG, these are kept
    extra_idx = [19]
    new_design_X = X[:, indices]
    new_design_extras = X[:, extra_idx]

    # now append the values of eta/s and zeta/s at various temperatures
    num_T = 10
    Temperature_grid = np.linspace(0.12, 0.4, num_T) # previous min at 1.35
    eta_vals = []
    zeta_vals = []
    for T in Temperature_grid:
        eta_vals.append(eta_over_s(T, X[:, 11], X[:, 12], X[:, 13], X[:, 14]))  # IPG
    for T in Temperature_grid:
        zeta_vals.append(zeta_over_s(T, X[:, 15], X[:, 16], X[:, 17], X[:, 18]))  # IPG

    eta_vals = np.array(eta_vals).T
    zeta_vals = np.array(zeta_vals).T

    new_design_X = np.concatenate((new_design_X, eta_vals), axis=1)
    new_design_X = np.concatenate((new_design_X, zeta_vals), axis=1)
    new_design_X = np.concatenate((new_design_X, new_design_extras), axis=1)
    return new_design_X

def transform_design_min_max(design_max,design_min):
    # pop out the viscous parameters
    indices = [0,1,2,3,4,5,6,7,8,9,10]  # IPG, these are kept
    extra_idx = [19]

    new_design_min = design_min[indices]
    new_design_min_extras = design_min[extra_idx]
    new_design_max = design_max[indices]
    new_design_max_extras = design_max[extra_idx]

    # now append the max and min values of eta/s and zeta/s at discrete points
    num_T = 10
    eta_min = []; eta_max = []; zeta_min = []; zeta_max = []
    for point in range(num_T):
        eta_min.append(0.01)
        eta_max.append(0.65)
        zeta_min.append(0.01)
        zeta_max.append(0.2)
    eta_min = np.array(eta_min); eta_max = np.array(eta_max)
    zeta_min = np.array(zeta_min); zeta_max = np.array(zeta_max)

    new_design_min = np.concatenate((new_design_min, eta_min))
    new_design_min = np.concatenate((new_design_min, zeta_min))
    new_design_min = np.concatenate((new_design_min, new_design_min_extras))

    new_design_max = np.concatenate((new_design_max, eta_max))
    new_design_max = np.concatenate((new_design_max, zeta_max))
    new_design_max = np.concatenate((new_design_max, new_design_max_extras))


    return new_design_max, new_design_min

# Function that runs viscosity discretization
def prepare_emu_design(system_str):
    design, design_max, design_min, labels = load_design(
        system_str=system_str, pset="main"
    )

    # transformation of design for viscosities
    if do_transform_design:
        logging.info("Note: Transforming design of viscosities")
        # replace this with function that transforms based on labels, not indices
        design = transform_design(design.values)
        design_max, design_min = transform_design_min_max(design_max,design_min)
    else:
        design = design.values

    # use this to set the actual parameter ranges in the training sample for emulator lengthscale and bayesian prior
    #design_max = np.max(design, axis=0)
    #design_min = np.min(design, axis=0)
    return design, design_max, design_min, labels

# Save MAP parameters for certain calibrations for future use
MAP_params = {}
MAP_params["Pb-Pb-2760"] = {}
MAP_params["Au-Au-200"] = {}

# Legacy IP-Glasma values from ptemcee sampler with 110 walkers, 500 step adaptive burn in, 20k steps, 10 temperatures
MAP_params["Pb-Pb-2760"]["Grad"] = [
    0.70816,  # mu_Qs
    0.5144,  # tau_0
    0.22333,  # T_eta,kink
    -0.19492,  # a_low
    -0.78461,  # a_high
    0.13824,  # eta_kink
    0.22731,  # zeta_max
    0.29265,  # T_(zeta,peak)
    0.03545,  # w_zeta
    -0.57749,  # lambda_zeta
    0.1542,  # T_s
]

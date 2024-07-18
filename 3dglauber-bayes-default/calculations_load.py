#!/usr/bin/env python3
import numpy as np
from configurations import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#############################################################################
#############################################################################

## Script to load model calculations from bayes_dtype saved format, perform
## observable transformations (like take the log) and check for NaNs

#############################################################################
#############################################################################

trimmed_model_data = {}
validation_data = {}
MAP_data = {}

if run_closure:
    print("\n Using model predictions for validation point #" + str(closure_val_pt) + " as pseudo-data in closure test!")
    print("Maintaining real experimental uncertainties as pseudo-data uncertainties.")

# loop over systems
for i, s in enumerate(system_strs):

    ######################### MAIN #########################
    Ndesign = SystemsInfo[s]["n_design"]
    # no deletions in current analysis
    Ndelete = len(SystemsInfo[s]["design_remove_idx"])

    print(
        "\nLoading {:s} main calculations from ".format(s)
        + SystemsInfo[s]["main_obs_file"]
    )
    ds = np.fromfile(SystemsInfo[s]["main_obs_file"], dtype=bayes_dtype)
    print("model_data.shape = " + str(ds.shape) + "\n")
    #print("--------------------\n")

    for point in range(Ndesign):
        for obs in active_obs_list[s]:
            values = np.array(ds[s][point][obs]['mean'])

            # chack for NaNs in training model calculations
            isnan = np.isnan(values)
            if (np.sum(isnan) > 0) and (not point in delete_design_pts_set):
                print("WARNING : FOUND NAN IN MODEL DATA : (design pt , obs)"\
                      +" = ( {:s} , {:s} )".format( str(point), obs) )

                # set the value of NaNs to the mean of the rest of the values
                ds[s][point][obs]['mean'][isnan] = np.mean(values[np.logical_not(isnan)])

            # Transform yield related observables
            is_mult = ('dN' in obs) or ('dET' in obs)
            if is_mult and transform_multiplicities:
                ds[s][point][obs]['mean'] = np.log(values) #1.0 +

    if Ndelete > 0:
        print(
            "Design points which will be deleted from training : "
            + str(SystemsInfo[s]["design_remove_idx"])
        )
        trimmed_model_data[s] = np.delete(ds[s], SystemsInfo[s]["design_remove_idx"], 0)
    else:
        #print("No design points will be deleted from training")
        trimmed_model_data[s] = ds[s]




    ######################### VALIDATION #########################
    Nvalidation = SystemsInfo[s]["n_validation"]
    # load the validation model calculations
    #if validation:
    if pseudovalidation:
        validation_data[s] = trimmed_model_data[s]
    #elif crossvalidation:
        #validation_data[s] = model_data[cross_validation_pts]  ### unclear what "model_data" refers to, not defined
    else:
        print(
            "\nLoading {:s} validation calculations from ".format(s)
            + SystemsInfo[s]["validation_obs_file"]
        )
        dsv = np.fromfile(SystemsInfo[s]["validation_obs_file"], dtype=bayes_dtype)
        print("validation_data.shape = " + str(dsv.shape) + "\n")

        for point in range(Nvalidation):
            for obs in active_obs_list[s]:
                values = np.array(dsv[s][point][obs]['mean'])
                #if point == 0 and obs == 'v22STAR':
                    #print(np.array(dsv[s][point][obs]['mean']))
                    #print(np.array(ds[s][point][obs]['mean']))

                # Transform yield related observables
                is_mult = ('dN' in obs) or ('dET' in obs)
                if is_mult and transform_multiplicities:
                    # check for nans and negative multiplicities and set them to 0
                    for i, entry in enumerate(values):
                        if np.isnan(entry) or entry<0:
                            values[i] = 0.1
                    # take the log
                    values = np.log(values) #1.0 +
                    dsv[s][point][obs]['mean'] = values

        validation_data[s] = dsv[s]
        # delete design points from the validation set
        # validation_data[s] = np.delete(dsv[s], delete_design_pts_validation_set, 0)




    ######################### MAP #########################
    # load the MAP calculations
    print(
        "Loading {:s} MAP calculations from ".format(s) + SystemsInfo[s]["MAP_obs_file"]
    )
    try:
        dsMAP = np.fromfile(SystemsInfo[s]["MAP_obs_file"], dtype=bayes_dtype)
        MAP_data[s] = dsMAP[s]
        print("MAP_data.shape = " + str(dsMAP.shape))
    except:
        print("No MAP calculations found for system " + s)

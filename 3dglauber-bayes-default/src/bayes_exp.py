#!/usr/bin/env python3
import numpy as np
from configurations import *
from bins_and_cuts import *
from calculations_load import validation_data
from Read_calculations_combined_array import obs_indices_dict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#############################################################################
#############################################################################

## Script to read in experimental data into bayes_dtype format for mcmc

#############################################################################
#############################################################################

# create the bayes_dtype array we want to store experimental data in
entry = np.zeros(1, dtype=np.dtype(bayes_dtype))

# system loop
for s in system_strs:

    #dsv = np.fromfile(SystemsInfo[s]["validation_obs_file"], dtype=bayes_dtype)
    #print("validation_data.shape = " + str(dsv.shape) + "\n")
    #dsv = validation_data[s]

    # data or pseudodata, set in configurations.py
    path = dir_obs_exp

    #### implement flag for closure test here to change the path above ####

    print("\nLoading experimental data from " + path + " for " + str(s) + '\n')
    #for exp in list(expt_and_obs_for_system[s].keys()):
    #for exp in list(expt_for_system_dict.keys()):
        #for obs in list(expt_and_obs_for_system[s][exp].keys()):
    for obs in list(obs_cent_list[s].keys()):
        # read the experiment from the observable label string
        exp = expt_label_dict[obs[-4:]]

        # the number of bins used for calibration for each observable (data files may contain more bins but cannot contain fewer)
        n_bins_bayes = obs_indices_dict[obs][1]-obs_indices_dict[obs][0]
        #n_bins_bayes = len(obs_cent_list[s][obs])

        # loop through the models (in our case one)
        for idf in range(number_of_models_per_run):
            # read data
            exp_data = pd.read_csv(
                path + '/' + s + "/" + exp + "/" + obs + ".dat", sep=" ", skiprows=2, escapechar="#"
            )
            # save data in bayes_dtype arrays

            # closure test flag
            if run_closure==True:
                print(" - Using model predictions for " + str(obs) + " for validation point #" + str(closure_val_pt) + " as pseudo-data in closure test!")
                #pseudo = validation_data[s][closure_val_pt][obs]['mean']
                pseudo = np.array(validation_data[s][closure_val_pt][obs]['mean'])
                entry[idf][s][obs]["mean"] = pseudo
                #print((pseudo))
            # read real experimental data as default
            else:
                print(" - Using experimental data of " + str(obs) ) #+ " in " + str(s)) # + " !")
                data = exp_data["val"].iloc[:n_bins_bayes]
                entry[idf][s][obs]["mean"] = data
                #print((np.array(data)))

            # read real experimental uncertainties, whether real or pseudo-data analysis
            entry[s][obs]["err"] = np.sqrt(exp_data["err"].iloc[:n_bins_bayes]) #np.sqrt

            # zero experimental error flag
            if set_exp_error_to_zero:
                entry[s][obs]["err"][:, idf] = entry[s][obs]["err"][:, idf] * 0.0

# pick out the (df) model we want (only use a single one in this analysis)
Y_exp_data = entry[0]

# check the saved experimental data bayes_dtype array
#print(Y_exp_data['Au-Au-200']['v32_int_STAR']['mean'])

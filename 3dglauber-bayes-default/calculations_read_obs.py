#!/usr/bin/env python3
import numpy as np
from configurations import *
from bins_and_cuts import *
#from Read_calculations_with_flow import obs_indices_dict
from Read_calculations_combined_array import obs_indices_dict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#############################################################################
#############################################################################

## Script to read in model calculations from text files into bayes_dtype
## format for emulator training and validation

#############################################################################
#############################################################################


def load_and_compute(inputfile, system, design, specify_idf=0):

    # create the bayes_dtype array we want to store calculations in
    entry = np.zeros(1, dtype=np.dtype(bayes_dtype))

    # read text file with data in csv format
    simulation = pd.read_csv(inputfile)

    for obs in list(obs_cent_list[s].keys()):
        Y = simulation.values[design][obs_indices_dict[obs][0]:obs_indices_dict[obs][1]]
        entry[system][obs]["mean"][specify_idf] = np.array(Y)
    return entry

if __name__ == "__main__":
    for s in system_strs:
        # this folder chain needs to be created beforehand using this naming convention (maybe change directory names?)
        # naming defined in configurations.py
        run_dir = SystemsInfo[s]["run_dir"]
        f_events_folder = './model_calculations/' + run_dir + '/Obs/';
        print("\n" + s + "\n")
        print("Averaging/reading events into " + f_events_folder)
        for dataset in ['main', 'validation']:
            # name of the output formatted observable files that will be fed to the emulator
            f_obs_file = f_events_folder + dataset + '.dat';
            # number of design points expected in the calculations sets, defined in configurations.py
            if dataset == 'main':
                n_design_pts =  SystemsInfo[s]["n_design"];
            elif dataset == 'validation':
                n_design_pts = SystemsInfo[s]["n_validation"];
            print("\n")
            print("Re-formatting observables for " + dataset + " design points")
            print("##########################")
            results = []
            # loop through design points
            for design in range(n_design_pts):
                print("Reading design pt : " + str(design) + "\n")
                # the below files are calculated separately and provided in this directory
                if dataset == 'main':
                    filename = f_events_folder + '/Simulation'   # text file with the training calculations
                elif dataset == 'validation':
                    filename = f_events_folder + '/Validation'    # text file with the validation calculations
                calc_point = load_and_compute(filename, s, design)[0]
                results.append(calc_point)
            results = np.array(results)
            #print("results.shape = " + str(results.shape))
            results.tofile(f_obs_file)

#!/usr/bin/env python3
import numpy as np
from configurations import *
from bins_and_cuts import *
from Read_calculations_with_flow import obs_indices_dict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#############################################################################
#############################################################################

## Script to read in model calculations from text files into bayes_dtype
## format for emulator training and validation

#############################################################################
#############################################################################


# dictionary with the the bin indices labeling particular observables saved in the model calculations text files
obs_index_list = {

    "Au-Au-200": {

        # "dNdeta_eta_cen_00_05_BRAH" : obs_indices_dict["dNdeta_eta_cen_00_05_BRAH"],
        # "dNdeta_eta_cen_05_10_BRAH" : obs_indices_dict["dNdeta_eta_cen_05_10_BRAH"],
        # "dNdeta_eta_cen_10_20_BRAH" : obs_indices_dict["dNdeta_eta_cen_10_20_BRAH"],
        # "dNdeta_eta_cen_20_30_BRAH" : obs_indices_dict["dNdeta_eta_cen_20_30_BRAH"],
        # "dNdeta_eta_cen_30_40_BRAH" : obs_indices_dict["dNdeta_eta_cen_30_40_BRAH"],
        # "dNdeta_eta_cen_40_50_BRAH" : obs_indices_dict["dNdeta_eta_cen_40_50_BRAH"],

        # "dNdeta_eta_cen_00_05_BRAH_2" : obs_indices_dict["dNdeta_eta_cen_00_05_BRAH_2"],
        # "dNdeta_eta_cen_05_10_BRAH_2" : obs_indices_dict["dNdeta_eta_cen_05_10_BRAH_2"],
        # "dNdeta_eta_cen_10_20_BRAH_2" : obs_indices_dict["dNdeta_eta_cen_10_20_BRAH_2"],
        # "dNdeta_eta_cen_20_30_BRAH_2" : obs_indices_dict["dNdeta_eta_cen_20_30_BRAH_2"],
        # "dNdeta_eta_cen_30_40_BRAH_2" : obs_indices_dict["dNdeta_eta_cen_30_40_BRAH_2"],
        # "dNdeta_eta_cen_40_50_BRAH_2" : obs_indices_dict["dNdeta_eta_cen_40_50_BRAH_2"],
        #
        # "dNdeta_eta_cen_00_05_BRAH_3" : obs_indices_dict["dNdeta_eta_cen_00_05_BRAH_3"],
        # "dNdeta_eta_cen_05_10_BRAH_3" : obs_indices_dict["dNdeta_eta_cen_05_10_BRAH_3"],
        # "dNdeta_eta_cen_10_20_BRAH_3" : obs_indices_dict["dNdeta_eta_cen_10_20_BRAH_3"],
        # "dNdeta_eta_cen_20_30_BRAH_3" : obs_indices_dict["dNdeta_eta_cen_20_30_BRAH_3"],
        # "dNdeta_eta_cen_30_40_BRAH_3" : obs_indices_dict["dNdeta_eta_cen_30_40_BRAH_3"],
        # "dNdeta_eta_cen_40_50_BRAH_3" : obs_indices_dict["dNdeta_eta_cen_40_50_BRAH_3"],

        # "dNdeta_eta_cen_00_05_frwd_BRAH" : obs_indices_dict["dNdeta_eta_cen_00_05_frwd_BRAH"],
        # "dNdeta_eta_cen_05_10_frwd_BRAH" : obs_indices_dict["dNdeta_eta_cen_05_10_frwd_BRAH"],
        # "dNdeta_eta_cen_10_20_frwd_BRAH" : obs_indices_dict["dNdeta_eta_cen_10_20_frwd_BRAH"],
        # "dNdeta_eta_cen_20_30_frwd_BRAH" : obs_indices_dict["dNdeta_eta_cen_20_30_frwd_BRAH"],
        # "dNdeta_eta_cen_30_40_frwd_BRAH" : obs_indices_dict["dNdeta_eta_cen_30_40_frwd_BRAH"],
        # "dNdeta_eta_cen_40_50_frwd_BRAH" : obs_indices_dict["dNdeta_eta_cen_40_50_frwd_BRAH"],

        #"dNdeta_eta_cen_0_3_PHOB" : obs_indices_dict["dNdeta_eta_cen_0_3_PHOB"],
        "dNdeta_eta_cen_00_03_PHOB" : obs_indices_dict["dNdeta_eta_cen_00_03_PHOB"],
        "dNdeta_eta_cen_03_06_PHOB" : obs_indices_dict["dNdeta_eta_cen_03_06_PHOB"],
        "dNdeta_eta_cen_06_10_PHOB" : obs_indices_dict["dNdeta_eta_cen_06_10_PHOB"],
        "dNdeta_eta_cen_10_15_PHOB" : obs_indices_dict["dNdeta_eta_cen_10_15_PHOB"],
        "dNdeta_eta_cen_15_20_PHOB" : obs_indices_dict["dNdeta_eta_cen_15_20_PHOB"],
        "dNdeta_eta_cen_20_25_PHOB" : obs_indices_dict["dNdeta_eta_cen_20_25_PHOB"],
        "dNdeta_eta_cen_25_30_PHOB" : obs_indices_dict["dNdeta_eta_cen_25_30_PHOB"],
        "dNdeta_eta_cen_30_35_PHOB" : obs_indices_dict["dNdeta_eta_cen_30_35_PHOB"],
        "dNdeta_eta_cen_35_40_PHOB" : obs_indices_dict["dNdeta_eta_cen_35_40_PHOB"],
        "dNdeta_eta_cen_40_45_PHOB" : obs_indices_dict["dNdeta_eta_cen_40_45_PHOB"],
        "dNdeta_eta_cen_45_50_PHOB" : obs_indices_dict["dNdeta_eta_cen_45_50_PHOB"],

        "v22_eta_cen_20_70_STAR" : obs_indices_dict["v22_eta_cen_20_70_STAR"],
        "v22_eta_cen_03_15_PHOB" : obs_indices_dict["v22_eta_cen_03_15_PHOB"],
        "v22_eta_cen_15_25_PHOB" : obs_indices_dict["v22_eta_cen_15_25_PHOB"],
        "v22_eta_cen_25_50_PHOB" : obs_indices_dict["v22_eta_cen_25_50_PHOB"],

        "v22_pt_cen_00_10_PHEN" : obs_indices_dict["v22_pt_cen_00_10_PHEN"],
        "v22_pt_cen_10_20_PHEN" : obs_indices_dict["v22_pt_cen_10_20_PHEN"],
        "v22_pt_cen_20_30_PHEN" : obs_indices_dict["v22_pt_cen_20_30_PHEN"],
        "v22_pt_cen_30_40_PHEN" : obs_indices_dict["v22_pt_cen_30_40_PHEN"],
        "v22_pt_cen_40_50_PHEN" : obs_indices_dict["v22_pt_cen_40_50_PHEN"],
        "v22_pt_cen_50_60_PHEN" : obs_indices_dict["v22_pt_cen_50_60_PHEN"],

        # "v32_pt_cen_00_10_PHEN" : obs_indices_dict["v32_pt_cen_00_10_PHEN"],
        # "v32_pt_cen_10_20_PHEN" : obs_indices_dict["v32_pt_cen_10_20_PHEN"],
        # "v32_pt_cen_20_30_PHEN" : obs_indices_dict["v32_pt_cen_20_30_PHEN"],
        # "v32_pt_cen_30_40_PHEN" : obs_indices_dict["v32_pt_cen_30_40_PHEN"],
        # "v32_pt_cen_40_50_PHEN" : obs_indices_dict["v32_pt_cen_40_50_PHEN"],
        # "v32_pt_cen_50_60_PHEN" : obs_indices_dict["v32_pt_cen_50_60_PHEN"],
        #
        # "v42_pt_cen_00_10_PHEN" : obs_indices_dict["v42_pt_cen_00_10_PHEN"],
        # "v42_pt_cen_10_20_PHEN" : obs_indices_dict["v42_pt_cen_10_20_PHEN"],
        # "v42_pt_cen_20_30_PHEN" : obs_indices_dict["v42_pt_cen_20_30_PHEN"],
        # "v42_pt_cen_30_40_PHEN" : obs_indices_dict["v42_pt_cen_30_40_PHEN"],
        # "v42_pt_cen_40_50_PHEN" : obs_indices_dict["v42_pt_cen_40_50_PHEN"],
        #"v42_pt_cen_50_60_PHEN" : obs_indices_dict["v42_pt_cen_50_60_PHEN"],

         "v22_int_STAR" : obs_indices_dict["v22_int_STAR"],
         "v32_int_STAR" : obs_indices_dict[ "v32_int_STAR"],

        #"r2_eta_cen_10_40_STAR" : obs_indices_dict["r2_eta_cen_10_40_STAR"],
        #"r3_eta_cen_10_40_STAR" : obs_indices_dict["r3_eta_cen_10_40_STAR"],


        #"meanpT_pi_STAR" : obs_indices_dict["meanpT_pi_STAR"],
        #"meanpT_k_STAR" : obs_indices_dict["meanpT_k_STAR"],
        "meanpT_pi_PHEN" : obs_indices_dict["meanpT_pi_PHEN"],
        #"meanpT_k_PHEN"  : obs_indices_dict["meanpT_k_PHEN"],

    },

    "d-Au-200": {

        'dNdeta_eta_cen_00_20_PHOB' : obs_indices_dict['dNdeta_eta_cen_00_20_PHOB'],
        #'dNdeta_eta_cen_20_40_PHOB' : obs_indices_dict['dNdeta_eta_cen_20_40_PHOB'],
        #'dNdeta_eta_cen_40_60_PHOB' : obs_indices_dict['dNdeta_eta_cen_40_60_PHOB'],

        'dNdeta_eta_cen_00_05_PHEN'     : obs_indices_dict['dNdeta_eta_cen_00_05_PHEN'],
        'dNdeta_eta_cen_05_10_PHEN'     : obs_indices_dict['dNdeta_eta_cen_05_10_PHEN'],
        'dNdeta_eta_cen_10_20_PHEN'     : obs_indices_dict['dNdeta_eta_cen_10_20_PHEN'],

        'v22_eta_cen_00_05_PHEN'    : obs_indices_dict['v22_eta_cen_00_05_PHEN'],

        'v22_pt_cen_00_05_PHEN'     : obs_indices_dict['v22_pt_cen_00_05_PHEN'],
        # 'v32_pt_cen_00_05_PHEN'     : obs_indices_dict['v32_pt_cen_00_05_PHEN'],
        'v22_pt_cen_00_10_STAR'     : obs_indices_dict['v22_pt_cen_00_10_STAR' ],
        # 'v32_pt_cen_00_10_STAR'     : obs_indices_dict['v32_pt_cen_00_10_STAR'],

    },

}

def load_and_compute(inputfile, system, design, specify_idf=0):

    # create the bayes_dtype array we want to store calculations in
    entry = np.zeros(1, dtype=np.dtype(bayes_dtype))

    # read text file with data in csv format
    simulation = pd.read_csv(inputfile)

    for obs in list(obs_index_list[s].keys()):
        Y = simulation.values[design][obs_index_list[s][obs][0]:obs_index_list[s][obs][1]]
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
                    filename = f_events_folder + '/Simulation_with_flow'   # text file with the training calculations
                elif dataset == 'validation':
                    filename = f_events_folder + '/Flow_Validation_Simulation'    # text file with the validation calculations
                calc_point = load_and_compute(filename, s, design)[0]
                results.append(calc_point)
            results = np.array(results)
            #print("results.shape = " + str(results.shape))
            results.tofile(f_obs_file)

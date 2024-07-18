#!/usr/bin/env python3
#import logging
import yaml
from configurations import *
import numpy as np
import matplotlib.pyplot as plt

# experimental data directory with yaml files
dir_obs_exp = "./experimental_data_yaml"

# define the paths to read and write
yaml_paths_to_data = {

    ##################################################################
    ### First entry contains the full path from current directory, ###
    ### second entry contains index of observable in yaml format,  ###
    ### third entry contains the full path to the new directory,   ###
    ### fourth entry contains the scale factor to convert the data ###
    ##################################################################


        # Au-Au-200

        "dNch_deta_cen_00_05" : [dir_obs_exp + "/3-1D_Text_Files/AuAu200/BRAHMS/dNch-deta_low-eta_ins567754.yaml", 0, "./HIC_experimental_data/Au-Au-200/BRAHMS/", 1],
        "dNch_deta_cen_05_10" : [dir_obs_exp + "/3-1D_Text_Files/AuAu200/BRAHMS/dNch-deta_low-eta_ins567754.yaml", 1, "./HIC_experimental_data/Au-Au-200/BRAHMS/", 1],
        "dNch_deta_cen_10_20" : [dir_obs_exp + "/3-1D_Text_Files/AuAu200/BRAHMS/dNch-deta_low-eta_ins567754.yaml", 2, "./HIC_experimental_data/Au-Au-200/BRAHMS/", 1],
        "dNch_deta_cen_20_30" : [dir_obs_exp + "/3-1D_Text_Files/AuAu200/BRAHMS/dNch-deta_low-eta_ins567754.yaml", 3, "./HIC_experimental_data/Au-Au-200/BRAHMS/", 1],
        "dNch_deta_cen_30_40" : [dir_obs_exp + "/3-1D_Text_Files/AuAu200/BRAHMS/dNch-deta_low-eta_ins567754.yaml", 4, "./HIC_experimental_data/Au-Au-200/BRAHMS/", 1],
        "dNch_deta_cen_40_50" : [dir_obs_exp + "/3-1D_Text_Files/AuAu200/BRAHMS/dNch-deta_low-eta_ins567754.yaml", 5, "./HIC_experimental_data/Au-Au-200/BRAHMS/", 1],
        "v22STAR" : [dir_obs_exp + "/3-1D_Text_Files/AuAu200/STAR/Elliptic-flow_vs_eta_ins660793.yaml", 0, "./HIC_experimental_data/Au-Au-200/STAR/", .01],
        #"meanpT_pi" : not in yaml format, hand-assembled,
        #"meanpT_k" : not in yaml format, hand-assembled,
        #"meanpT_p" : not in yaml format, hand-assembled,
        #"meanpT_pi_minus" : not in yaml format, hand-assembled,
        #"meanpT_k_minus" : not in yaml format, hand-assembled,
        #"meanpT_p_bar" : not in yaml format, hand-assembled,


        # d-Au-200

        "dNch_deta_cen_00_20" : [dir_obs_exp + "/3-1D_Text_Files/dAu200/PHOBOS/dNch-deta_eprint10111940.yaml",1, "./HIC_experimental_data/d-Au-200/PHOBOS/", 1],
        "dNch_deta_cen_20_40" : [dir_obs_exp + "/3-1D_Text_Files/dAu200/PHOBOS/dNch-deta_eprint10111940.yaml",2, "./HIC_experimental_data/d-Au-200/PHOBOS/", 1],
        "dNch_deta_cen_40_60" : [dir_obs_exp + "/3-1D_Text_Files/dAu200/PHOBOS/dNch-deta_eprint10111940.yaml",3, "./HIC_experimental_data/d-Au-200/PHOBOS/", 1],
        #"v22PHENIX" : not in yaml format, hand-assembled,

}


# read the yaml files and write experimental data in "SIMS" format
for obs in yaml_paths_to_data.keys():
    path = yaml_paths_to_data[obs][0]
    new_path_and_filename = yaml_paths_to_data[obs][2] + obs + '_yaml_test'

    #try:

    # initialize new file
    newfile = open(new_path_and_filename, "w")
    newfile.write('#https\n#cuts\n#val err\n')

    # read and load yaml file
    with open(path, "r") as read:
        try:
            y = yaml.safe_load(read)

            # loop through "items" in yaml file
            for item, doc in y.items():

                # select for observables (leave out bin edges)
                if item == 'dependent_variables':

                    # the index specified in the dictionary
                    index = yaml_paths_to_data[obs][1]

                    # loop through the entries/bins
                    for i in range(len(doc[index]['values'])):

                        # flag for missing values and set them to 0
                        print(doc[index]['values'][i])
                        if doc[index]['values'][i]['value'] == '--':
                            doc[index]['values'][i]['value'] = 0; doc[index]['values'][i]['errors'][0]['symerror'] = 0

                        # set the scale factor (to convert, ex: percent values to decimal, etc)
                        scale = yaml_paths_to_data[obs][3]

                        # add the observable measurements to new file
                        newfile = open(new_path_and_filename, "a")
                        newfile.write(str(scale*doc[index]['values'][i]['value']) + ' ')

                        # square and add the observable errors (take only one side of asymmetric error) to new file
                        newfile = open(new_path_and_filename, "a")
                        try:
                            newfile.write(str((scale*doc[index]['values'][i]['errors'][0]['symerror'])**2) + '\n')
                        except:
                            newfile.write(str((scale*doc[index]['values'][i]['errors'][0]['asymerror']['plus'])**2) + '\n')

        except yaml.YAMLError as exc:
            print(exc)
    #except:
        #print('no file named ' +  path) #print(exc)

print('Reading data from yaml to SIMS format has finished')

#!/usr/bin/env python3
#import csv
import numpy as np
import pandas as pd
from bins_and_cuts import obs_cent_list

# dictionary with observable names as keys, and [filename, bin_low, bin_high]
obsfile_list = {
    # "dNdeta_eta_cen_00_05_BRAH"      : ['dNdeta___diff_eta__00_05_BRAHMS_AuAu________', None, None],
    # "dNdeta_eta_cen_05_10_BRAH"      : ['dNdeta___diff_eta__05_10_BRAHMS_AuAu________', None, None],
    # "dNdeta_eta_cen_10_20_BRAH"      : ['dNdeta___diff_eta__10_20_BRAHMS_AuAu________', None, None],
    # "dNdeta_eta_cen_20_30_BRAH"      : ['dNdeta___diff_eta__20_30_BRAHMS_AuAu________', None, None],
    # "dNdeta_eta_cen_30_40_BRAH"      : ['dNdeta___diff_eta__30_40_BRAHMS_AuAu________', None, None],
    # "dNdeta_eta_cen_40_50_BRAH"      : ['dNdeta___diff_eta__40_50_BRAHMS_AuAu________', None, None],
    #
    # "dNdeta_eta_cen_00_05_2_BRAH"    : ['dNdeta___diff_eta__00_05_BRAHMS_AuAu_______2', None, None],
    # "dNdeta_eta_cen_05_10_2_BRAH"    : ['dNdeta___diff_eta__05_10_BRAHMS_AuAu_______2', None, None],
    # "dNdeta_eta_cen_10_20_2_BRAH"    : ['dNdeta___diff_eta__10_20_BRAHMS_AuAu_______2', None, None],
    # "dNdeta_eta_cen_20_30_2_BRAH"    : ['dNdeta___diff_eta__20_30_BRAHMS_AuAu_______2', None, None],
    # "dNdeta_eta_cen_30_40_2_BRAH"    : ['dNdeta___diff_eta__30_40_BRAHMS_AuAu_______2', None, None],
    # "dNdeta_eta_cen_40_50_2_BRAH"    : ['dNdeta___diff_eta__40_50_BRAHMS_AuAu_______2', None, None],
    #
    # "dNdeta_eta_cen_00_05_3_BRAH"    : ['dNdeta___diff_eta__00_05_BRAHMS_AuAu_______3', None, None],
    # "dNdeta_eta_cen_05_10_3_BRAH"    : ['dNdeta___diff_eta__05_10_BRAHMS_AuAu_______3', None, None],
    # "dNdeta_eta_cen_10_20_3_BRAH"    : ['dNdeta___diff_eta__10_20_BRAHMS_AuAu_______3', None, None],
    # "dNdeta_eta_cen_20_30_3_BRAH"    : ['dNdeta___diff_eta__20_30_BRAHMS_AuAu_______3', None, None],
    # "dNdeta_eta_cen_30_40_3_BRAH"    : ['dNdeta___diff_eta__30_40_BRAHMS_AuAu_______3', None, None],
    # "dNdeta_eta_cen_40_50_3_BRAH"    : ['dNdeta___diff_eta__40_50_BRAHMS_AuAu_______3', None, None],
    #
    # "dNdeta_eta_cen_00_05_frwd_BRAH" : ['dNdeta___diff_eta__00_05_BRAHMS_AuAu_forward', None, None],
    # "dNdeta_eta_cen_05_10_frwd_BRAH" : ['dNdeta___diff_eta__05_10_BRAHMS_AuAu_forward', None, None],
    # "dNdeta_eta_cen_10_20_frwd_BRAH" : ['dNdeta___diff_eta__10_20_BRAHMS_AuAu_forward', None, None],
    # "dNdeta_eta_cen_20_30_frwd_BRAH" : ['dNdeta___diff_eta__20_30_BRAHMS_AuAu_forward', None, None],
    # "dNdeta_eta_cen_30_40_frwd_BRAH" : ['dNdeta___diff_eta__30_40_BRAHMS_AuAu_forward', None, None],
    # "dNdeta_eta_cen_40_50_frwd_BRAH" : ['dNdeta___diff_eta__40_50_BRAHMS_AuAu_forward', None, None],

    "dNdeta_eta_cen_00_03_PHOB"      : ['dNdeta___diff_eta__00_03_PHOBOS_AuAu________', None, None],
    "dNdeta_eta_cen_03_06_PHOB"      : ['dNdeta___diff_eta__03_06_PHOBOS_AuAu________', None, None],
    "dNdeta_eta_cen_06_10_PHOB"      : ['dNdeta___diff_eta__06_10_PHOBOS_AuAu________', None, None],
    "dNdeta_eta_cen_10_15_PHOB"      : ['dNdeta___diff_eta__10_15_PHOBOS_AuAu________', None, None],
    "dNdeta_eta_cen_15_20_PHOB"      : ['dNdeta___diff_eta__15_20_PHOBOS_AuAu________', None, None],
    "dNdeta_eta_cen_20_25_PHOB"      : ['dNdeta___diff_eta__20_25_PHOBOS_AuAu________', None, None],
    "dNdeta_eta_cen_25_30_PHOB"      : ['dNdeta___diff_eta__25_30_PHOBOS_AuAu________', None, None],
    "dNdeta_eta_cen_30_35_PHOB"      : ['dNdeta___diff_eta__30_35_PHOBOS_AuAu________', None, None],
    "dNdeta_eta_cen_35_40_PHOB"      : ['dNdeta___diff_eta__35_40_PHOBOS_AuAu________', None, None],
    "dNdeta_eta_cen_40_45_PHOB"      : ['dNdeta___diff_eta__40_45_PHOBOS_AuAu________', None, None],
    "dNdeta_eta_cen_45_50_PHOB"      : ['dNdeta___diff_eta__45_50_PHOBOS_AuAu________', None, None],

    'dNdeta_eta_cen_00_20_PHOB'      : ['dNdeta___diff_eta__00_20_PHOBOS_dAu_________', None, None],
    'dNdeta_eta_cen_00_05_PHEN'      : ['dNdeta___diff_eta__00_05_PHENIX_dAu_________', None, None],
    'dNdeta_eta_cen_05_10_PHEN'      : ['dNdeta___diff_eta__05_10_PHENIX_dAu_________', None, None],
    'dNdeta_eta_cen_10_20_PHEN'      : ['dNdeta___diff_eta__10_20_PHENIX_dAu_________', None, None],

    "v22_eta_cen_20_70_STAR"         : ['v22______diff_eta__20_70_STAR___AuAu________', None, None],
    "v22_eta_cen_03_15_PHOB"         : ['v22______diff_eta__03_15_PHOBOS_AuAu________', None, None],
    "v22_eta_cen_15_25_PHOB"         : ['v22______diff_eta__15_25_PHOBOS_AuAu________', None, None],
    "v22_eta_cen_25_50_PHOB"         : ['v22______diff_eta__25_50_PHOBOS_AuAu________', None, None],

    "v22_pt_cen_00_10_PHEN"          : ['v22______diff_pt___00_10_PHENIX_AuAu________', 0, 6],
    "v22_pt_cen_10_20_PHEN"          : ['v22______diff_pt___10_20_PHENIX_AuAu________', 0, 6],
    "v22_pt_cen_20_30_PHEN"          : ['v22______diff_pt___20_30_PHENIX_AuAu________', 0, 6],
    "v22_pt_cen_30_40_PHEN"          : ['v22______diff_pt___30_40_PHENIX_AuAu________', 0, 6],
    "v22_pt_cen_40_50_PHEN"          : ['v22______diff_pt___40_50_PHENIX_AuAu________', 0, 6],
    "v22_pt_cen_50_60_PHEN"          : ['v22______diff_pt___50_60_PHENIX_AuAu________', 0, 6],

    'v22_eta_cen_00_05_PHEN'         : ['v22______diff_eta__00_05_PHENIX_dAu_________', 10, 25],
    'v22_pt_cen_00_05_PHEN'          : ['v22______diff_pt___00_05_PHENIX_dAu_________', 0, 6],
    'v22_pt_cen_00_10_STAR'          : ['v22______diff_pt___00_10_STAR___dAu_________', 0, 6],
    "v22_int_STAR"                   : ['v22______int__cent_______STAR___AuAu________', 0, 7],
    "v32_int_STAR"                   : ['v32______int__cent_______STAR___AuAu________', 0, 7],
    "meanpT_pi_PHEN"                 : ['mnpt_pi__int__cent_______PHENIX_AuAu________', 0, 9],
}

# initializing dictionary to keep track of observable indices in the combined observable dataframe
obs_indices_dict = {}

    # Name two new files, the "simulation" and the "validation" files, and loop through them
for design_split in ['./model_calculations/production_414pts_Au_Au_200/Obs/Simulation','./model_calculations/production_414pts_Au_Au_200/Obs/Validation',
                     './model_calculations/production_414pts_d_Au_200/Obs/Simulation','./model_calculations/production_414pts_d_Au_200/Obs/Validation']:

        # Define the design point ranges for the simulation and validation sets,
        # The design_max for the upper bound will not throw an error
    if 'Simulation' in design_split:
        rows_skip_slice = range(414,420)
    if 'Validation' in design_split:
        rows_skip_slice = range(414)


    # Read files into pandas dataframes and combine them into a single file
    AuAu_obs = list(obs_cent_list['Au-Au-200'].keys())#.append(list(obs_cent_list['d-Au-200'].keys()))
    dAu_obs = list(obs_cent_list['d-Au-200'].keys())
    AuAu_obs.extend(dAu_obs)
    obs_files = AuAu_obs

    # initialize column index to keep track of the observable position indices in dataframe
    column_idx = 0

    for i, obs in enumerate(obs_files):

        # initialize index bin for observaable
        idx_bin = [column_idx,'dummy']

        file_name, idx1, idx2 = obsfile_list[obs]
        #idx2 = int(len(obs_cent_list['Au-Au-200'][obs]))

        # set the usecols argument for slices of the observable columns
        if idx1 == None:
            cols_slice = None
        else:
            cols_slice = range(idx1,idx2)

        # initialize the file by adding the first observable group to a dataframe
        if i == 0:
            combined_df = pd.read_csv('./Text_files_MAP/' + file_name, skiprows=rows_skip_slice, usecols=cols_slice, header=None) # header=None leaves placeholder integer column names
            # drop the last (dummy) column
            combined_df_cut = combined_df.drop(combined_df.columns[-1],axis=1)

            column_idx+=np.shape(combined_df_cut)[1]

        # add all the other observables by concatenating dataframes                                     # this should be updated
        else:
            current_df = pd.read_csv('./Text_files_MAP/' + file_name, skiprows=rows_skip_slice, usecols=cols_slice, header=None) # header=None leaves placeholder integer column names
            # drop the last (dummy) column
            current_df_cut = current_df.drop(current_df.columns[-1],axis=1)                                                                                           # this should be updated

            column_idx+=np.shape(current_df_cut)[1]

            # concatenate
            combined_df_cut = pd.concat([combined_df_cut,current_df_cut], axis=1)



        idx_bin[1] = column_idx
        obs_indices_dict[obs] = idx_bin

    combined_df_cut.to_csv(design_split,header=True,index=False) #design_split

print(obs_indices_dict.items())

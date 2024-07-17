import csv
import numpy as np
#import bins_and_cuts

#######################################################################################
# This script reads model calculations of dNch/deta for a set of model design
# points for both AuAu and dAu at 200 GeV for eta bins in different centrality
# classes and writes them into the emulator input format, splitting  into a training
# and a validation set.

# Andi M.
#######################################################################################


    # List of eta bins for BRAHMS and PHOBOS (and STAR and PHENIX) measurements in AuAu and dAu respectively (used to define individual observables)
# 20 bins
B_cen = [[-3.0,-2.7],[-2.7,-2.4],[-2.4,-2.1],[-2.1,-1.8],[-1.8,-1.5],[-1.5,-1.2],[-1.21,-0.91],[-0.91,-0.6],[-0.6,-0.3],[-0.3,0],[0,0.3],[0.3,0.6],[0.6,0.91],[0.91,1.21],[1.21,1.51],[1.51,1.81],[1.81,2.1],[2.1,2.4],[2.4,2.7],[2.7,3.0]]
# 54 bins
PHOBOS_mult_rap_bins = [ [-5.4, -5.2], [-5.2, -5.0], [-5.0, -4.8], [-4.8, -4.6], [-4.6, -4.4], [-4.4, -4.2], [-4.2, -4.0], [-4.0, -3.8], [-3.8, -3.6], [-3.6, -3.4], [-3.4, -3.2], [-3.2, -3.0], [-3.0, -2.8], [-2.8, -2.6], [-2.6, -2.4], [-2.4, -2.2], [-2.2, -2.0], [-2.0, -1.8], [-1.8, -1.6], [-1.6, -1.4], [-1.4, -1.2], [-1.2, -1.0], [-1.0, -0.8], [-0.8, -0.6], [-0.6, -0.4], [-0.4, -0.2], [-0.2, 0.0], [0.0, 0.2],
                          [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0], [1.0, 1.2], [1.2, 1.4], [1.4, 1.6], [1.6, 1.8], [1.8, 2.0], [2.0, 2.2], [2.2, 2.4], [2.4, 2.6], [2.6, 2.8], [2.8, 3.0], [3.0, 3.2], [3.2, 3.4], [3.4, 3.6], [3.6, 3.8], [3.8, 4.0], [4.0, 4.2], [4.2, 4.4], [4.4, 4.6], [4.6, 4.8], [4.8, 5.0], [5.0, 5.2], [5.2, 5.4] ]
# 16 bins
B_cen_frwd = [ [-4.9,-4.5],[-4.5,-4.1],[-4.2,-3.8],[-4.0,-3.6],[-3.7,-3.3],[-3.3,-2.9],[-3.1,-2.7],[-2.7,-2.3],[2.3,2.7],[2.7,3.1],[2.9,3.3],[3.3,3.7],[3.6,4.0],[3.8,4.2],[4.1,4.5],[4.5,4.9] ]
# 54 bins
P_cen = [[-5.4, -5.2], [-5.2, -5.0], [-5.0, -4.8], [-4.8, -4.6], [-4.6, -4.4], [-4.4, -4.2], [-4.2, -4.0], [-4.0, -3.8], [-3.8, -3.6], [-3.6, -3.4], [-3.4, -3.2], [-3.2, -3.0], [-3.0, -2.8], [-2.8, -2.6], [-2.6, -2.4], [-2.4, -2.2],
        [-2.2, -2.0], [-2.0, -1.8], [-1.8, -1.6], [-1.6, -1.4], [-1.4, -1.2], [-1.2, -1.0], [-1.0, -0.8], [-0.8, -0.6], [-0.6, -0.4], [-0.4, -0.2], [-0.2, 0.0], [0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0], [1.0, 1.2],
        [1.2, 1.4], [1.4,1.6], [1.6, 1.8], [1.8, 2.0], [2.0, 2.2], [2.2, 2.4], [2.4, 2.6], [2.6, 2.8], [2.8, 3.0], [3.0, 3.2], [3.2, 3.4], [3.4, 3.6], [3.6, 3.8], [3.8, 4.0], [4.0, 4.2], [4.2, 4.4], [4.4, 4.6], [4.6, 4.8], [4.8, 5.0], [5.0, 5.2], [5.2, 5.4]]
# 21 bins
S_cen = [[-3.9,-3.7],[-3.7,-3.5],[-3.5,-3.3],[-3.3,-3.1],[-3.1,-2.9],[-2.9,-2.7],[-0.9,-0.7],[-0.7,-0.5],[-0.5,-0.3],[-0.3,-0.1],[-0.1,0.1],[0.1,0.3],[0.3,0.5],[0.5,0.7],[0.7,0.9],[2.7,2.9],[2.9,3.1],[3.1,3.3],[3.3,3.5],[3.5,3.7],[3.7,3.9]]
# 24 bins
Ph_cen =  [[-3.0,-2.8],[-2.8,-2.6],[-2.6,-2.4],[-2.4,-2.2],[-2.2,-2.0],[-2.0,-1.8],[-1.8,-1.6],[-1.6,-1.4],[-1.4,-1.2],[-1.2,-1.0],[-0.4,-0.2],[-0.2,0],[0,0.2],[0.2,0.4],[1.1,1.3],[1.3,1.5],[1.5,1.7],[1.7,1.9],[1.9,2.1],[2.1,2.3],[2.3,2.5],[2.5,2.7],[2.7,2.9],[2.9,3.0]]
# 9 bins
STAR_meanpT_cen = [[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80]]
# 11 bins
PHENIX_meanpT_cen = [ [0,5],[5,10],[10,15],[15,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,92] ]
# 9 bins
STAR_identified_yield_integrated = STAR_meanpT_cen
# 16 bins
PHOBOS_v2_cen = [ [-5.06, -4.86],[-4.23, -4.03],[-3.77, -3.57],[-3.14, -2.94],[-1.79, -1.59],[-1.32, -1.12],[-0.86, -0.66],
                                [-0.41, -0.21],[.21,.41],[.66,.86],[1.12,1.32],[1.59,1.79],[2.94,3.14],[3.57,3.77],[4.03,4.23],[4.86,5.06] ]
# 7 bins
PHENIX_vn_pt = [ [0.25,0.5],[0.5,0.75],[0.75,1.0],[1.0,1.25],[1.25,1.5],[1.5,1.75],[1.75,2.0] ]
# 8 bins
PHENIX_dAu_vn_pt = [ [0.4,0.6],[0.6,0.8],[0.8,1.0],[1.0,1.2],[1.2,1.4],[1.4,1.6],[1.6,1.8],[1.8,2.0] ]
# 7 bins
STAR_dAu_vn_pt = [ [0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.1],[1.1,1.4],[1.4,1.7],[1.7,2.0] ]
# bins
STAR_vn_int_cen = STAR_meanpT_cen

STAR_rn_eta_bins = [[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.0]]



obs_indices_dict = {}

obsfile_list = {
'dNdeta___diff_eta__00_05_BRAHMS_AuAu________': "dNdeta_eta_cen_00_05_BRAH",
'dNdeta___diff_eta__05_10_BRAHMS_AuAu________': "dNdeta_eta_cen_05_10_BRAH",
'dNdeta___diff_eta__10_20_BRAHMS_AuAu________': "dNdeta_eta_cen_10_20_BRAH",
'dNdeta___diff_eta__20_30_BRAHMS_AuAu________': "dNdeta_eta_cen_20_30_BRAH",
'dNdeta___diff_eta__30_40_BRAHMS_AuAu________': "dNdeta_eta_cen_30_40_BRAH",
'dNdeta___diff_eta__40_50_BRAHMS_AuAu________': "dNdeta_eta_cen_40_50_BRAH",

# 'dNdeta___diff_eta__00_05_BRAHMS_AuAu_______2': "dNdeta_eta_cen_00_05_BRAH_2",
# 'dNdeta___diff_eta__05_10_BRAHMS_AuAu_______2': "dNdeta_eta_cen_05_10_BRAH_2",
# 'dNdeta___diff_eta__10_20_BRAHMS_AuAu_______2': "dNdeta_eta_cen_10_20_BRAH_2",
# 'dNdeta___diff_eta__20_30_BRAHMS_AuAu_______2': "dNdeta_eta_cen_20_30_BRAH_2",
# 'dNdeta___diff_eta__30_40_BRAHMS_AuAu_______2': "dNdeta_eta_cen_30_40_BRAH_2",
# 'dNdeta___diff_eta__40_50_BRAHMS_AuAu_______2': "dNdeta_eta_cen_40_50_BRAH_2",
#
# 'dNdeta___diff_eta__00_05_BRAHMS_AuAu_______3': "dNdeta_eta_cen_00_05_BRAH_3",
# 'dNdeta___diff_eta__05_10_BRAHMS_AuAu_______3': "dNdeta_eta_cen_05_10_BRAH_3",
# 'dNdeta___diff_eta__10_20_BRAHMS_AuAu_______3': "dNdeta_eta_cen_10_20_BRAH_3",
# 'dNdeta___diff_eta__20_30_BRAHMS_AuAu_______3': "dNdeta_eta_cen_20_30_BRAH_3",
# 'dNdeta___diff_eta__30_40_BRAHMS_AuAu_______3': "dNdeta_eta_cen_30_40_BRAH_3",
# 'dNdeta___diff_eta__40_50_BRAHMS_AuAu_______3': "dNdeta_eta_cen_40_50_BRAH_3",

'dNdeta___diff_eta__00_05_BRAHMS_AuAu_forward': "dNdeta_eta_cen_00_05_frwd_BRAH",
'dNdeta___diff_eta__05_10_BRAHMS_AuAu_forward': "dNdeta_eta_cen_05_10_frwd_BRAH",
'dNdeta___diff_eta__10_20_BRAHMS_AuAu_forward': "dNdeta_eta_cen_10_20_frwd_BRAH",
'dNdeta___diff_eta__20_30_BRAHMS_AuAu_forward': "dNdeta_eta_cen_20_30_frwd_BRAH",
'dNdeta___diff_eta__30_40_BRAHMS_AuAu_forward': "dNdeta_eta_cen_30_40_frwd_BRAH",
'dNdeta___diff_eta__40_50_BRAHMS_AuAu_forward': "dNdeta_eta_cen_40_50_frwd_BRAH",

'dNdeta___diff_eta__00_03_PHOBOS_AuAu________' : "dNdeta_eta_cen_00_03_PHOB",
'dNdeta___diff_eta__03_06_PHOBOS_AuAu________' : "dNdeta_eta_cen_03_06_PHOB",
'dNdeta___diff_eta__06_10_PHOBOS_AuAu________' : "dNdeta_eta_cen_06_10_PHOB",
'dNdeta___diff_eta__10_15_PHOBOS_AuAu________' : "dNdeta_eta_cen_10_15_PHOB",
'dNdeta___diff_eta__15_20_PHOBOS_AuAu________' : "dNdeta_eta_cen_15_20_PHOB",
'dNdeta___diff_eta__20_25_PHOBOS_AuAu________' : "dNdeta_eta_cen_20_25_PHOB",
'dNdeta___diff_eta__25_30_PHOBOS_AuAu________' : "dNdeta_eta_cen_25_30_PHOB",
'dNdeta___diff_eta__30_35_PHOBOS_AuAu________' : "dNdeta_eta_cen_30_35_PHOB",
'dNdeta___diff_eta__35_40_PHOBOS_AuAu________' : "dNdeta_eta_cen_35_40_PHOB",
'dNdeta___diff_eta__40_45_PHOBOS_AuAu________' : "dNdeta_eta_cen_40_45_PHOB",
'dNdeta___diff_eta__45_50_PHOBOS_AuAu________' : "dNdeta_eta_cen_45_50_PHOB",

'dNdeta___diff_eta__00_20_PHOBOS_dAu_________': 'dNdeta_eta_cen_00_20_PHOB',
#'dNdeta___diff_eta__20_40_PHOBOS_dAu_________': 'dNdeta_eta_cen_20_40_PHOB',
#'dNdeta___diff_eta__40_60_PHOBOS_dAu_________': 'dNdeta_eta_cen_40_60_PHOB',
#
'v22______diff_eta__20_70_STAR___AuAu________': "v22_eta_cen_20_70_STAR",
'v22______diff_eta__03_15_PHOBOS_AuAu________': "v22_eta_cen_03_15_PHOB",
'v22______diff_eta__15_25_PHOBOS_AuAu________': "v22_eta_cen_15_25_PHOB",
'v22______diff_eta__25_50_PHOBOS_AuAu________': "v22_eta_cen_25_50_PHOB",

'v22______diff_pt___00_10_PHENIX_AuAu________': "v22_pt_cen_00_10_PHEN",
'v22______diff_pt___10_20_PHENIX_AuAu________': "v22_pt_cen_10_20_PHEN",
'v22______diff_pt___20_30_PHENIX_AuAu________': "v22_pt_cen_20_30_PHEN",
'v22______diff_pt___30_40_PHENIX_AuAu________': "v22_pt_cen_30_40_PHEN",
'v22______diff_pt___40_50_PHENIX_AuAu________': "v22_pt_cen_40_50_PHEN",
'v22______diff_pt___50_60_PHENIX_AuAu________': "v22_pt_cen_50_60_PHEN",

'v32______diff_pt___00_10_PHENIX_AuAu________': "v32_pt_cen_00_10_PHEN",
'v32______diff_pt___10_20_PHENIX_AuAu________': "v32_pt_cen_10_20_PHEN",
'v32______diff_pt___20_30_PHENIX_AuAu________': "v32_pt_cen_20_30_PHEN",
'v32______diff_pt___30_40_PHENIX_AuAu________': "v32_pt_cen_30_40_PHEN",
'v32______diff_pt___40_50_PHENIX_AuAu________': "v32_pt_cen_40_50_PHEN",
'v32______diff_pt___50_60_PHENIX_AuAu________': "v32_pt_cen_50_60_PHEN",

'v42______diff_pt___00_10_PHENIX_AuAu________': "v42_pt_cen_00_10_PHEN",
'v42______diff_pt___10_20_PHENIX_AuAu________': "v42_pt_cen_10_20_PHEN",
'v42______diff_pt___20_30_PHENIX_AuAu________': "v42_pt_cen_20_30_PHEN",
'v42______diff_pt___30_40_PHENIX_AuAu________': "v42_pt_cen_30_40_PHEN",
'v42______diff_pt___40_50_PHENIX_AuAu________': "v42_pt_cen_40_50_PHEN",
#'v42______diff_pt___50_60_PHENIX_AuAu________': "v42_pt_cen_50_60_PHEN",

'v22______diff_eta__00_05_PHENIX_dAu_________': 'v22_eta_cen_00_05_PHEN',   # [10:]: exclude far backward bins

'v22______diff_pt___00_05_PHENIX_dAu_________': 'v22_pt_cen_00_05_PHEN',
'v32______diff_pt___00_05_PHENIX_dAu_________': 'v32_pt_cen_00_05_PHEN',

'v22______diff_pt___00_10_STAR___dAu_________': 'v22_pt_cen_00_10_STAR',
'v32______diff_pt___00_10_STAR___dAu_________': 'v32_pt_cen_00_10_STAR',

'v22______int__cent_______STAR___AuAu________':  "v22_int_STAR",    # [:6]: exclude very peripheral bins
'v32______int__cent_______STAR___AuAu________':  "v32_int_STAR",    # [:6]: exclude very peripheral bins

#'r2_______diff_eta__10_40_STAR___AuAu________': 'r2_eta_cen_10_40_STAR',
#'r3_______diff_eta__10_40_STAR___AuAu________': 'r3_eta_cen_10_40_STAR',

#'mnpt_pi__int__cent_______STAR___AuAu________': 'meanpT_pi_STAR',    # [:7]: exclude very peripheral bins
#'mnpt_k___int__cent_______STAR___AuAu________': 'meanpT_k_STAR',    # [:7]: exclude very peripheral bins

'mnpt_pi__int__cent_______PHENIX_AuAu________': "meanpT_pi_PHEN",    # [:8]: exclude very peripheral bins
#'mnpt_k___int__cent_______PHENIX_AuAu________': 'meanpT_k_PHEN',    # [:8]: exclude very peripheral bins

}

    # Name two new files, the "simulation" and the "validation" files, and loop through them
for design_split in ['./Text_files_MAP/Simulation_with_flow','./Text_files_MAP/Flow_Validation_Simulation']:
        # Open the two new files, read the centrality and eta bins and write them as observable names in the two files
    open(design_split, mode='w')
        # Loop for the AuAu dNdeta observable names
    for cen in ['0_5','5_10','10_20','20_30','30_40','40_50']:
        for eta_range in B_cen:
            with open(design_split, mode='a') as Simulation:
                    # Define observable naming convention
                Simulation.write('dNdeta_AuAu200_' + cen + '_cen[' + str(eta_range[0]) + ' ' + str(eta_range[1]) + '],')

    # for cen in ['0_5','5_10','10_20','20_30','30_40','40_50']:
    #     for eta_range in B_cen:
    #         with open(design_split, mode='a') as Simulation:
    #                 # Define observable naming convention
    #             Simulation.write('dNdeta2_AuAu200_' + cen + '_cen[' + str(eta_range[0]) + ' ' + str(eta_range[1]) + '],')
    #
    # for cen in ['0_5','5_10','10_20','20_30','30_40','40_50']:
    #     for eta_range in B_cen:
    #         with open(design_split, mode='a') as Simulation:
    #                 # Define observable naming convention
    #             Simulation.write('dNdeta3_AuAu200_' + cen + '_cen[' + str(eta_range[0]) + ' ' + str(eta_range[1]) + '],')


        # Loop for the AuAu dNdeta observable names from PHOBOS
    for cen in ['0_3','3_6','6_10','10_15','15_20','20_25','25_30','30_35','35_40','40_45','45_50']:
        for eta_range in PHOBOS_mult_rap_bins:
            with open(design_split, mode='a') as Simulation:
                    # Define observable naming convention
                Simulation.write('dNdeta_AuAu200_PHOB_' + cen + '_cen[' + str(eta_range[0]) + ' ' + str(eta_range[1]) + '],')


        # Loop for the AuAu dNdeta observable names
    for cen in ['0_5','5_10','10_20','20_30','30_40','40_50']:
        for eta_range in B_cen_frwd:
            with open(design_split, mode='a') as Simulation:
                    # Define observable naming convention
                Simulation.write('dNdeta_frwd_AuAu200_' + cen + '_cen[' + str(eta_range[0]) + ' ' + str(eta_range[1]) + '],')

        # Loop for the dAu dNdeta observable names
    for cen in ['0_20']: #, '20_40', '40_60']:
        for eta_range in P_cen:
            with open(design_split, mode='a') as Simulation:
                    # Define observable naming convention
                Simulation.write('dNdeta_dAu200_' + cen + '_cen[' + str(eta_range[0]) + ' ' + str(eta_range[1]) + '],')

        # Loop for the STAR AuAu v2 observable names
    for eta_range in S_cen:
        with open(design_split, mode='a') as Simulation:
                # Define observable naming convention
            Simulation.write('v2_AuAu200_20_70_cen[' + str(eta_range[0]) + ' ' + str(eta_range[1]) + '],')
        # Loop for the PHOBOS AuAu v2 observable names
    for cen in ['2_15','15_25','25_50']:
        for eta_range in PHOBOS_v2_cen:
            with open(design_split, mode='a') as Simulation:
                    # Define observable naming convention
                Simulation.write('v2_AuAu200' + cen + '_cen[' + str(eta_range[0]) + ' ' + str(eta_range[1]) + '],')
        # Loop for the PHENIX AuAu v2(pt) observable names
    for cen in ['0_10','10_20','20_30','30_40','40_50','50_60']:
        for pt_range in PHENIX_vn_pt:
            with open(design_split, mode='a') as Simulation:
                    # Define observable naming convention
                Simulation.write('v2_PHEN_AuAu200' + cen + '_cen[' + str(pt_range[0]) + ' ' + str(pt_range[1]) + '],')
        # Loop for the PHENIX AuAu v3(pt) observable names
    for cen in ['0_10','10_20','20_30','30_40','40_50','50_60']:
        for pt_range in PHENIX_vn_pt:
            with open(design_split, mode='a') as Simulation:
                    # Define observable naming convention
                Simulation.write('v3_PHEN_AuAu200' + cen + '_cen[' + str(pt_range[0]) + ' ' + str(pt_range[1]) + '],')
        # Loop for the PHENIX AuAu v4(pt) observable names
    for cen in ['0_10','10_20','20_30','30_40','40_50']: #,'50_60']:
        for pt_range in PHENIX_vn_pt:
            with open(design_split, mode='a') as Simulation:
                    # Define observable naming convention
                Simulation.write('v4_PHEN_AuAu200' + cen + '_cen[' + str(pt_range[0]) + ' ' + str(pt_range[1]) + '],')
        # Loop for the dAu v2 observable names
    for eta_range in Ph_cen[10:]: # exclude far backward bins
        with open(design_split, mode='a') as Simulation:
                # Define observable naming convention
            Simulation.write('v2_dAu200_0_5_cen[' + str(eta_range[0]) + ' ' + str(eta_range[1]) + '],')
        # Loop for the PHENIX dAu v2(pt) observable naming convention
    for pt_range in PHENIX_dAu_vn_pt:
        with open(design_split, mode='a') as Simulation:
                # Define observable naming convention
            Simulation.write('v2_PHEN_dAu200_0_5_cen[' + str(pt_range[0]) + ' ' + str(pt_range[1]) + '],')
        # Loop for the PHENIX dAu v3(pt) observable naming convention
    for pt_range in PHENIX_dAu_vn_pt:
        with open(design_split, mode='a') as Simulation:
                # Define observable naming convention
            Simulation.write('v3_PHEN_dAu200_0_5_cen[' + str(pt_range[0]) + ' ' + str(pt_range[1]) + '],')
        # Loop for the STAR dAu v2(pt) observable naming convention
    for pt_range in STAR_dAu_vn_pt:
        with open(design_split, mode='a') as Simulation:
                # Define observable naming convention
            Simulation.write('v2_STAR_dAu200_0_5_cen[' + str(pt_range[0]) + ' ' + str(pt_range[1]) + '],')
        # Loop for the STAR dAu v3(pt) observable naming convention
    for pt_range in STAR_dAu_vn_pt:
        with open(design_split, mode='a') as Simulation:
                # Define observable naming convention
            Simulation.write('v3_STAR_dAu200_0_5_cen[' + str(pt_range[0]) + ' ' + str(pt_range[1]) + '],')
        # Loop for the STAR AuAu integrated v2 observable naming convention
    for cent_range in STAR_vn_int_cen[:6]: # exclude very peripheral bins/those not used in JETSCAPE 2D Calib
        with open(design_split, mode='a') as Simulation:
                # Define observable naming convention
            Simulation.write('v2_int_STAR_AuAu200_[' + str(cent_range[0]) + ' ' + str(cent_range[1]) + '],')
        # Loop for the STAR AuAu integrated v3 observable naming convention
    for cent_range in STAR_vn_int_cen[:6]: # exclude very peripheral bins/those not used in JETSCAPE 2D Calib
        with open(design_split, mode='a') as Simulation:
                # Define observable naming convention
            Simulation.write('v3_int_STAR_AuAu200_[' + str(cent_range[0]) + ' ' + str(cent_range[1]) + '],')

    #     # Loop for the STAR AuAu r2(eta) observable names
    # for cen in ['10_40']:
    #     for eta_range in STAR_rn_eta_bins:
    #         with open(design_split, mode='a') as Simulation:
    #                 # Define observable naming convention
    #             Simulation.write('r2_STAR_AuAu200' + cen + '_cen[' + str(eta_range[0]) + ' ' + str(eta_range[1]) + '],')
    #     # Loop for the STAR AuAu r3(eta) observable names
    # for cen in ['10_40']:
    #     for eta_range in STAR_rn_eta_bins:
    #         with open(design_split, mode='a') as Simulation:
    #                 # Define observable naming convention
    #             Simulation.write('r3_STAR_AuAu200' + cen + '_cen[' + str(eta_range[0]) + ' ' + str(eta_range[1]) + '],')


    #     # Loop for the STAR AuAu mean pt observable naming convention
    # for species in ['pi','k']:
    #     for cent_range in STAR_meanpT_cen[:7]: # exclude very peripheral bins
    #         with open(design_split, mode='a') as Simulation:
    #                 # if/else-statement to add newline character after the final observable name
    #             #if (cent_range == [50,60] and species == 'k'):
    #                     # Define observable naming convention
    #                 #Simulation.write('meanpT_STAR_AuAu_' + species + '_[' + str(cent_range[0]) + ' ' + str(cent_range[1]) + ']\n')
    #             #else:
    #                     # Define observable naming convention
    #             Simulation.write('meanpT_STAR_AuAu_' + species + '_[' + str(cent_range[0]) + ' ' + str(cent_range[1]) + '],')

        # Loop for the PHENIX AuAu mean pt observable naming convention
    for species in ['pi']: #, 'k']:
        for cent_range in PHENIX_meanpT_cen[:8]: # exclude very peripheral bins
            with open(design_split, mode='a') as Simulation:
                    # if/else-statement to add newline character after the final observable name
                if (cent_range == [50,60] and species == 'pi'):
                        # Define observable naming convention
                    Simulation.write('meanpT_PHEN_AuAu_' + species + '_[' + str(cent_range[0]) + ' ' + str(cent_range[1]) + ']\n')
                else:
                        # Define observable naming convention
                    Simulation.write('meanpT_PHEN_AuAu_' + species + '_[' + str(cent_range[0]) + ' ' + str(cent_range[1]) + '],')


        # Define the design point ranges for the simulation and validation sets,
        # The design_max for the validation will not throw an error if greater than the total number of designs due to j==i flag below
    if design_split == './Text_files_MAP/Simulation_with_flow':
        design_min = 0; design_max = 375
    if design_split == './Text_files_MAP/Flow_Validation_Simulation':
        design_min = 375; design_max = 500

        # Loop through the training or the validation design point range
    for i in range(design_min,design_max):
        column_idx = 0
        for obsfile in obsfile_list.keys():
        #['dNchdeta_0_5','dNchdeta_5_10','dNchdeta_10_20','dNchdeta_20_30','dNchdeta_30_40','dNchdeta_40_50','dNchdeta_dAu200_0_20_rdc','dNchdeta_dAu200_20_40_rdc','dNchdeta_dAu200_40_60_rdc','v2_AuAu200_20_70','v2_dAu200_0_5','dNpideta_integrated_STAR','dNkdeta_integrated_STAR','dNpdeta_integrated_STAR','dNpiminusdeta_integrated_STAR','dNkminusdeta_integrated_STAR','dNpbardeta_integrated_STAR','meanpT_pi','meanpT_k','meanpT_p', 'meanpT_pi_minus','meanpT_k_minus','meanpT_p_bar']:
            obsfile_path = './Text_files_MAP/'+ obsfile
            with open(obsfile_path) as Calc_file:
                for j, line in enumerate(Calc_file):
                    if j == i:
                        line = line.strip('\n')
                        splitline = line.split(",")
                        del splitline[-1]
                        with open(design_split, mode='a') as Simulation:
                            if obsfile == 'mnpt_pi__int__cent_______PHENIX_AuAu________':
                                idx_bin = [column_idx,'dummy']
                                for k in range(len(splitline))[:8]:
                                    if k == 7: #(len(splitline)-1):
                                        Simulation.write(splitline[k] + '\n')
                                    else:
                                        Simulation.write(splitline[k] + ',')
                                    column_idx+=1
                                idx_bin[1] = column_idx
                                obs_indices_dict[obsfile_list[obsfile]] = idx_bin


                            else:
                                if obsfile == 'v22______diff_eta__00_05_PHENIX_dAu_________':
                                    idx_bin = [column_idx,'dummy']
                                    for k in range(len(splitline))[10:]:
                                        Simulation.write(splitline[k] + ',')
                                        column_idx+=1
                                    idx_bin[1] = column_idx
                                    obs_indices_dict[obsfile_list[obsfile]] = idx_bin
                                elif obsfile == 'v22______int__cent_______STAR___AuAu________':
                                    idx_bin = [column_idx,'dummy']
                                    for k in range(len(splitline))[:6]:
                                        Simulation.write(splitline[k] + ',')
                                        column_idx+=1
                                    idx_bin[1] = column_idx
                                    obs_indices_dict[obsfile_list[obsfile]] = idx_bin
                                elif obsfile == 'v32______int__cent_______STAR___AuAu________':
                                    idx_bin = [column_idx,'dummy']
                                    for k in range(len(splitline))[:6]:
                                        Simulation.write(splitline[k] + ',')
                                        column_idx+=1
                                    idx_bin[1] = column_idx
                                    obs_indices_dict[obsfile_list[obsfile]] = idx_bin
                                elif obsfile == 'mnpt_pi__int__cent_______PHENIX_AuAu________':
                                    idx_bin = [column_idx,'dummy']
                                    for k in range(len(splitline))[:8]:
                                        Simulation.write(splitline[k] + ',')
                                        column_idx+=1
                                    idx_bin[1] = column_idx
                                    obs_indices_dict[obsfile_list[obsfile]] = idx_bin
                                elif obsfile == 'mnpt_k___int__cent_______PHENIX_AuAu________':
                                    idx_bin = [column_idx,'dummy']
                                    for k in range(len(splitline))[:8]:
                                        Simulation.write(splitline[k] + ',')
                                        column_idx+=1
                                    idx_bin[1] = column_idx
                                    obs_indices_dict[obsfile_list[obsfile]] = idx_bin
                                elif obsfile == 'mnpt_pi__int__cent_______STAR___AuAu________':
                                    idx_bin = [column_idx,'dummy']
                                    for k in range(len(splitline))[:7]:
                                        Simulation.write(splitline[k] + ',')
                                        column_idx+=1
                                    idx_bin[1] = column_idx
                                    obs_indices_dict[obsfile_list[obsfile]] = idx_bin
                                elif obsfile == 'mnpt_k___int__cent_______STAR___AuAu________':
                                    idx_bin = [column_idx,'dummy']
                                    for k in range(len(splitline))[:7]:
                                        Simulation.write(splitline[k] + ',')
                                        column_idx+=1
                                    idx_bin[1] = column_idx
                                    obs_indices_dict[obsfile_list[obsfile]] = idx_bin
                                else:
                                    idx_bin = [column_idx,'dummy']
                                    for k in range(len(splitline)):
                                        Simulation.write(splitline[k] + ',')
                                        column_idx+=1
                                    idx_bin[1] = column_idx
                                    obs_indices_dict[obsfile_list[obsfile]] = idx_bin

    for string in obs_indices_dict.keys():
        print(string,obs_indices_dict[string])
    # Check that the right number of design point calculations have been read into the new files
    with open(design_split) as Calc_file:
        for i, line in enumerate(Calc_file):
            line = line.strip('\n')
            splitline = line.split(",")
            #print((splitline[60])) #corresponds to the first eta bin in the 20_30 cen file
    print("Finished reading " + str(i) + ' design point calculations into '  + design_split)

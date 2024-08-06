#!/usr/bin/env python3
import numpy as np

#############################################################################
#############################################################################

## Script to define observable bins for emulator training and the analysis

#############################################################################
#############################################################################

#n_bins = 120

# we can define these centrality bins for all STAR observables (sample from previous analysis)
STAR_cent_bins = np.array(
    [
        [0, 5],
        [5, 10],
        [10, 20],
        [20, 30],
        [30, 40],
        [40, 50],
        [50, 60],
        [60, 70],
        [70, 80],
    ]
)  # 9 bins

# 21 bins
STAR_v22_rap_bins = np.array([ [-3.9,-3.7],[-3.7,-3.5],[-3.5,-3.3],[-3.3,-3.1],[-3.1,-2.9],[-2.9,-2.7],[-0.9,-0.7],[-0.7,-0.5],[-0.5,-0.3],[-0.3,-0.1],[-0.1,0.1],[0.1,0.3],[0.3,0.5],[0.5,0.7],[0.7,0.9],[2.7,2.9],[2.9,3.1],[3.1,3.3],[3.3,3.5],[3.5,3.7],[3.7,3.9] ])

# 24 bins
PHENIX_v22_rap_bins = np.array([[-3.0,-2.8],[-2.8,-2.6],[-2.6,-2.4],[-2.4,-2.2],[-2.2,-2.0],[-2.0,-1.8],[-1.8,-1.6],[-1.6,-1.4],[-1.4,-1.2],[-1.2,-1.0],[-0.4,-0.2],[-0.2,0],
                        [0,0.2],[0.2,0.4],[1.1,1.3],[1.3,1.5],[1.5,1.7],[1.7,1.9],[1.9,2.1],[2.1,2.3],[2.3,2.5],[2.5,2.7],[2.7,2.9],[2.9,3.0]])

# 20 bins
BRAHMS_mult_rap_bins = np.array([[-3.0,-2.7],[-2.7,-2.4],[-2.4,-2.1],[-2.1,-1.8],[-1.8,-1.5],[-1.5,-1.2],[-1.21,-0.91],[-0.91,-0.6],[-0.6,-0.3],[-0.3,0],[0,0.3],[0.3,0.6],[0.6,0.91],[0.91,1.21],[1.21,1.51],[1.51,1.81],[1.81,2.1],[2.1,2.4],[2.4,2.7],[2.7,3.0]])
#BRAHMS_mult_rap_bins = np.array(BRAHMS_mult_rap_bins[0:n_bins])

# 16 bins
BRAHMS_mult_large_rap_bins = np.array([[-4.9,-4.5],[-4.5,-4.1],[-4.2,-3.8],[-4.0,-3.6],[-3.7,-3.3],[-3.3,-2.9],[-3.1,-2.7],[-2.7,-2.3],[2.3,2.7],[2.7,3.1],[2.9,3.3],[3.3,3.7],[3.6,4.0],[3.8,4.2],[4.1,4.5],[4.5,4.9]])

# 54 bins
PHOBOS_mult_rap_bins = np.array([ [-5.4, -5.2], [-5.2, -5.0], [-5.0, -4.8], [-4.8, -4.6], [-4.6, -4.4], [-4.4, -4.2], [-4.2, -4.0], [-4.0, -3.8], [-3.8, -3.6], [-3.6, -3.4], [-3.4, -3.2], [-3.2, -3.0], [-3.0, -2.8], [-2.8, -2.6], [-2.6, -2.4], [-2.4, -2.2], [-2.2, -2.0], [-2.0, -1.8], [-1.8, -1.6], [-1.6, -1.4], [-1.4, -1.2], [-1.2, -1.0], [-1.0, -0.8], [-0.8, -0.6], [-0.6, -0.4], [-0.4, -0.2], [-0.2, 0.0], [0.0, 0.2],
                          [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0], [1.0, 1.2], [1.2, 1.4], [1.4, 1.6], [1.6, 1.8], [1.8, 2.0], [2.0, 2.2], [2.2, 2.4], [2.4, 2.6], [2.6, 2.8], [2.8, 3.0], [3.0, 3.2], [3.2, 3.4], [3.4, 3.6], [3.6, 3.8], [3.8, 4.0], [4.0, 4.2], [4.2, 4.4], [4.4, 4.6], [4.6, 4.8], [4.8, 5.0], [5.0, 5.2], [5.2, 5.4] ])
# 16 bins
PHOBOS_v2_cen = np.array([ [-5.06, -4.86],[-4.23, -4.03],[-3.77, -3.57],[-3.14, -2.94],[-1.79, -1.59],[-1.32, -1.12],[-0.86, -0.66],
                                [-0.41, -0.21],[.21,.41],[.66,.86],[1.12,1.32],[1.59,1.79],[2.94,3.14],[3.57,3.77],[4.03,4.23],[4.86,5.06] ])

# 9 bins
STAR_meanpT_cen = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80]])

# 9 bins
STAR_identified_yield_integrated = STAR_meanpT_cen

# 8 bins
PHENIX_dAu_vn_pt = np.array([ [0.4,0.6],[0.6,0.8],[0.8,1.0],[1.0,1.2],[1.2,1.4],[1.4,1.6],[1.6,1.8],[1.8,2.0] ])

# 7 bins
STAR_dAu_vn_pt = np.array([ [0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.1],[1.1,1.4],[1.4,1.7],[1.7,2.0] ])

# 7 bins
PHENIX_vn_pt = np.array([ [0.25,0.5],[0.5,0.75],[0.75,1.0],[1.0,1.25],[1.25,1.5],[1.5,1.75],[1.75,2.0] ])

# 11 bins
PHENIX_meanpT_cen = np.array([ [0,5],[5,10],[10,15],[15,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,92] ])

# 5 bins
STAR_rn_eta_bins = np.array([[0,0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.0]])

# 34 bins
PHENIX_dAu_dNdeta_eta_bins = np.array([[-2.6, -2.5], [-2.5, -2.4], [-2.4, -2.3], [-2.3, -2.2], [-2.2, -2.1], [-2.1, -2.0], [-2.0, -1.9], [-1.9, -1.8], [-1.8, -1.7], [-1.7, -1.6], [-1.6, -1.5], [-1.5, -1.4], [-1.4, -1.3], [-0.4, -0.3], [-0.3, -0.2], [-0.2, -0.1], [-0.1, 0.0], [0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [1.3, 1.4], [1.4, 1.5], [1.5, 1.6], [1.6, 1.7], [1.7, 1.8], [1.8, 1.9], [1.9, 2.0], [2.0, 2.1], [2.1, 2.2], [2.2, 2.3], [2.3, 2.4], [2.4, 2.5], [2.5, 2.6]])


# the observables which will be used for parameter estimation
# inactive observables for an active system need to be commented out here
obs_cent_list = {
    "Pb-Pb-5020": { # left as a sample from previous analysis
        "dNch_deta": np.array(
            [
                [0, 2.5],
                [2.5, 5],
                [5, 7.5],
                [7.5, 10],
                [10, 20],
                [20, 30],
                [30, 40],
                [40, 50],
                [50, 60],
                [60, 70],
            ]
        ),
    },

    "d-Au-200": {

        'dNdeta_eta_cen_00_20_PHOB' : PHOBOS_mult_rap_bins,
        #'dNdeta_eta_cen_20_40_PHOB' : PHOBOS_mult_rap_bins,
        #'dNdeta_eta_cen_40_60_PHOB' : PHOBOS_mult_rap_bins,

        'dNdeta_eta_cen_00_05_PHEN'  : PHENIX_dAu_dNdeta_eta_bins,
        'dNdeta_eta_cen_05_10_PHEN'  : PHENIX_dAu_dNdeta_eta_bins,
        'dNdeta_eta_cen_10_20_PHEN'  : PHENIX_dAu_dNdeta_eta_bins,

        'v22_eta_cen_00_05_PHEN'    : PHENIX_v22_rap_bins[10:],

        'v22_pt_cen_00_05_PHEN'     : PHENIX_dAu_vn_pt[:5],
        # 'v32_pt_cen_00_05_PHEN'     : PHENIX_dAu_vn_pt,
        'v22_pt_cen_00_10_STAR'     : STAR_dAu_vn_pt[:5],
        # 'v32_pt_cen_00_10_STAR'     : STAR_dAu_vn_pt,

    },



    "Au-Au-200": {

        # "dNdeta_eta_cen_00_05_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_05_10_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_10_20_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_20_30_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_30_40_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_40_50_BRAH" : BRAHMS_mult_rap_bins,

        # "dNdeta_eta_cen_00_05_2_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_05_10_2_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_10_20_2_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_20_30_2_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_30_40_2_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_40_50_2_BRAH" : BRAHMS_mult_rap_bins,
        #
        #
        # "dNdeta_eta_cen_00_05_3_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_05_10_3_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_10_20_3_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_20_30_3_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_30_40_3_BRAH" : BRAHMS_mult_rap_bins,
        # "dNdeta_eta_cen_40_50_3_BRAH" : BRAHMS_mult_rap_bins,


        # "dNdeta_eta_cen_00_05_frwd_BRAH" : BRAHMS_mult_large_rap_bins,
        # "dNdeta_eta_cen_05_10_frwd_BRAH" : BRAHMS_mult_large_rap_bins,
        # "dNdeta_eta_cen_10_20_frwd_BRAH" : BRAHMS_mult_large_rap_bins,
        # "dNdeta_eta_cen_20_30_frwd_BRAH" : BRAHMS_mult_large_rap_bins,
        # "dNdeta_eta_cen_30_40_frwd_BRAH" : BRAHMS_mult_large_rap_bins,
        # "dNdeta_eta_cen_40_50_frwd_BRAH" : BRAHMS_mult_large_rap_bins,


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


        "v22_eta_cen_20_70_STAR" : STAR_v22_rap_bins,
        "v22_eta_cen_03_15_PHOB" : PHOBOS_v2_cen,
        "v22_eta_cen_15_25_PHOB" : PHOBOS_v2_cen,
        "v22_eta_cen_25_50_PHOB" : PHOBOS_v2_cen,

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
        # #"v42_pt_cen_50_60_PHEN" : PHENIX_vn_pt,
        #
        "v22_int_STAR" : STAR_meanpT_cen[:6],
        "v32_int_STAR" : STAR_meanpT_cen[:6],

        #"r2_eta_cen_10_40_STAR" : STAR_rn_eta_bins,
        #"r3_eta_cen_10_40_STAR" : STAR_rn_eta_bins,

        "meanpT_pi_PHEN" : PHENIX_meanpT_cen[:8],
        #"meanpT_k_PHEN"  : PHENIX_meanpT_cen[:8],
        #"meanpT_pi_STAR" : STAR_meanpT_cen[:7],
        #"meanpT_k_STAR" : STAR_meanpT_cen[:7],

    },

}


# these just define some 'reasonable' ranges for plotting purposes
obs_range_list = {
    "Au-Au-200": {
        "dNch_deta": [0, 1000],
        "dET_deta": [0, 1200],
        "dN_dy_pion": [0, 800],
        "dN_dy_kaon": [0, 120],
        "dN_dy_proton": [0, 40],
        "dN_dy_Lambda": [0, 40],
        "dN_dy_Omega": [0, 2],
        "dN_dy_Xi": [0, 10],
        "mean_pT_pion": [0, 1],
        "mean_pT_kaon": [0, 1.5],
        "mean_pT_proton": [0, 2],
        "pT_fluct": [0, 0.05],
        "v22": [0, 0.16],
        "v32": [0, 0.1],
        "v42": [0, 0.1],
    },
}

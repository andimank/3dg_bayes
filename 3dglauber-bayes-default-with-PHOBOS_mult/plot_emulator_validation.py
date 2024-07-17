#!/usr/bin/env python3
"""
Generates plots / figures of a performance comparison between the nominal emulator
and the provided calculations of a second emulator at 5 validation points for various observables.

In the code, each plot is generated by a function tagged with the ``@plot``
decorator, which saves the figure.
"""

import copy
import itertools
import logging
import warnings
from collections import OrderedDict
from pathlib import Path

import dill
#import hsluv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import plotly.graph_objects as go
import seaborn as sns
from matplotlib import lines, patches, ticker
#from plotly.subplots import make_subplots
#from SALib.analyze import sobol
#from SALib.sample import saltelli
from scipy import optimize, stats
from scipy.interpolate import PchipInterpolator

#from bayes_exp import Y_exp_data
from bayes_mcmc import *
#from calculations_load import MAP_data, trimmed_model_data
#from compare_events import model_data_1, model_data_2
from configurations import *
from bins_and_cuts import *
#from design import Design
#from emulator import Trained_Emulators
#from mcmc_diagnostics import autocorrelation
#from priors import *
from emulator import *
#from bayes_exp import Y_exp_data
from calculations_load import trimmed_model_data, validation_data
import random

# read in validation model calculations
dsv = {}
dsv['Au-Au-200'] = np.fromfile("./model_calculations/production_375pts_Au_Au_200/Obs/validation.dat", dtype=bayes_dtype) #, dtype=bayes_dtype
dsv['d-Au-200'] = np.fromfile("./model_calculations/production_375pts_d_Au_200/Obs/validation.dat", dtype=bayes_dtype) #, dtype=bayes_dtype

plotdir = workdir / "emulator_validation" / "" #"plots" /
plotdir.mkdir(exist_ok=True)


# Load the emulators
emu_dict = {}
for s in system_strs:
    # load the dill'ed emulator from emulator file
    print(
        "Loading emulators from emulator/emulator-"
        + s
        + "-idf-"
        + str(idf)
        + ".dill"
    )
    emu_dict[s] = dill.load(
        open("emulator/emulator-" + s + "-idf-" + str(idf) + "-npc-" + str(SystemsInfo[s]["npc"]) + ".dill", "rb")
    )
    #emu = dill.load(
        #open("emulator/emulator-" + s + "-idf-" + str(idf) + ".dill", "rb")
    #)
    print("NPC = " + str(emu_dict[s].npc))

design, design_max, design_min, labels = load_design(s, pset="validation")
#print("Validation design set shape : (Npoints, Nparams) =  ", design.shape)




def plot(f):
    """
        Decorator for plot function. Calls the function and saves the figure produced.
        More functionalities to be added.
    """

    def wrapper():
        logging.info("Generating emulator validation plots!")

        print("Calling function: {}".format(f.__name__))
        figs, sys, obs, spec_plotdir = f()
            # save figures
        for i, figure in enumerate(figs):
            figure.savefig(str(spec_plotdir)+ "/{}_".format(f.__name__) + sys[i] + '_' + obs[i] + '_' + str(SystemsInfo[sys[i]]["npc"]) + 'pc', dpi=150)
    return wrapper

# Plot dictionary
# sets the dataset text label, the plot colors, the axes ranges, the axes labels, the observable scalar, and the axis yscale for each system/observable
Plot_dict = {
'v22_eta_cen_00_05_PHEN' : ["dAu 200 GeV", ['darkorange','purple','navajowhite','plum'], [[-3.2,3.2],[0,18],[-3.2,3.2],[-34,34]], ["$v_2$ (%)", "$\eta$"], 100, 'linear'],
'v22_pt_cen_00_05_PHEN'  : ["dAu 200 GeV", ['red','purple','navajowhite','pink'], [[0,2.2],[0,30],[0,2.2],[-34,34]], ["$v_2$ (%)", "$p_T$"], 100, 'linear'],
'v32_pt_cen_00_05_PHEN'  : ["dAu 200 GeV", ['blue','purple','navajowhite','darkorange'], [[0,2.2],[0,13],[0,2.2],[-34,34]], ["$v_3$ (%)", "$p_T$"], 100, 'linear'],
'v22_pt_cen_00_10_STAR'  : ["dAu 200 GeV", ['green','purple','navajowhite','plum'], [[0,2.2],[0,30],[0,2.2],[-34,34]], ["$v_2$ (%)", "$p_T$"], 100, 'linear'],
'v32_pt_cen_00_10_STAR'  : ["dAu 200 GeV", ['springgreen','purple','black','pink'], [[0,2.2],[0,13],[0,2.2],[-34,34]], ["$v_3$ (%)", "$p_T$"], 100, 'linear'],

'v22_eta_cen_20_70_STAR' : ["AuAu 200 GeV", ['green','royalblue','limegreen','cornflowerblue'], [[-4.4,4.4],[0,18],[-4.4,4.4],[-34,34]], ["$v_2$ (%)", "$\eta$"], 100, 'linear'],
'dNdeta_eta_cen_00_20_PHOB' : ["dAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[.2,1500],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
#'dNch_deta_cen_20_40' : ["dAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[.2,1500],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
#dNch_deta_cen_40_60' : ["dAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[.2,1500],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],

'dNdeta_eta_cen_00_05_BRAH' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_05_10_BRAH' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_10_20_BRAH' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_20_30_BRAH' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_30_40_BRAH' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_40_50_BRAH' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],

'dNdeta_eta_cen_00_05_frwd_BRAH' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-5.8,5.8],[10,2000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_05_10_frwd_BRAH' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-5.8,5.8],[10,2000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],


# 'dNdeta_eta_cen_00_05_BRAH_2' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
# 'dNdeta_eta_cen_05_10_BRAH_2' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
# 'dNdeta_eta_cen_10_20_BRAH_2' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
# 'dNdeta_eta_cen_20_30_BRAH_2' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
# 'dNdeta_eta_cen_30_40_BRAH_2' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
# 'dNdeta_eta_cen_40_50_BRAH_2' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
#
# 'dNdeta_eta_cen_00_05_BRAH_3' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
# 'dNdeta_eta_cen_05_10_BRAH_3' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
# 'dNdeta_eta_cen_10_20_BRAH_3' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
# 'dNdeta_eta_cen_20_30_BRAH_3' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
# 'dNdeta_eta_cen_30_40_BRAH_3' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
# 'dNdeta_eta_cen_40_50_BRAH_3' : ["AuAu 200 GeV", ['turquoise','red','teal','brown'], [[-3.7,3.7],[50,10000],[-3.7,3.7],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],

'dNdeta_eta_cen_00_03_PHOB' : ["AuAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[20,10000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_03_06_PHOB' : ["AuAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[20,10000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_06_10_PHOB' : ["AuAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[20,10000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_10_15_PHOB' : ["AuAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[20,10000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_15_20_PHOB' : ["AuAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[20,10000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_20_25_PHOB' : ["AuAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[20,10000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_25_30_PHOB' : ["AuAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[20,10000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_30_35_PHOB' : ["AuAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[20,10000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_35_40_PHOB' : ["AuAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[20,10000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_40_45_PHOB' : ["AuAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[20,10000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],
'dNdeta_eta_cen_45_50_PHOB' : ["AuAu 200 GeV", ['orangered','darkviolet','lightcoral','blue'], [[-5.8,5.8],[20,10000],[-5.8,5.8],[-34,34]], [r"$\dfrac{dN_{ch}}{d\eta}$", "$\eta$"], 1, 'log'],



"v22_eta_cen_03_15_PHOB"    : ["AuAu 200 GeV", ['green','purple','limegreen','cornflowerblue'], [[-4.4,4.4],[0,10],[-4.4,4.4],[-34,34]], ["$v_2$ (%)", "$\eta$"], 100, 'linear'],
"v22_pt_cen_00_10_PHEN"     : ["AuAu 200 GeV", ['green','blue','limegreen','pink'], [[0,2.2],[0,11],[0,2.2],[-34,34]], ["$v_2$ (%)", "$p_T$"], 100, 'linear'],
"v32_pt_cen_00_10_PHEN"     : ["AuAu 200 GeV", ['red','purple','limegreen','cornflowerblue'], [[0,2.2],[0,9],[0,2.2],[-34,34]], ["$v_2$ (%)", "$p_T$"], 100, 'linear'],
"v42_pt_cen_00_10_PHEN"     : ["AuAu 200 GeV", ['green','pink','blue','cornflowerblue'], [[0,2.2],[0,7],[0,2.2],[-34,34]], ["$v_2$ (%)", "$p_T$"], 100, 'linear'],

'meanpT_pi_PHEN' : ["AuAu 200 GeV", ['blue','lightskyblue','lightskyblue','black'], [[0,85],[0,2.2],[0,85],[-34,34]], [r"$\pi$+ mean $p_T$", "Centrality (%)"], 1, 'linear'],
#'meanpT_k' : ["AuAu 200 GeV", ['red', 'salmon', 'salmon','red'], [[0,85],[0,2.2],[0,85],[-34,34]], [r"k+ mean $p_T$", "Centrality (%)"], 1, 'linear'],
#'meanpT_p' : ["AuAu 200 GeV", ['black', 'violet', 'violet','black'], [[0,85],[0,2.2],[0,85],[-34,34]], [r"p mean $p_T$", "Centrality (%)"], 1, 'linear'],
#'meanpT_pi_minus' : ["AuAu 200 GeV", ['blue','lightskyblue','lightskyblue','black'], [[0,85],[0,2.2],[0,85],[-34,34]], [r"$\pi$- mean $p_T$", "Centrality (%)"], 1, 'linear'],
#'meanpT_k_minus' : ["AuAu 200 GeV", ['red', 'salmon', 'salmon','red'], [[0,85],[0,2.2],[0,85],[-34,34]], [r"k- mean $p_T$", "Centrality (%)"], 1, 'linear'],
#'meanpT_p_bar' : ["AuAu 200 GeV", ['black', 'violet', 'violet','black'], [[0,85],[0,2.2],[0,85],[-34,34]], [r"p bar mean $p_T$", "Centrality (%)"], 1, 'linear'],

#'dNpideta' : ["AuAu 200 GeV", ['blue','lightskyblue','lightskyblue','black'], [[0,85],[0.01,10000],[0,85],[-34,34]], [r"$\dfrac{dN_{pi+}}{d\eta}$", "Centrality (%)"], 1, 'log'],
#'dNkdeta' : ["AuAu 200 GeV", ['red', 'salmon', 'salmon','red'], [[0,85],[0.01,10000],[0,85],[-34,34]], [r"$\dfrac{dN_{k+}}{d\eta}$", "Centrality (%)"], 1, 'log'],
#'dNpdeta' : ["AuAu 200 GeV", ['black', 'violet', 'violet','black'], [[0,85],[0.01,10000],[0,85],[-34,34]], [r"$\dfrac{dN_{p}}{d\eta}$", "Centrality (%)"], 1, 'log'],
#'dNpiminusdeta' : ["AuAu 200 GeV", ['blue','lightskyblue','lightskyblue','black'], [[0,85],[0.01,10000],[0,85],[-34,34]], [r"$\dfrac{dN_{pi-}}{d\eta}$", "Centrality (%)"], 1, 'log'],
#'dNkminusdeta' : ["AuAu 200 GeV", ['red', 'salmon', 'salmon','red'], [[0,85],[0.01,10000],[0,85],[-34,34]], [r"$\dfrac{dN_{k-}}{d\eta}$", "Centrality (%)"], 1, 'log'],
#'dNpbardeta' : ["AuAu 200 GeV", ['black', 'violet', 'violet','black'], [[0,85],[0.01,10000],[0,85],[-34,34]], [r"$\dfrac{dN_{pbar}}{d\eta}$", "Centrality (%)"], 1, 'log'],

}


@plot
def plot_emu_five_val_pts():

    # create specific plot directory to store this type of emulator diagnostic plot
    spec_plotdir = workdir / "emulator_validation" / "five_validation_points" / ""
    spec_plotdir.mkdir(exist_ok=True)

    # set flag to plot residuals between emulator predictions and model calculations
    plot_residuals = True
    # create lists to be returned to decorator wrapper
    list_of_figs = []; list_of_sys = []; list_of_obs = [];

    for sys in system_strs:
        active_observables = list(obs_cent_list[sys].keys())
        for obs in list(obs_cent_list[sys].keys()):
            # if statement to specify plotting a subset of obseravbles if desired
            active_observables = list(obs_cent_list[sys].keys())
            if (obs in active_observables) and (obs in Plot_dict.keys()):

                # create figure
                fig = plt.figure(figsize=(16, 6)) #constrained_layout=True

                # read plotting scalar and yscale from dictionary for each observable
                scale = Plot_dict[obs][4]; yscale = Plot_dict[obs][5];

                #--------------------------------------------------------------------------------------------------------#
                # Create main plot frames
                    # set axes ranges
                xlim_main = Plot_dict[obs][2][0]; ylim_main = Plot_dict[obs][2][1]
                    # define axes
                frame1=fig.add_axes((.06,.4,.186,.5),xlim=xlim_main,ylim=ylim_main,yscale=yscale)
                frame1.set_ylabel(Plot_dict[obs][3][0], fontsize=14.5)
                frame2=fig.add_axes((.246,.4,.186,.5),xlim=xlim_main,ylim=ylim_main,yscale=yscale)
                frame3=fig.add_axes((.432,.4,.186,.5),xlim=xlim_main,ylim=ylim_main,yscale=yscale)
                frame4=fig.add_axes((.618,.4,.186,.5),xlim=xlim_main,ylim=ylim_main,yscale=yscale)
                frame5=fig.add_axes((.804,.4,.186,.5),xlim=xlim_main,ylim=ylim_main,yscale=yscale)
                if plot_residuals == True:
                    frame1.set_xticklabels([]);frame2.set_xticklabels([]);frame3.set_xticklabels([]);frame4.set_xticklabels([]);frame5.set_xticklabels([])
                else:
                    frame1.set_xlabel(Plot_dict[obs][3][1], fontsize=14); frame2.set_xlabel(Plot_dict[obs][3][1], fontsize=14); frame3.set_xlabel(Plot_dict[obs][3][1], fontsize=14); frame4.set_xlabel(Plot_dict[obs][3][1], fontsize=14); frame5.set_xlabel(Plot_dict[obs][3][1], fontsize=14)
                frame2.set_yticklabels([]);frame3.set_yticklabels([]);frame4.set_yticklabels([]);frame5.set_yticklabels([])

                if plot_residuals == True:
                    # Create residual plot frames
                        # set axes ranges
                    xlim_res = Plot_dict[obs][2][2]; ylim_res = Plot_dict[obs][2][3]
                        # define axes
                    frame11=fig.add_axes((.06,0.1,.186,.3),xlim=xlim_res,ylim=ylim_res) #yscale=yscale
                    frame11.set_ylabel("% Error", fontsize=14)
                    frame12=fig.add_axes((.246,0.1,.186,.3),xlim=xlim_res,ylim=ylim_res)
                    frame13=fig.add_axes((.432,0.1,.186,.3),xlim=xlim_res,ylim=ylim_res)
                    frame14=fig.add_axes((.618,0.1,.186,.3),xlim=xlim_res,ylim=ylim_res)
                    frame15=fig.add_axes((.804,0.1,.186,.3),xlim=xlim_res,ylim=ylim_res)
                    frame12.set_yticklabels([]);frame13.set_yticklabels([]);frame14.set_yticklabels([]);frame15.set_yticklabels([])
                    frame11.set_xlabel(Plot_dict[obs][3][1], fontsize=14); frame12.set_xlabel(Plot_dict[obs][3][1], fontsize=14); frame13.set_xlabel(Plot_dict[obs][3][1], fontsize=14); frame14.set_xlabel(Plot_dict[obs][3][1], fontsize=14); frame15.set_xlabel(Plot_dict[obs][3][1], fontsize=14);
                #--------------------------------------------------------------------------------------------------------#


                #--------------------------------------------------------------------------------------------------------#
                # Plot emulator predictions and uncertainty along with model calculations
                for i, axes_frame in enumerate([frame1,frame2,frame3,frame4,frame5]):
                    bin_cents = np.mean(obs_cent_list[sys][obs], axis=1)

                    # run emulator
                    params = design.iloc[i].values
                    mean, cov = emu_dict[sys].predict(np.array([params]), return_cov=True)
                    mean = mean[obs].flatten(); err = (np.diagonal(np.abs(cov[obs, obs][0])) ** 0.5);

                     # for log(mult)-trained emulators only
                    is_mult = ("dN" in obs) or ("dET" in obs)
                    if is_mult and transform_multiplicities:
                            # direct approximation
                        err = np.exp(mean+err)-np.exp(mean)
                            # following log-norm formula
                        #err = np.sqrt(np.exp(mean)**2 * (np.exp(err**2) - 1))
                        mean = np.exp(mean) #-1

                    # model
                    model_mean = dsv[sys][sys][obs]['mean'][i]
                    axes_frame.plot(bin_cents,scale*model_mean, '+k', label = 'Model')

                    # nominal emulator
                    axes_frame.plot(bin_cents,scale*mean,'o', color = Plot_dict[obs][1][1], fillstyle='none', label = 'Nominal Emu.', markersize = 5)

                    # uncertainty of nominal emulator (plot 1 and/or 2 sigma)
                    y = mean;
                    y2 = scale*(y + err); y1 = scale*(y - err)
                    y4 = scale*(y + 2*err); y3 = scale*(y - 2*err)
                    axes_frame.fill_between(bin_cents, y1, y2, interpolate=True, alpha=0.7, color= Plot_dict[obs][1][3])
                    axes_frame.fill_between(bin_cents, y3, y4, interpolate=True, alpha=0.3, color= Plot_dict[obs][1][3])

                        # validation point label
                    axes_frame.text(.04, .96, "Validation point " + str(i), ha='left', va='top', transform=axes_frame.transAxes, fontsize = 15, family = "serif")

                    # set the legend on the leftmost subplot/frame
                    if i == 0:
                        axes_frame.legend(loc=(.04,.62),fontsize = 11.5)
                    if not(plot_residuals) == True:
                        # system and centrality label
                        axes_frame.text(.04, .1 ,Plot_dict[obs][0], ha='left', va='top', transform=axes_frame.transAxes, family = "serif",fontsize = 14)


                if plot_residuals:
                    # Plot residuals
                    for i, axes_frame in enumerate([frame11,frame12,frame13,frame14,frame15]):

                        # run emulator
                        params = design.iloc[i].values
                        mean, cov = emu_dict[sys].predict(np.array([params]), return_cov=True)
                        mean = mean[obs].flatten(); err = (np.diagonal(np.abs(cov[obs, obs][0])) ** 0.5);

                         # for log(mult)-trained emulators only
                        is_mult = ("dN" in obs) or ("dET" in obs)
                        if is_mult and transform_multiplicities:
                                # direct approximation
                            err = np.exp(mean+err)-np.exp(mean)
                                # following log-norm formula
                            #err = np.sqrt(np.exp(mean)**2 * (np.exp(err**2) - 1))
                            mean = np.exp(mean) #-1

                        # model
                        model_mean = dsv[sys][sys][obs]['mean'][i]

                        # plot the 0 line
                        axes_frame.plot([-1000,1000],[0,0],'--k')
                        # emulator - model % residual
                        axes_frame.plot(bin_cents,100*(model_mean-mean)/model_mean,'s', color = 'k',fillstyle='none', label = '% (Model - Emu) / Model')

                        # set the legend on the leftmost subplot/frame
                        if i == 0:
                            axes_frame.legend(loc=(.04,.70), fontsize = 11.5)
                        # system and obs label
                        axes_frame.text(.04, .25 ,Plot_dict[obs][0], ha='left', va='top', transform=axes_frame.transAxes, family = "serif",fontsize = 14)
                        axes_frame.text(.04, .15 ,obs, ha='left', va='top', transform=axes_frame.transAxes, family = "serif",fontsize = 14)
                #--------------------------------------------------------------------------------------------------------#

                plt.suptitle('Emulator Predictions and Uncertainty along with Model Caluclations for 5 Validation Points',family = "serif", fontsize = 18)
                list_of_figs.append(fig); list_of_sys.append(sys); list_of_obs.append(obs)

        print("\nPlotting emulator predictions and model calculations for five validation points for: \n")
        print(Plot_dict.keys())
    return list_of_figs, list_of_sys, list_of_obs, spec_plotdir




@plot
def plot_emu_single_bin_scatter():

    # create specific plot directory to store this type of emulator diagnostic plot
    spec_plotdir = workdir / "emulator_validation" / "single_bin_scatter_plots" / ""
    spec_plotdir.mkdir(exist_ok=True)

    # create lists to be returned to decorator wrapper
    list_of_figs = []; list_of_sys = []; list_of_obs = [];

    for sys in system_strs:
        active_observables = list(obs_cent_list[sys].keys())
        for obs in list(obs_cent_list[sys].keys()):
            # if statement to specify plotting a subset of obseravbles if desired
            #active_observables = ['meanpT_pi']
            if obs in active_observables and (obs in Plot_dict.keys()):

                # create figure
                fig = plt.figure(figsize=(6, 6)) #constrained_layout=True
                ax = fig.add_subplot(1,1,1)
                # read plotting scalar and yscale from dictionary for each observable
                scale = Plot_dict[obs][4]; yscale = Plot_dict[obs][5];

                # randomly select bin to plot and construct bin label
                bin_cents = np.mean(obs_cent_list[sys][obs], axis=1)
                bin_idx = np.random.randint(0,len(bin_cents))
                bin_label = str(obs_cent_list[sys][obs][bin_idx]) + ' ' + Plot_dict[obs][3][1]

                # initialize dynamic axis plotting bounds
                plot_min = 1000; plot_max = 0

                # initialize error sum
                err_sum = 0

                # calculate points
                n_pts = len(design)
                for i in range(n_pts):

                    # run emulator
                    params = design.iloc[i].values
                    mean = emu_dict[sys].predict(np.array([params]), return_cov=False)
                    mean = mean[obs].flatten()

                     # for log(mult)-trained emulators only
                    is_mult = ("dN" in obs) or ("dET" in obs)
                    if is_mult and transform_multiplicities:
                        mean = np.exp(mean) #-1

                    # read model calculation
                    model_mean = dsv[sys][sys][obs]['mean'][i]

                    # pick out the selected bin
                    mean = scale*mean[bin_idx]
                    model_mean = scale*model_mean[bin_idx]

                    # update the running relative error sum
                    err_sum += ((np.abs(model_mean - mean)/model_mean))**2

                    # update dynamic axis plotting bounds
                    if mean < plot_min:
                        plot_min = mean
                    if mean > plot_max:
                        plot_max = mean

                    # plot the points
                    plt.scatter(mean,model_mean, color = Plot_dict[obs][1][1])

                # calculate the average and take the root percent error
                avg_rootper_err = 100*np.sqrt(err_sum/n_pts)

                plt.plot([-10000,10000],[-10000,10000], 'k-')
                plt.xlim([plot_min*0.9,plot_max*1.1]); plt.ylim([plot_min*0.9,plot_max*1.1])

                plt.text(.04, .96, Plot_dict[obs][0], ha='left', va='top', transform=ax.transAxes, fontsize = 15, family = "serif")
                plt.text(.04, .89, obs, ha='left', va='top', transform=ax.transAxes, fontsize = 15, family = "serif")
                plt.text(.04, .82, bin_label, ha='left', va='top', transform=ax.transAxes, fontsize = 15, family = "serif")
                plt.text(.04, .75, "Avg. RMS % Err. = " + str(np.round(avg_rootper_err,2)) + "%", ha='left', va='top', transform=ax.transAxes, fontsize = 15, family = "serif")

                ax.yaxis.set_ticks_position('both')
                ax.xaxis.set_ticks_position('both')
                plt.minorticks_on()
                ax.tick_params(axis="y",direction="in", pad=5)
                ax.tick_params(axis="x",direction="in", pad=5)

                plt.xlabel("Emulator prediction", fontsize = 16, family = "serif")
                plt.ylabel("Model calculation", fontsize = 16, family = "serif")
                plt.title("Model vs Emulator", fontsize = 17, family = "serif")

                list_of_figs.append(fig); list_of_sys.append(sys); list_of_obs.append(obs)

        print("\nPlotting emulator predictions and model calculations for all points for a single bin of: \n")
        print(Plot_dict.keys())
    return list_of_figs, list_of_sys, list_of_obs, spec_plotdir




@plot
def plot_emu_RMSPE_emu_uncert():

    # create specific plot directory to store this type of emulator diagnostic plot
    spec_plotdir = workdir / "emulator_validation" / "RMSPE_and_emu_uncertainty" / ""
    spec_plotdir.mkdir(exist_ok=True)

    # create lists to be returned to decorator wrapper
    list_of_figs = []; list_of_sys = []; list_of_obs = [];

    for sys in system_strs:
        active_observables = list(obs_cent_list[sys].keys())
        for obs in list(obs_cent_list[sys].keys()):
                # if statement to specify plotting a subset of obseravbles if desired
                #active_observables = ['meanpT_pi']
                if (obs in active_observables) and (obs in Plot_dict.keys()):

                    # create figure
                    fig = plt.figure(figsize=(6, 6)) #constrained_layout=True
                    ax = fig.add_subplot(1,1,1)

                    n_bins = len(obs_cent_list[sys][obs])
                    # define arrays for storing sums
                    rel_sqrerr_avg = np.zeros(n_bins); per_err_avg = np.zeros(n_bins)

                    n_pts = len(design)
                    # loop through the validation design
                    for i in range(n_pts):
                        # parameter set for design point
                        params = design.iloc[i].values
                        # run emulator for given parameter set
                        mean, cov = emu_dict[sys].predict(np.array([params]), return_cov=True)
                        # flatten mean aray and and pick out the standard deviation
                        mean = mean[obs].flatten(); err = (np.diagonal(np.abs(cov[obs, obs][0])) ** 0.5)

                        # separately treat multiplicity and non-multiplicity observables
                        is_mult = ("dN" in obs) or ("dET" in obs)
                        if is_mult and transform_multiplicities:
                            # convert mean and standard deviation from log(mult)- to mult-space
                            err = (np.exp(mean+err) - np.exp(mean)); mean = np.exp(mean);

                        # relative square error between emulator prediction and model calculation
                        model_mean = dsv[sys][sys][obs]['mean'][i]
                        rel_sqrerr = (mean/model_mean - 1)**2

                        # emulator uncertainty as a percentage of the emulator mean prediction
                        per_err = (err/mean)*100

                        # sum the relative error and percent uncertainty contributions from all the validation points
                        rel_sqrerr_avg += rel_sqrerr
                        per_err_avg += per_err

                    # get the average among validation points
                    rel_sqrerr_avg = rel_sqrerr_avg/n_pts
                    per_err_avg = per_err_avg/n_pts

                    # take the root of the relative square error and convert to RMSPE
                    rmspe = np.sqrt(rel_sqrerr_avg)*100

                    # bin centers for plotting
                    bin_cents = np.mean(obs_cent_list[sys][obs], axis=1)

                    # make RMSPE and Emu. Uncertainty plots
                    plt.plot(bin_cents,rmspe, 's', color = Plot_dict[obs][1][1], markersize = 5, alpha = .8, label = 'RMSPE')
                    plt.plot(bin_cents,per_err_avg, 'sk', fillstyle='none', markersize = 5, alpha = .8, label = 'Emu. Uncertainty')

                    plt.xlim([bin_cents[0]-0.5,bin_cents[-1]+0.5]); plt.ylim([0,60])

                    plt.text(.04, .96, Plot_dict[obs][0] + '  -  ' + obs, ha='left', va='top', transform=ax.transAxes, fontsize = 15, family = "serif")
                    plt.text(.04, .9, r'RMSPE = $ \sqrt{ \dfrac{1}{N_{val}} \sum_{n=1}^{N_{val}} (\dfrac{y_{emu}}{y_{model}} - 1)^2} $ x $100\%$', ha='left', va='top', transform=ax.transAxes, fontsize = 13, family = "serif")
                    plt.text(.04, .75, r'Emu. Uncert. = $ \dfrac{1}{N_{val}} \sum_{n=1}^{N_{val}} \dfrac{\sigma_{emu}}{\mu_{emu}} $ x $100\%$', ha='left', va='top', transform=ax.transAxes, fontsize = 13, family = "serif")

                    # set the legend
                    plt.legend(loc=(.04,.48), fontsize = 11.5)

                    ax.yaxis.set_ticks_position('both')
                    ax.xaxis.set_ticks_position('both')
                    plt.minorticks_on()
                    ax.tick_params(axis="y",direction="in", pad=5)
                    ax.tick_params(axis="x",direction="in", pad=5)

                    plt.xlabel(Plot_dict[obs][3][1], fontsize=14, family = "serif")
                    plt.ylabel("%", fontsize = 18, family = "serif")
                    plt.title("Root Mean Square Error and Emulator Uncertainty", fontsize = 15, family = "serif")

                    list_of_figs.append(fig); list_of_sys.append(sys); list_of_obs.append(obs)
        print("\nPlotting RMSPE and emulator uncertainty averaged over all validation points for: \n")
        print(Plot_dict.keys())
    return list_of_figs, list_of_sys, list_of_obs, spec_plotdir




# Call plotting functions to be run
plot_emu_five_val_pts()
plot_emu_single_bin_scatter()
plot_emu_RMSPE_emu_uncert()

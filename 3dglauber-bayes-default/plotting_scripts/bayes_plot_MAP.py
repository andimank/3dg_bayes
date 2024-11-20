from configurations import *
from bins_and_cuts import *
from bayes_exp import Y_exp_data
from calculations_load import trimmed_model_data
import matplotlib.pyplot as plt
import dill
from matplotlib import lines, patches, ticker
from emulator import *


MAP_10_iter = [ 0.95238743,  3.94851671,  0.98285454,  0.77659349,  0.27602136,  0.19291016,
  0.21069501, 12.84906179,  0.49649796,  0.60225757,  0.55534187,  0.22343433,
 -1.69289836,  0.7673603,   0.03909206,  0.10256354,  0.22903833,  0.04532772,
 -0.30885663,  0.56965297]

# 966 iterations
MAP = [ 1.88390055e-01,  3.17747161e+00,  1.89067202e+00,  9.60504520e-01,
  1.00001452e-03,  4.79297981e-01,  2.00001363e-01,  2.12701486e+01,
  4.32353479e-01,  7.99999976e-01,  9.99991586e-01,  2.29995734e-01,
 -1.99999883e+00, -8.96047350e-01,  1.20491669e-02,  1.51026518e-01,
  2.47844977e-01,  3.53000756e-02, -7.99999976e-01,  5.74998416e-01]

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

# loop through the systems
for s in system_strs:
    fig= plt.figure(constrained_layout=True, figsize=(6, 12.5)) #constrained_layout=True
    plot_num = 1
    for obs in active_obs_list[s]:
        ax = fig.add_subplot(9,6,plot_num)
        x_bins = np.mean(obs_cent_list[s][obs],axis=1)

        # experimental data
        obsdata = Y_exp_data[s][obs]["mean"]
        obsdata_err = (Y_exp_data[s][obs]["err"])
        plt.errorbar(x_bins, obsdata, yerr=obsdata_err, linestyle='', marker='s', mfc='k', mec='k', ms=3.5)

        # MAP calculations
        params = MAP
        # run emulator
        mean, cov = emu_dict[s].predict(np.array([params]), return_cov=True)
        mean = mean[obs].flatten(); err = (np.diagonal(np.abs(cov[obs, obs][0])) ** 0.5);
        is_mult = ("dN" in obs) or ("dET" in obs)
        if is_mult and transform_multiplicities:
            err = np.exp(mean+err) - np.exp(mean)
            mean = np.exp(mean)
        plt.plot(x_bins,mean,'-r')
        #plt.fill_between(x_bins,mean-err,mean+err,color='r',alpha=0.25)

        # compute chi2 for the observable
        dof = len(mean)

        chi2 = np.sum((obsdata - mean)**2/(obsdata_err+err)**2)

        chi2_nu = chi2/dof
        chi2_nu = np.round(chi2_nu,3)

        ax.text(.05, .95, '$\u03C7^2$=' + str(chi2_nu), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize = 9)
        #ax.text(.95, .95, str(len(x_bins)) + ' bins', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        ax.text(.05, .05, str(obs), horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)
        plot_num += 1

        #ax.legend([ax], [obs], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False)
        #ax.legend(loc="upper right")
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(axis="y",direction="in", pad=10)
        ax.tick_params(axis="x",direction="in", pad=10)

        is_mult = ("dN" in obs) or ("dET" in obs)
        #if is_mult and transform_multiplicities:
            #plt.yscale("log")
    plt.suptitle(s,fontsize = 16)
    plt.show()

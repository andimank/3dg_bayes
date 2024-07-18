from configurations import *
from bins_and_cuts import *
from bayes_exp import Y_exp_data
from calculations_load import trimmed_model_data
import matplotlib.pyplot as plt

# loop through the systems
for s in system_strs:
    fig= plt.figure(constrained_layout=True,figsize=(4, 8))
    plot_num = 1
    for obs in active_obs_list[s]:
        ax = fig.add_subplot(13,4,plot_num)
        x_bins = np.mean(obs_cent_list[s][obs],axis=1)

        # experimental data
        obsdata = Y_exp_data[s][obs]["mean"]
        obsdata_err = (Y_exp_data[s][obs]["err"])
        plt.errorbar(x_bins, obsdata, yerr=obsdata_err, linestyle='', marker='s', mfc='k', mec='k', ms=3.5)

        # saved model calculations
        for ipt, data in enumerate(trimmed_model_data[s][obs]):
            values = np.array(data["mean"])
            is_mult = ("dN" in obs) or ("dET" in obs)
            if is_mult and transform_multiplicities:
                values = np.exp(values)
            # adjustable condition to pick out weird designs of any observable:
            #if (obs == "v22PHENIX") and any(y > .2 for y in values):
            #if ipt == 349:
                #print(ipt)
                #ax.plot(x_bins, values, '-b',alpha=0.5)
            # the rest of the calculations
            #else:
            ax.plot(x_bins, values, '-r',alpha=0.05)

        ax.text(.95, .95, str(len(x_bins)) + ' bins', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        ax.text(.05, .05, str(obs), horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)
        plot_num += 1

        #ax.legend([ax], [obs], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False)
        #ax.legend(loc="upper right")
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(axis="y",direction="in", pad=5)
        ax.tick_params(axis="x",direction="in", pad=5)

        is_mult = ("dN" in obs) or ("dET" in obs)
        if is_mult and transform_multiplicities:
            plt.yscale("log")
    plt.suptitle(s,fontsize = 16)
    plt.show()

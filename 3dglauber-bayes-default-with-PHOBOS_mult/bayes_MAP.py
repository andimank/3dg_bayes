import argparse
import logging
import time
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count

import emcee
import h5py
import numpy as np
import ptemcee
from scipy.linalg import lapack

from bayes_exp import Y_exp_data
from configurations import *
from emulator import Trained_Emulators, Trained_Emulators_all_df
from emulator import _Covariance
#from priors import *  # Import priors

from scipy import optimize


def compute_cov(obs1, obs2, dy1, dy2):
    #dy1 = np.array([dy1])
    #dy2 = np.array([dy2])
    x1 = np.linspace(0, 1, len(dy1))
    x2 = np.linspace(0, 1, len(dy2))
    if assume_corr_exp_error:
        if obs1 == obs2:
            cov_mat = np.outer(dy1, dy2) * np.exp(
                -np.subtract.outer(x1, x2) ** 2 / cent_corr_length**2
            )
        elif expt_obs_corr_group[obs1] == expt_obs_corr_group[obs2]:
            cov_mat = (
                np.outer(dy1, dy2)
                * np.exp(-np.subtract.outer(x1, x2) ** 2 / cent_corr_length**2)
                * 0.8
            )
        else:
            cov_mat = np.zeros([dy1.size, dy2.size])

    else:
        if obs1 == obs2:
            is_mult_1 = ("dN" in obs1) or ("dET" in obs1)
            if is_mult_1 and transform_multiplicities:
                cov_mat = np.diag(np.log(dy1**2 + 1)) #0.01+ <------ change this to add extra std to multiplicity only
            else:
                cov_mat = np.diag((dy1**2))
        else:
            cov_mat = np.zeros([dy1.size, dy2.size])

    return cov_mat


slices   =  {}
expt_y   =  {}
expt_cov =  {}

Yexp = Y_exp_data

# if this flag is set to "True", the whole parameter set is selected to be in the "common" list
set_common = True

sysdep_max = []
sysdep_min = []
sysdep_labels = []
sys_idx = {}  # keep track what parameters are used for each system
sysdep_range = np.array([sysdep_min, sysdep_max]).T
sysdep_ndim = len(sysdep_max)
Nsys = len(system_strs)

for i, s in enumerate(system_strs):
    _, design_max, design_min, labels = load_design(s, pset="main")

    # pick out the system-dependent parameters and add them to lists (these remain empty in current default analysis)
    #self.sysdep_max.append(design_max[0])
    #self.sysdep_min.append(design_min[0])
    #self.sysdep_labels.append(labels[0])

    # the indecies of the parameters used in each system (the full set is used in each system in this analysis)
    #dummy = len(design_max)
    #print(type(dummy))
    #dummy = range(int(dummy))
    #dummy = list(dummy)

    #sys_idx[s] = dummy
    sys_idx[s] = list(range(len(design_max))) #[i] + list(range(Nsys, Nsys + len(design_max) - 1))

    # if this flag is set to "True", the whole parameter set is selected to be in the "common" list
    if set_common:

        common_max = np.array(list(design_max[0:]))
        common_min = np.array(list(design_min[0:]))
        common_labels = list(labels[0:])
        common_ndim = len(common_max)
        common_range = np.array([common_min, common_max]).T

    else:
        continue
        # if using system-dependent parameters, select here the parameters that
        # will remain the same/common across the systems

        #
        #
        #
        #
        #

    # combine the (by default, empty) system-dependent parameter lists with the common parameter lists for each system
    max = np.concatenate([sysdep_max, common_max])
    min = np.concatenate([sysdep_min, common_min])
    #range = np.array([min, max]).T
    labels = common_labels
    ndim = common_ndim

# the n-dimensional volume of the uniform prior
diff = max - min
prior_volume = np.prod(diff)

#logging.info("Pre-compute experimental covariance matrix")

# read the experimental data from the bayes_dtype array into obseravble "sub-blocks" or slices to match emulator format
for s in system_strs:
    nobs = 0
    slices[s] = []

    for obs in active_obs_list[s]:
        try:
            if number_of_models_per_run > 1:
                obsdata = Yexp[s][obs]["mean"][idf]
            else:
                obsdata = Yexp[s][obs]["mean"]
        except KeyError:
            continue

        n = obsdata.size
        slices[s].append((obs, slice(nobs, nobs + n)))
        nobs += n

    expt_y[s] = np.empty(nobs)
    expt_cov[s] = np.empty((nobs, nobs))

    for obs1, slc1 in slices[s]:
        is_mult_1 = ("dN" in obs1) or ("dET" in obs1)
        if is_mult_1 and transform_multiplicities:
            if number_of_models_per_run > 1:
                expt_y[s][slc1] = np.log(Yexp[s][obs1]["mean"][idf] + 1.0)
                dy1 = Yexp[s][obs1]["err"][idf] / (
                    Yexp[s][obs1]["mean"][idf] + 1.0
                )
            else:
                expt_y[s][slc1] = np.log(Yexp[s][obs1]["mean"]) #+ 1.0)
                dy1 = Yexp[s][obs1]["err"] / (Yexp[s][obs1]["mean"]) #+ 1.0)
        else:
            if number_of_models_per_run > 1:
                expt_y[s][slc1] = Yexp[s][obs1]["mean"][idf]
                dy1 = Yexp[s][obs1]["err"][idf]
            else:
                expt_y[s][slc1] = Yexp[s][obs1]["mean"]
                dy1 = Yexp[s][obs1]["err"]

        for obs2, slc2 in slices[s]:
            is_mult_2 = ("dN" in obs2) or ("dET" in obs2)
            if is_mult_2 and transform_multiplicities:
                if number_of_models_per_run > 1:
                    dy2 = Yexp[s][obs2]["err"][idf] / (
                        Yexp[s][obs2]["mean"][idf] + 1.0
                    )
                else:
                    dy2 = Yexp[s][obs2]["err"] / (Yexp[s][obs2]["mean"]) #+ 1.0)
            else:
                if number_of_models_per_run > 1:
                    dy2 = Yexp[s][obs2]["err"][idf]
                else:
                    dy2 = Yexp[s][obs2]["err"]
            #if is_mult_2 and transform_multiplicities:
            expt_cov[s][slc1, slc2] = compute_cov(obs1, obs2, dy1, dy2)



def predict_obs(X, **kwargs):
    """
    Call each system emulator to predict model output at X. (using df model specified by idf in configurations.py)

    """
    # flag for setting specific parameter values to a constant in every emulator prediction
    #if hold_parameters:
    #    for idx, value in self.hold:
    #        X[:, idx] = value
    return {
        s: Trained_Emulators[s].predict(X[:, ], **kwargs)
        for s in system_strs                    # self.sys_idx[s]
    }

def mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    This likelihood is NOT NORMALIZED, since this does not affect parameter estimation.
    The normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """

    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            "lapack dpotrf error: "
            "the {}-th argument had an illegal value".format(-info)
        )
    elif info < 0:
        raise np.linalg.LinAlgError(
            "lapack dpotrf error: "
            "the leading minor of order {} is not positive definite".format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            "lapack dpotrs error: "
            "the {}-th argument had an illegal value".format(-info)
        )
    np.fill_diagonal(L,np.maximum(1e-10,np.diag(L)))
    return -0.5 * np.dot(y, alpha) - np.log(L.diagonal()).sum()


def log_posterior(X, extra_std_prior_scale=0.001):
    """
    Evaluate the posterior at `X`.

    `extra_std_prior_scale` is the scale parameter for the prior
    distribution on the model sys error parameter:

        prior ~ sigma^2 * exp(-sigma/scale)

    """
    X = np.array(X, copy=False, ndmin=2)

    lp = np.zeros(X.shape[0])

    inside = np.all((X > min) & (X < max), axis=1)
    lp[~inside] = -np.inf

    nsamples = np.count_nonzero(inside)
    if nsamples > 0:

        pred = predict_obs(X[inside], return_cov=True)
        for sys in system_strs:
            nobs = expt_y[sys].size
            # allocate difference (model - expt) and covariance arrays
            dY = np.empty((nsamples, nobs))
            cov = np.empty((nsamples, nobs, nobs))

            Y_pred, cov_pred = pred[sys]

            # copy predictive mean and covariance into allocated arrays
            for obs1, slc1 in slices[sys]:
                dY[:, slc1] = Y_pred[obs1] - expt_y[sys][slc1]
                for obs2, slc2 in slices[sys]:
                    cov[:, slc1, slc2] = cov_pred[obs1, obs2]

            # add expt cov to model cov
            cov += expt_cov[sys]

            # compute log likelihood at each point, w/o normalization
            #lp[inside] += list(map(mvn_loglike, dY, cov))
            lp[inside] += list(map(mvn_loglike,dY, cov))

        # Remove extra_std but leave framework for it in place
        ## add prior for extra_std (model sys error)
        # lp[inside] += 2*np.log(extra_std) - extra_std/extra_std_prior_scale

    return lp

#test_x = [0.14,0.14,0.14,0.14,0.14,0.14,0.24,5,0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.14,0.14,]
#test_x = np.array(test_x, copy=False, ndmin=2)
#print(log_posterior(test_x))
#print(((test_x > min) & (test_x < max)))
#print(min)
#print(max)

# If false, do not try to find the Maximum A posterior values.
find_map_param = True
if find_map_param == True:
    bounds=[(a,b) for (a,b) in zip(min,max)]
    result = optimize.differential_evolution(lambda x: -log_posterior(x),
                                           bounds=bounds,
                                           maxiter=100,
                                          disp=True,
                                          tol=1e-9,
                                          seed=16,

                                         )
    print(result.x)

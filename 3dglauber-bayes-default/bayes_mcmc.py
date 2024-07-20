#!/usr/bin/env python3
"""
Markov chain Monte Carlo model calibration using the `affine-invariant ensemble
sampler (emcee) <http://dfm.io/emcee>`_.

This module must be run explicitly to create the posterior distribution.
Run ``python -m src.mcmc --help`` for complete usage information.

On first run, the number of walkers and burn-in steps must be specified, e.g.
::

    python -m src.mcmc --nwalkers 500 --nburnsteps 100 200

would run 500 walkers for 100 burn-in steps followed by 200 production steps.
This will create the HDF5 file :file:`mcmc/chain.hdf` (default path).

On subsequent runs, the chain resumes from the last point and the number of
walkers is inferred from the chain, so only the number of production steps is
required, e.g. ::

    python -m src.mcmc 300

would run an additional 300 production steps (total of 500).

To restart the chain, delete (or rename) the chain HDF5 file.
"""

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

logging.getLogger().setLevel(logging.INFO)

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


def normed_mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    This likelihood IS NORMALIZED.
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

    n = len(y)
    norm_const = -n / (2.0 * np.log(2.0 * np.pi))
    return -0.5 * np.dot(y, alpha) - np.log(L.diagonal()).sum() + norm_const


class LoggingEnsembleSampler(emcee.EnsembleSampler):
    def run_mcmc(self, X0, nsteps, status=None, **kwargs):
        """
        Run MCMC with logging every 'status' steps (default: approx 10% of
        nsteps).

        """

        #logging.info("running {:d} walkers for {:d} steps".format(self.k, nsteps))

        if status is None:
            status = nsteps // 10

        for n, result in enumerate(
            self.sample(X0, iterations=nsteps, **kwargs), start=1
        ):
            if n % status == 0 or n == nsteps:
                af = self.acceptance_fraction
                logging.info(
                    "step {:d}: acceptance fraction: "
                    "mean {:.4f}, std {:.4f}, min {:.4f}, max {:.4f}".format(
                        n, af.mean(), af.std(), af.min(), af.max()
                    )
                )

        return result


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


class Chain:
    """
    High-level interface for running MCMC calibration and accessing results.

    Currently all design parameters except for the normalizations are required
    to be the same at all beam energies.  It is assumed (NOT checked) that all
    system designs have the same parameters and ranges (except for the norms).

    """
    if run_closure==False:
        filename = "chain-idf-{:d}.hdf".format(idf)
    else:
        filename = "chain-idf-0-closure-val-pt_{:d}.hdf".format(closure_val_pt)

    def __init__(self, path=workdir / "mcmc" / filename):
        self.path = path
        self.path.parent.mkdir(exist_ok=True)

        self._slices = {}
        self._expt_y = {}
        self._expt_cov = {}

        Yexp = Y_exp_data

        # For multi-system calibration, we need to specify what parameters
        # are considered universal (the same across all system), and what
        # parameters are allowed to be a function of projectile / target /
        # collision energy
        # Now, for simplicity, I only let the energy density normalization
        # of the initial condition to be system-dependent, while others are
        # universal in the high-energy limit

        # if this flag is set to "True", the whole parameter set is selected to be in the "common" list
        set_common = True

        self.sysdep_max = []
        self.sysdep_min = []
        self.sysdep_labels = []
        self.sys_idx = {}  # keep track what parameters are used for each system
        self.sysdep_range = np.array([self.sysdep_min, self.sysdep_max]).T
        self.sysdep_ndim = len(self.sysdep_max)
        Nsys = len(system_strs)

        for i, s in enumerate(system_strs):
            _, design_max, design_min, labels = load_design(s, pset="main")

            # pick out the system-dependent parameters and add them to lists (these remain empty in current default analysis)
            #self.sysdep_max.append(design_max[0])
            #self.sysdep_min.append(design_min[0])
            #self.sysdep_labels.append(labels[0])

            # the indecies of the parameters used in each system (the full set is used in each system in this analysis)
            self.sys_idx[s] = list(range(len(design_max))) #[i] + list(range(Nsys, Nsys + len(design_max) - 1))

            # if this flag is set to "True", the whole parameter set is selected to be in the "common" list
            if set_common:

                self.common_max = np.array(list(design_max[0:]))
                self.common_min = np.array(list(design_min[0:]))
                self.common_labels = list(labels[0:])
                self.common_ndim = len(self.common_max)
                self.common_range = np.array([self.common_min, self.common_max]).T

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
            self.max = np.concatenate([self.sysdep_max, self.common_max])
            self.min = np.concatenate([self.sysdep_min, self.common_min])
            self.range = np.array([self.min, self.max]).T
            self.labels = self.common_labels
            self.ndim = self.common_ndim

        if hold_parameters:  # flag to hold specific parameters constant in bayesian analyis, unused in current analysis
            self.hold = hold_parameters_set
            # modify the range of the holding parameters to a delta-
            # function like range:
            #       p0-delta < p < p0+delta,
            # this range is not used in MCMC but only for plotting
            # the hold values of p0 will be passed to emulator directly
            for idx, value in self.hold:
                if value < self.min[idx] or self.max[idx] < value:
                    logging.info(
                        "hold value out of range for idx = ", idx, ", value = ", value
                    )
                    logging.info(
                        "range is ["
                        + str(self.min[idx])
                        + " , "
                        + str(self.max[idx])
                        + " ]"
                    )
                    exit(-1)
                delta = (self.max[idx] - self.min[idx]) / 1e3

                self.min[idx] = value - delta
                self.max[idx] = value + delta

        if change_parameters_range: # flag to change range of specific parameters in bayesian analyis, unused in current analysis
            # this range is used only in MCMC and is not passed to the emulator
            self.change_range = change_parameters_range_set
            # modify the ranges of a set of parameters
            for idx, pmin, pmax in self.change_range:
                if pmin < self.min[idx] or self.max[idx] < pmax:
                    logging.info(
                        "update range value out of design range for idx = "
                        + str(idx)
                        + ": pmin = "
                        + str(pmin)
                        + " , pmax = "
                        + str(pmax)
                    )
                    logging.info(
                        "design range is ["
                        + str(self.min[idx])
                        + " , "
                        + str(self.max[idx])
                        + " ]"
                    )
                    exit(-1)

                self.min[idx] = pmin
                self.max[idx] = pmax

        # the n-dimensional volume of the uniform prior
        diff = self.max - self.min
        self.prior_volume = np.prod(diff)

        logging.info("Pre-compute experimental covariance matrix")

        # flag to adjust experimental error "by hand" as specified in configurations.py
        if change_exp_error:
            logging.info(
                "WARNING! Multiplying experimental error by values in change_exp_error_vals: "
            )
            logging.info(change_exp_error_vals)
            for s in system_strs:
                for obs in change_exp_error_vals[s].keys():
                    Yexp[s][obs]["err"] *= change_exp_error_vals[s][obs]

        # read the experimental data from the bayes_dtype array into obseravble "sub-blocks" or slices to match emulator format
        for s in system_strs:
            nobs = 0
            self._slices[s] = []

            for obs in active_obs_list[s]:
                try:
                    if number_of_models_per_run > 1:
                        obsdata = Yexp[s][obs]["mean"][idf]
                    else:
                        obsdata = Yexp[s][obs]["mean"]
                except KeyError:
                    continue

                n = obsdata.size
                self._slices[s].append((obs, slice(nobs, nobs + n)))
                nobs += n

            self._expt_y[s] = np.empty(nobs)
            self._expt_cov[s] = np.empty((nobs, nobs))

            for obs1, slc1 in self._slices[s]:
                is_mult_1 = ("dN" in obs1) or ("dET" in obs1)
                if is_mult_1 and transform_multiplicities:
                    if number_of_models_per_run > 1:
                        self._expt_y[s][slc1] = np.log(Yexp[s][obs1]["mean"][idf] + 1.0)
                        dy1 = Yexp[s][obs1]["err"][idf] / (
                            Yexp[s][obs1]["mean"][idf] + 1.0
                        )
                    else:
                        self._expt_y[s][slc1] = np.log(Yexp[s][obs1]["mean"]) #+ 1.0)
                        dy1 = Yexp[s][obs1]["err"] / (Yexp[s][obs1]["mean"]) #+ 1.0)
                else:
                    if number_of_models_per_run > 1:
                        self._expt_y[s][slc1] = Yexp[s][obs1]["mean"][idf]
                        dy1 = Yexp[s][obs1]["err"][idf]
                    else:
                        self._expt_y[s][slc1] = Yexp[s][obs1]["mean"]
                        dy1 = Yexp[s][obs1]["err"]

                for obs2, slc2 in self._slices[s]:
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
                    self._expt_cov[s][slc1, slc2] = compute_cov(obs1, obs2, dy1, dy2)


    def _predict(self, X, **kwargs):
        """
        Call each system emulator to predict model output at X. (using df model specified by idf in configurations.py)

        """
        # flag for setting specific parameter values to a constant in every emulator prediction
        if hold_parameters:
            for idx, value in self.hold:
                X[:, idx] = value
        return {
            s: Trained_Emulators[s].predict(X[:, ], **kwargs)
            for s in system_strs                    # self.sys_idx[s]
        }

    def log_posterior(self, X, extra_std_prior_scale=0.001):
        """
        Evaluate the posterior at `X`.

        `extra_std_prior_scale` is the scale parameter for the prior
        distribution on the model sys error parameter:

            prior ~ sigma^2 * exp(-sigma/scale)

        """
        X = np.array(X, copy=False, ndmin=2)

        monotonicity = True
        if (X[0,0] > X[0,1]) or (X[0,1] > X[0,2]):
            monotonicity = False

        lp = np.zeros(X.shape[0])

        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf

        # This was taking the last parameter as extra_std
        # extra_std = X[inside, -1]

        nsamples = np.count_nonzero(inside)
        if nsamples > 0:
            # Removing extra_std but leaving framework for it in place
            # pred = self._predict( X[inside], return_cov=True, extra_std=extra_std )
            pred = self._predict(X[inside], return_cov=True)
            for sys in system_strs:
                nobs = self._expt_y[sys].size
                # allocate difference (model - expt) and covariance arrays
                dY = np.empty((nsamples, nobs))
                cov = np.empty((nsamples, nobs, nobs))

                Y_pred, cov_pred = pred[sys]

                # copy predictive mean and covariance into allocated arrays
                for obs1, slc1 in self._slices[sys]:
                    dY[:, slc1] = Y_pred[obs1] - self._expt_y[sys][slc1]
                    for obs2, slc2 in self._slices[sys]:
                        cov[:, slc1, slc2] = cov_pred[obs1, obs2]

                # add expt cov to model cov
                cov += self._expt_cov[sys]

                # compute log likelihood at each point, w/o normalization

                if monotonicity == False:
                    lp[inside] += -np.inf
                else:
                    lp[inside] += list(map(mvn_loglike,dY, cov))
                #lp[inside] += list(map(mvn_loglike, dY, cov))

            # Remove extra_std but leave framework for it in place
            ## add prior for extra_std (model sys error)
            # lp[inside] += 2*np.log(extra_std) - extra_std/extra_std_prior_scale

        return lp

    def log_prior(self, X):
        """
        Evaluate the prior at `X`.

        """
        X = np.array(X, copy=False, ndmin=2)

        # not normalized
        # lp = np.zeros(X.shape[0])

        # normalize the prior
        lp = np.log(np.ones(X.shape[0]) / self.prior_volume)

        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf  # Removes outside the 99% prior range as emulator safety

        lp_specifiedprior = prior.get_pdf(X)

        if lp == -np.inf:
            return lp
        else:
            return lp * lp_specifiedprior

    def log_likelihood(self, X, extra_std_prior_scale=0.001):
        """
        Evaluate the likelihood at `X`.

        """
        X = np.array(X, copy=False, ndmin=2)

        lp = np.zeros(X.shape[0])

        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[
            ~inside
        ] = (
            -np.inf
        )  # Automatically removes outside the 99% region as safety check on emulator

        # Removing extra_std but leaving framework for it in place
        # extra_std = X[inside, -1]

        nsamples = np.count_nonzero(inside)
        if nsamples > 0:
            # Removing extra_std but leaving framework for it in place
            # pred = self._predict( X[inside], return_cov=True, extra_std=extra_std )
            pred = self._predict(X[inside], return_cov=True)
            for sys in system_strs:
                nobs = self._expt_y[sys].size
                # allocate difference (model - expt) and covariance arrays
                dY = np.empty((nsamples, nobs))
                cov = np.empty((nsamples, nobs, nobs))

                Y_pred, cov_pred = pred[sys]

                # copy predictive mean and covariance into allocated arrays
                for obs1, slc1 in self._slices[sys]:
                    dY[:, slc1] = Y_pred[obs1] - self._expt_y[sys][slc1]
                    for obs2, slc2 in self._slices[sys]:
                        cov[:, slc1, slc2] = cov_pred[obs1, obs2]
                print(dY, cov)

                # TEMPORARY FIX EMULATION COVARIANCE TO ZERO
                # cov = np.empty((nsamples, nobs, nobs))

                # TEMPORARY REDUCE EMU COVARIANCE BY FACTOR
                # cov *= 0.1

                # add expt cov to model cov
                cov += self._expt_cov[sys]

                # compute normalized log likelihood at each point
                lp[inside] += list(map(normed_mvn_loglike, dY, cov))

            # Removing extra_std but leaving framework for it in place
            ## add prior for extra_std (model sys error)
            # lp[inside] += 2*np.log(extra_std) - extra_std/extra_std_prior_scale

        return lp

    def random_pos(self, n=1):
        """
        Generate `n` random positions in parameter space.

        """
        return np.random.uniform(self.min, self.max, (n, self.ndim))

    @staticmethod
    def map(f, args):
        """
        Dummy function so that this object can be used as a 'pool' for
        :meth:`emcee.EnsembleSampler`.

        """
        return f(args)

    def run_mcmc(
        self,
        nsteps,
        nburnsteps=None,
        nwalkers=None,
        status=None,
        ntemps=1,
        ncpu=cpu_count(),
    ):
        """
        Run MCMC model calibration.  If the chain already exists, continue from
        the last point, otherwise burn-in and start the chain.

        """
        with self.open("a") as f:
            try:
                dset = f["chain"]
                print(dset)
            except KeyError:
                burn = True
                if nburnsteps is None or nwalkers is None:
                    logging.info("must specify nburnsteps and nwalkers to start chain")
                    return
                dset = f.create_dataset(
                    "chain",
                    dtype="f8",
                    shape=(nwalkers, 0, self.ndim),
                    chunks=(nwalkers, 1, self.ndim),
                    maxshape=(nwalkers, None, self.ndim),
                    compression="lzf",
                )
            else:
                burn = False
                nwalkers = dset.shape[0]

            # choose number of temperatures for PTSampler
            if usePTSampler:
                logging.info("Using PTSampler")
                logging.info("ntemps = " + str(ntemps))
                # ncpu = cpu_count()
                logging.info("{0} CPUs".format(ncpu))
                Tmax = np.inf
                with Pool(ncpu) as pool:
                    sampler = ptemcee.Sampler(
                        nwalkers,
                        self.ndim,
                        self.log_likelihood,
                        self.log_prior,
                        ntemps,
                        Tmax,
                        pool=pool,
                    )
                    logging.info("Running burn-in phase")
                    nburn0 = nburnsteps
                    pos0 = np.random.uniform(
                        self.min, self.max, (ntemps, nwalkers, self.ndim)
                    )
                    start = time.time()
                    sampler.run_mcmc(pos0, nburn0, adapt=True)
                    end = time.time()
                    logging.info("... finished in " + str(end - start) + " sec")
                    logging.info("sampler.chain.shape " + str(sampler.chain.shape))
                    logging.info("betas = " + str(sampler.betas))
                    # get the last step of the chain
                    pos0 = sampler.chain[:, :, -1, :]
                    logging.info("pos0.shape " + str(pos0.shape))
                    sampler.reset()
                    logging.info("Running MCMC chains")
                    niters = 10
                    for iter in range(niters):
                        logging.info("betas = " + str(sampler.betas))
                        logging.info("iteration " + str(iter) + " ...")
                        start = time.time()
                        sampler.run_mcmc(pos0, nsteps // int(niters))
                        end = time.time()
                        logZ, dlogZ = sampler.log_evidence_estimate()
                        logging.info("logZ = " + str(logZ) + " +/- " + str(dlogZ))
                        logging.info("... finished in " + str(end - start) + " sec")

                logging.info("sampler.chain.shape " + str(sampler.chain.shape))
                logging.info("writing chain to file")
                dset.resize(dset.shape[1] + nsteps, 1)
                # save only the zero temperature chain
                dset[:, -nsteps:, :] = sampler.chain[0, :, :, :]

                # save the thermodynamic log evidence
                # logZ, dlogZ = sampler.thermodynamic_integration_log_evidence()
                logZ, dlogZ = sampler.log_evidence_estimate()
                logging.info("logZ = " + str(logZ) + " +/- " + str(dlogZ))
                with open("mcmc/chain-idf-" + str(idf) + "-info.dat", "w") as f:
                    f.write("logZ " + str(logZ) + "\n")
                    f.write("dlogZ " + str(dlogZ))

            else:
                print('------ Running MCMC ------')
                sampler = LoggingEnsembleSampler(
                    nwalkers, self.ndim, self.log_posterior, #pool=self
                )
                if burn:
                    logging.info("no existing chain found, starting initial burn-in")
                    # Run first half of burn-in starting from random positions.
                    nburn0 = nburnsteps // 2
                    sampler.run_mcmc(self.random_pos(nwalkers), nburn0, status=status)
                    logging.info("resampling walker positions")
                    # Reposition walkers to the most likely points in the chain,
                    # then run the second half of burn-in.  This significantly
                    # accelerates burn-in and helps prevent stuck walkers.
                    X0 = sampler.flatchain[
                        np.unique(sampler.flatlnprobability, return_index=True)[1][
                            -nwalkers:
                        ]
                    ]
                    #print(X0)
                    #print(np.shape(X0[0]))
                    #print(X0[0])
                    #print('last point of chain')

                    X0 = sampler.run_mcmc(
                        X0, nburnsteps - nburn0, status=status, store=False    # used to be "storechain"? bug?
                    )#[0]
                    sampler.reset()
                    logging.info("burn-in complete, starting production")
                else:
                    logging.info("restarting from last point of existing chain")
                    X0 = dset[:, -1, :]
                sampler.run_mcmc(X0, nsteps, status=status)
                logging.info("writing chain to file")
                dset.resize(dset.shape[1] + nsteps, 1)
                dset[:, -nsteps:, :] = sampler.chain
                print('last point of chain')
                print(dset[:, -1, :])
                print(len(dset[:, -1, :][0]))
                print(dset[:, -1, :][0])
                sampler.reset()


    def open(self, mode="r"):
        """
        Return a handle to the chain HDF5 file.

        """
        return h5py.File(str(self.path), mode)

    @contextmanager
    def dataset(self, mode="r", name="chain"):
        """
        Context manager for quickly accessing a dataset in the chain HDF5 file.

        >>> with Chain().dataset() as dset:
                # do something with dset object

        """
        with self.open(mode) as f:
            yield f[name]

    def load(self, thin=1):
        """
        Read the chain from file.  If `keys` are given, read only those
        parameters.  Read only every `thin`'th sample from the chain.

        """
        ndim = self.ndim
        indices = slice(None)

        with self.dataset() as d:
            return np.array(d[:, ::thin, indices]).reshape(-1, ndim)

    def load_wo_reshape(self, thin=1):
        """
        Read the chain from file.  If `keys` are given, read only those
        parameters.  Read only every `thin`'th sample from the chain. Don't reshape chain

        """
        # ndim = self.ndim
        indices = slice(None)

        with self.dataset() as d:
            # return np.array(d[:, ::thin, indices]).reshape(-1, ndim)
            return np.array(d[:, ::thin, indices])

    def samples(self, n=1):
        """
        Predict model output at `n` parameter points randomly drawn from the
        chain. (Uses emulator given by idf setting in configurations.py)

        """
        with self.dataset() as d:
            X = np.array(
                [
                    d[i]
                    for i in zip(*[np.random.randint(s, size=n) for s in d.shape[:2]])
                ]
            )

        return self._predict(X)



def credible_interval(samples, ci=0.9):
    """
    Compute the highest-posterior density (HPD) credible interval (default 90%)
    for an array of samples.

    """
    # number of intervals to compute
    nci = int((1 - ci) * samples.size)

    # find highest posterior density (HPD) credible interval
    # i.e. the one with minimum width
    argp = np.argpartition(samples, [nci, samples.size - nci])
    cil = np.sort(samples[argp[:nci]])  # interval lows
    cih = np.sort(samples[argp[-nci:]])  # interval highs
    ihpd = np.argmin(cih - cil)

    return cil[ihpd], cih[ihpd]


def main():
    parser = argparse.ArgumentParser(description="MCMC")

    parser.add_argument("nsteps", type=int, help="number of steps")
    parser.add_argument("--nwalkers", type=int, help="number of walkers")
    parser.add_argument(
        "--ntemps",
        type=int,
        help="number of points in temperature (for PTSampler only)",
    )
    parser.add_argument("--nburnsteps", type=int, help="number of burn-in steps")
    parser.add_argument(
        "--status", type=int, help="number of steps between logging status"
    )

    parser.add_argument("--ncpus", type=int, help="number of cpus to use")

    args = parser.parse_args()

    Chain().run_mcmc(
        nsteps=args.nsteps,
        nwalkers=args.nwalkers,
        nburnsteps=args.nburnsteps,
        status=args.status,
        ntemps=args.ntemps,
        ncpu=args.ncpus,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import logging
import math
import pickle
import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA, KernelPCA

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed, parallel_backend

from calculations_load import trimmed_model_data
from configurations import *

###########################################################
############### Emulator and help functions ###############
###########################################################


class _Covariance:
    """
    Proxy object to extract observable sub-blocks from a covariance array.
    Returned by Emulator.predict().

    """

    def __init__(self, array, slices):
        self.array = array
        self._slices = slices

    def __getitem__(self, key):
        (obs1), (obs2) = key
        return self.array[..., self._slices[obs1], self._slices[obs2]]

class Emulator:
    """
    Multidimensional Gaussian process emulator using principal component
    analysis.

    The model training data are standardized (subtract mean and scale to unit
    variance), then transformed through PCA.  The first `npc` principal
    components (PCs) are emulated by independent Gaussian processes (GPs).  The
    remaining components are neglected, which is equivalent to assuming they
    are standard zero-mean unit-variance GPs.

    This class has become a bit messy but it still does the job.  It would
    probably be better to refactor some of the data transformations /
    preprocessing into modular classes, to be used with an sklearn pipeline.
    The classes would also need to handle transforming uncertainties, which
    could be tricky.

    """

    def __init__(self, system_str, npc, nrestarts=4):
        print("Emulators for system " + system_str)
        print("with viscous correction type {:d}".format(idf))
        print("NPC: " + str(npc))
        print("Nrestart: " + str(nrestarts))

        # list of observables is defined in configurations and in bins_and_cuts
        # here we get their names and sum all the bins to find the total number of observables nobs
        self.nobs = 0
        self.observables = []
        self._slices = {}

        for obs, cent_list in obs_cent_list[system_str].items():
            # for obs, cent_list in calibration_obs_cent_list[system_str].items():
            self.observables.append(obs)
            n = np.array(cent_list).shape[0]
            self._slices[obs] = slice(self.nobs, self.nobs + n)
            self.nobs += n

        print("self.nobs = " + str(self.nobs))
        # read in the model data from file
        print(
            "Loading model calculations from "
            + SystemsInfo[system_str]["main_obs_file"]
        )

        # build a matrix of dimension (num design pts) x (number of observables)

        Y = []
        for ipt, data in enumerate(trimmed_model_data[system_str]):
            row = np.array([])
            for obs in self.observables:

                if number_of_models_per_run > 1:
                    values = np.array(data[idf][obs]["mean"])
                else:
                    values = np.array(data[obs]["mean"])

                row = np.append(row, values)
            Y.append(row)
        Y = np.array(Y)
        print("Y_Obs shape[Ndesign, Nobs] = " + str(Y.shape))

        # Principal Components
        self.npc = npc
        self.scaler = StandardScaler(copy=True)

        # use_KPCA = True
        # try kernel PCA with 3rd degree poly kernel
        # if use_KPCA:
        #    self.pca = KernelPCA(kernel='poly', fit_inverse_transform=True, n_components=npc, degree=2)
        #    Z = self.pca.fit_transform( self.scaler.fit_transform(Y) )
        # else:
        # whiten to ensure uncorrelated outputs with unit variances
        self.pca = PCA(copy=False, whiten=True, svd_solver="full")
        # Standardize observables and transform through PCA.  Use the first
        # `npc` components but save the full PC transformation for later.
        Z = self.pca.fit_transform(self.scaler.fit_transform(Y))[
            :, :npc
        ]  # save all the rows (design points), but keep first npc columns

        # read the parameter design and design ranges
        design, design_max, design_min, labels = prepare_emu_design(system_str)

        # added manually <--- Andi
        #design_max = np.array([2.0, 4.0, 6.0, 1.0, 1.0, 1.0, 1.0, 25.0, 0.5, 0.8, 1.0, 0.3, 1.0, 2.0, 0.2, 0.2, 0.3, 0.15, 0.6, 0.6])
        #design_min = np.array([0.002, 0.004, 0.006, 0.001, 0.001, 0.001, 0.2, 2.0, 0.1, 0.1, 0.001, 0.13, -2.0, -1.0, 0.01, 0.01, 0.12, 0.025, -0.8, 0.1])


        # delete undesirable data
        #if len(delete_design_pts_set) > 0:
            #print(
                #"Warning! Deleting "
                #+ str(len(delete_design_pts_set))
                #+ " points from data"
            #)
        #design = np.delete(design, list(delete_design_pts_set), 0)

        ptp = design_max - design_min
        print("Design shape[Ndesign, Nparams] = " + str(design.shape))
        # Define kernel (covariance function):
        # Gaussian correlation (RBF) plus a noise term.
        # noise term is necessary since model calculations contain statistical noise
        k0 = 1.0 * kernels.RBF(
            length_scale=ptp,
            length_scale_bounds=np.outer(ptp, (2e-1, 1e2)),
            #length_scale_bounds=np.outer(ptp, (4e-1, 1e2)),
            # nu = 3.5
        )
        k1 = kernels.ConstantKernel()
        k2 = kernels.WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1e1))
        #k2 = kernels.WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1e2))

        # kernel = (k0 + k1 + k2) #this includes a constant kernel
        kernel = k0 + k2  # this does not

#        # Fit a GP (optimize the kernel hyperparameters) to each PC.
#        self.gps = []
#        with parallel_backend('threading', n_jobs=):
#            # Your scikit-learn code here
#            for i, z in enumerate(Z.T):
#
#
#                print("Fitting PC #", i)
#                print(ptp)
#                print(design[157])
#                print(z[0:5])
#                self.gps.append(
#                    GPR(
#                        kernel=kernel,
#                        alpha=0.001,
#                        n_restarts_optimizer=nrestarts,
#                        copy_X_train=True,
#                        random_state=32,
#                    ).fit(design, z)
#                )
#
#        for n, (z, gp) in enumerate(zip(Z.T, self.gps)):
#            print("GP " + str(n) + " score: " + str(gp.score(design, z)))
#
#        print("Constructing full linear transformation matrix")


        # Function to fit GPR to a PC
        def fit_gp(i, z, design, kernel, nrestarts):
            print("Fitting PC #", i)
            gp = GPR(
                kernel=kernel,
                alpha=0.001,
                n_restarts_optimizer=nrestarts,
                copy_X_train=True,
                random_state=32,
            ).fit(design, z)
            return gp

        # Parallelized fitting of GPs
        def fit_gps_parallel(Z, design, kernel, nrestarts, n_jobs):
            with parallel_backend('threading', n_jobs=n_jobs):
                gps = Parallel(n_jobs=n_jobs)(delayed(fit_gp)(i, z, design, kernel, nrestarts) for i, z in enumerate(Z.T))
            return gps

        # Usage of the parallelized function and scoring
        self.gps = fit_gps_parallel(Z, design, kernel, nrestarts, n_jobs=-1)  # replace -1 with the number of jobs you desire

        for n, (z, gp) in enumerate(zip(Z.T, self.gps)):
            print(f"GP {n} score: {gp.score(design, z)}")

        print("Constructing full linear transformation matrix")


        # Construct the full linear transformation matrix, which is just the PC
        # matrix with the first axis multiplied by the explained standard
        # deviation of each PC and the second axis multiplied by the
        # standardization scale factor of each observable.

        # if not use_KPCA:
        self._trans_matrix = (
            self.pca.components_
            * np.sqrt(self.pca.explained_variance_[:, np.newaxis])
            * self.scaler.scale_
        )

        # Pre-calculate some arrays for inverse transforming the predictive
        # variance (from PC space to physical space).

        # Assuming the PCs are uncorrelated, the transformation is
        #
        #   cov_ij = sum_k A_ki var_k A_kj
        #
        # where A is the trans matrix and var_k is the variance of the kth PC.
        # https://en.wikipedia.org/wiki/Propagation_of_uncertainty

        print("Computing partial transformation for first npc components")
        # Compute the partial transformation for the first `npc` components
        # that are actually emulated.
        A = self._trans_matrix[:npc]
        self._var_trans = np.einsum("ki,kj->kij", A, A, optimize=False).reshape(
            npc, self.nobs**2
        )

        # Compute the covariance matrix for the remaining neglected PCs
        # (truncation error).  These components always have variance == 1.
        B = self._trans_matrix[npc:]
        self._cov_trunc = np.dot(B.T, B)

        # Add small term to diagonal for numerical stability.
        self._cov_trunc.flat[:: self.nobs + 1] += 1e-4 * self.scaler.var_

    @classmethod
    def build_emu(cls, system, retrain=False, **kwargs):
        emu = cls(system, **kwargs)

        return emu

    def _inverse_transform(self, Z):
        """
        Inverse transform principal components to observables.

        Returns a nested dict of arrays.

        """
        # Z shape (..., npc)
        # Y shape (..., nobs)

        # use_KPCA = True
        # if use_KPCA:
        #    Y = self.pca.inverse_transform(Z)
        #    Y = self.scaler.inverse_transform(Y)
        # else:
        Y = np.dot(Z, self._trans_matrix[: Z.shape[-1]])
        Y += self.scaler.mean_

        """
        return {
            obs: {
                subobs: Y[..., s]
                for subobs, s in slices.items()
            } for obs, slices in self._slices.items()
        }
        """

        return {obs: Y[..., s] for obs, s in self._slices.items()}

    def predict(self, X, return_cov=False, extra_std=0):
        """
        Predict model output at `X`.

        X must be a 2D array-like with shape ``(nsamples, ndim)``.  It is passed
        directly to sklearn :meth:`GaussianProcessRegressor.predict`.

        If `return_cov` is true, return a tuple ``(mean, cov)``, otherwise only
        return the mean.

        The mean is returned as a nested dict of observable arrays, each with
        shape ``(nsamples, n_cent_bins)``.

        The covariance is returned as a proxy object which extracts observable
        sub-blocks using a dict-like interface:

        >>> mean, cov = emulator.predict(X, return_cov=True)

        >>> mean['dN_dy']['pion']
        <mean prediction of pion dN/dy>

        >>> cov[('dN_dy', 'pion'), ('dN_dy', 'pion')]
        <covariance matrix of pion dN/dy>

        >>> cov[('dN_dy', 'pion'), ('mean_pT', 'kaon')]
        <covariance matrix between pion dN/dy and kaon mean pT>

        The shape of the extracted covariance blocks are
        ``(nsamples, n_cent_bins_1, n_cent_bins_2)``.

        NB: the covariance is only computed between observables and centrality
        bins, not between sample points.

        `extra_std` is additional uncertainty which is added to each GP's
        predictive uncertainty, e.g. to account for model systematic error.  It
        may either be a scalar or an array-like of length nsamples.

        """
        if do_transform_design:
            X = transform_design(X)

        gp_mean = [gp.predict(X, return_cov=return_cov) for gp in self.gps]

        if return_cov:
            gp_mean, gp_cov = zip(*gp_mean)

        mean = self._inverse_transform(
            np.concatenate([m[:, np.newaxis] for m in gp_mean], axis=1)
        )

        if return_cov:
            # Build array of the GP predictive variances at each sample point.
            # shape: (nsamples, npc)
            gp_var = np.concatenate(
                [c.diagonal()[:, np.newaxis] for c in gp_cov], axis=1
            )

            # Add extra uncertainty to predictive variance.
            extra_std = np.array(extra_std, copy=False).reshape(-1, 1)
            gp_var += extra_std**2

            # Compute the covariance at each sample point using the
            # pre-calculated arrays (see constructor).
            cov = np.dot(gp_var, self._var_trans).reshape(
                X.shape[0], self.nobs, self.nobs
            )

            # add truncation error to covariance matrix
            #cov += self._cov_trunc

            return mean, _Covariance(cov, self._slices)
        else:
            return mean

    def sample_y(self, X, n_samples=1, random_state=None):
        """
        Sample model output at `X`.

        Returns a nested dict of observable arrays, each with shape
        ``(n_samples_X, n_samples, n_cent_bins)``.

        """
        # Sample the GP for each emulated PC.  The remaining components are
        # assumed to have a standard normal distribution.
        return self._inverse_transform(
            np.concatenate(
                [
                    gp.sample_y(X, n_samples=n_samples, random_state=random_state)[
                        :, :, np.newaxis
                    ]
                    for gp in self.gps
                ]
                + [
                    np.random.standard_normal(
                        (X.shape[0], n_samples, self.pca.n_components_ - self.npc)
                    )
                ],
                axis=2,
            )
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="train emulators for each collision system",
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument("--nrestarts", type=int, help="number of optimizer restarts")

    parser.add_argument(
        "--retrain", action="store_true", help="retrain even if emulator is cached"
    )

    args = parser.parse_args()
    kwargs = vars(args)

    for s in system_strs:
        print("system = " + str(s), ", npc = ", SystemsInfo[s]["npc"])
        emu = Emulator.build_emu(s, npc=SystemsInfo[s]["npc"], **kwargs)

        # EDIT
        print(
            "{} PCs explain {:.5f} of variance".format(
                emu.npc, emu.pca.explained_variance_ratio_[: emu.npc].sum()
            )
        )

        for n, (evr, gp) in enumerate(zip(emu.pca.explained_variance_ratio_, emu.gps)):
            print(
                "GP {}: {:.5f} of variance, LML = {:.5g}, kernel: {}".format(
                    n, evr, gp.log_marginal_likelihood_value_, gp.kernel_
                )
            )

        # dill the emulator to be loaded later
        with open(
            "emulator/emulator-" + s + "-idf-" + str(idf) + "-npc-" + str(SystemsInfo[s]["npc"]) + ".dill", "wb"
        ) as file:
            dill.dump(emu, file)


if __name__ == "__main__":
    main()

Trained_Emulators = {}
for s in system_strs:
    try:
        Trained_Emulators[s] = dill.load(
            open("emulator/emulator-" + s + "-idf-" + str(idf) + "-npc-" + str(SystemsInfo[s]["npc"]) + ".dill", "rb")
        )
    except:
        print("WARNING! Can't load emulator for system " + s)

# contains all the emulators for all df models
Trained_Emulators_all_df = {}
for s in system_strs:
    Trained_Emulators_all_df[s] = {}
    # for idf in [0, 1, 2, 3]:
    for idf in [0]:  # only have one viscous correction model in current analysis
        try:
            Trained_Emulators_all_df[s][idf] = dill.load(
                open("emulator/emulator-" + s + "-idf-" + str(idf) + "-npc-" + str(SystemsInfo[s]["npc"]) + ".dill", "rb")
            )
        except:
            print("WARNING! Can't load emulator for system " + s)

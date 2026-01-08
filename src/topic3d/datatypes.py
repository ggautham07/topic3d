#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data types for topic decomposition (:mod:`topic3d.datatypes`)
==============================================================

This module defines core data structures for handling spatial data in chromatin 
tracing experiments and polymer modeling. It includes:

- `CoordsArrayStack`: a subclass of `np.ndarray` for managing stacks of 3D coordinate traces.
- `PWDistArrayStack`: a subclass of `np.ndarray` for managing stacks of pairwise distance vectors.
  
Both classes support filtering, imputation, statistical aggregation, and 
visualization (via `topic3d.visualization`) of chromatin structure data. These 
are designed to be flexible and memory-efficient, while offering high-level 
methods for preprocessing and analyzing conformational data.

Intended for use in the `topic3d` pipeline after trace extraction and before
or during topic decomposition and modeling.

Authors
-------
ggautham07, jbfiche, marcnol
"""



# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations      # required for type hinting a class method returning an instance of the class
import warnings
from typing import Iterable
from numbers import Number

import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import issymmetric

import matplotlib.pyplot as plt
from . import visualization as vzl


# =============================================================================
# CLASSES
# =============================================================================


class CoordsArrayStack(np.ndarray):
    """
    A stack of arrays of 3D coordinates, used for representing collections of conformations from chromatin tracing
    experiments or polymer simulations.

    Attributes
    ----------
    num_items : int
        Number of individual items in the stack.
    num_bc : int
        Number of barcodes (points) per item.

    Methods
    -------
    center_of_mass()
        Compute the center of mass of each conformation.
    pwds()
        Compute pairwise distances within each conformation.
    efficiency(by="item")
        Compute barcode detection efficiency.
    filter(min_efficiency=60, by="item")
        Filter out low-quality traces.
    impute(method="naive")
        Impute missing positions using a specified method.
    plot_barcode_efficiency(path_to_save="", include_imputed=False)
        Plot barcode-wise detection efficiency.

    Examples
    --------
    >>> stack = CoordsArrayStack(np.random.rand(100, 30, 3))
    >>> coms = stack.center_of_mass()
    >>> filtered = stack.filter(min_efficiency=70)
    >>> imputed = stack.impute(method="naive")
    """

    # Initializing using __new__ instead of __init__ in line with Numpy
    def __new__(cls, arr:np.ndarray):
        """Create a new instance of CoordsArrayStack from a NumPy array."""

        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2 or 3
        assert arr.shape[-1] == 3

        if arr.ndim == 2:
            arr = arr[np.newaxis, :]

        obj = np.asarray(arr).view(cls)

        # add new attributes to the created instance
        obj.num_items = arr.shape[0]
        obj.num_bc = arr.shape[1]
        return obj


    # Since new attributes were added in __new__
    def __array_finalize__(self, obj):
        """Finalize the array view, ensuring metadata is preserved."""

        # reset the attribute from passed original object
        self.num_items = getattr(obj, "num_items", None)
        self.num_bc = getattr(obj, "num_bc", None)


    def __str__(self):
        return f"CoordsArrayStack\n# items: {self.num_items}\n# barcodes: {self.num_bc}"
    

    @staticmethod
    def _set_figsave_flag(fp:str) -> bool:
        """Set the figure save flag based on path string"""

        if len(fp.strip()) > 0:
            return True
        return False


    def center_of_mass(self) -> np.ndarray:
        """
        Compute the center of mass for each trace.

        Returns
        -------
        np.ndarray
            Array of shape (N, 3), where each row is the center of mass of a trace.

        Examples
        --------
        >>> coords.center_of_mass()
        array([[1.2, 2.3, 3.1],
               [0.9, 1.1, 2.0], ...])
        """

        return np.nanmean(self.data, axis=1)    # using self results in another CoordsArrayStack


    def pwds(self) -> PWDistArrayStack:
        """
        Compute the pairwise distances between all barcodes within each item.

        Returns
        -------
        PairwiseArrayStack
            An array of shape (num_items, num_bc*(num_bc-1)//2), containing pairwise distances per trace.

        Examples
        --------
        >>> coords.pwds()
        array([[0.0, 1.4, 3.2, ...], [...], ...])
        """

        pwds = np.empty((self.num_items, int((self.num_bc * (self.num_bc-1)) / 2)))
        for i in range(self.num_items):
            pwds[i] = pdist(self[i])
        pwds = PWDistArrayStack(pwds)

        return pwds


    def efficiency(self, by: str = "barcode") -> np.ndarray:
        """
        Compute the detection efficiency of barcodes or traces.

        Parameters
        ----------
        by : str, optional
            Either "barcode" (default) or "item"/"trace".

        Returns
        -------
        np.ndarray
            1D array of efficiencies as percentages (0-100), rounded to two decimals.

        Examples
        --------
        >>> coords.efficiency(by="trace")
        array([92.3, 100.0, 85.5, ...])
        """

        assert isinstance(by, str) and by in ["barcode", "item", "trace"]

        is_detected = ~(np.isnan(self).any(axis=-1))   # gives (num_items x num_bc) bool matrix of detection
        effs = np.around(np.mean(is_detected, axis=0 if by=="barcode" else 1) * 100, decimals=2)

        return effs


    def filter(self, min_efficiency=60, by="item", return_idxs_only=True):
        """
        Filter out traces with low barcode detection efficiency.

        Parameters
        ----------
        min_efficiency : float
            Minimum percentage of barcodes that must be detected in a trace to keep it.
        by : str, optional
            Either "barcode" or "item"/"trace" (default).

        Returns
        -------
        CoordsArrayStack
            A new stack containing only the traces passing the filter.

        Examples
        --------
        >>> filtered = coords.filter(min_efficiency=0.9)
        >>> filtered.shape
        (87, 30, 3)
        """

        effs = self.efficiency(by=by)
        idxs = np.argwhere(effs >= min_efficiency)
        if return_idxs_only:
            return idxs
        out = CoordsArrayStack(np.squeeze(self[:, idxs]) if by=="barcode" else np.squeeze(self[idxs]))
        return idxs, out


    def impute(self, method="naive") -> CoordsArrayStack:
        """
        Impute missing coordinates in traces using the specified method.

        Parameters
        ----------
        method : str
            Either "naive" (neighbor averaging) or "jost" (Gaussian inference via pairwise distances).

        Returns
        -------
        CoordsArrayStack
            New stack with missing values imputed.

        Examples
        --------
        >>> imputed = coords.impute(method="naive")
        >>> imputed.shape
        (100, 30, 3)
        """

        method = method.lower()
        assert method in ["naive", "jost"]
        if np.all(~np.isnan(self)):
            warnings.warn("No NaNs to impute in the stack")
            return self
        
        stack_imp = np.copy(self)
        
        if method == "naive":
            neighbor_means = (self[:, 2:] + self[:, :-2]) / 2   # compute the mean pos of all potential neighbors
            is_missing = np.isnan(stack_imp[:, 1:-1])   # check missing coordinates
            stack_imp[:, 1:-1][is_missing] = neighbor_means[np.where(is_missing)]   # replace NaNs with mean pos

        elif method == "jost":
            # needs the mean-squared pairwise distance matrix, regularized
            pwds_mean2 = squareform(self.pwds().ensemble(method="mean") ** 2)
            pwds_mean2 = PWDistArrayStack._regz_pw_sq_matrix(pwds_mean2)
            assert pwds_mean2.ndim == 2
            for n in range(self.num_items):
                tr = stack_imp[n][:]
                assert tr.shape[-1] == 3
                assert pwds_mean2.shape[0] == self.num_bc
                # missing positions
                missing_pos_bool = np.isnan(tr[:,0])
                available_pos_bool = np.logical_not(missing_pos_bool)
                s = np.sum(available_pos_bool)
                # reorganizing each axis' coordinates
                x = tr[:,0]
                y = tr[:,1]
                z = tr[:,2]
                x_reorg = np.concatenate((x[available_pos_bool], x[missing_pos_bool]))
                y_reorg = np.concatenate((y[available_pos_bool], y[missing_pos_bool]))
                z_reorg = np.concatenate((z[available_pos_bool], z[missing_pos_bool]))
                # reorganizing matrix as well
                missing_idxs = np.argwhere(missing_pos_bool).T[0]
                available_idxs = np.argwhere(available_pos_bool).T[0]
                new_order = np.concatenate((available_idxs, missing_idxs))
                mean2_reorg = PWDistArrayStack._reorg_pw_sq_matrix(pwds_mean2, new_order)
                # get sigma matrix
                sigma = PWDistArrayStack._cov_from_D2_matrix(mean2_reorg)
                mu = np.zeros((1, self.num_bc-1))
                mu_available = mu[:,:s-1]
                mu_missing = mu[:,s-1:]
                sigma_nz_nz = sigma[0:s-1, 0:s-1]
                sigma_z_z = sigma[s-1:, s-1:]
                sigma_z_nz = sigma[s-1:, 0:s-1]
                sigma_nz_z = sigma[0:s-1, s-1:]
                # infer conditional mean and sigma
                mucondx = mu_missing + \
                    (sigma_z_nz @ (np.linalg.inv(sigma_nz_nz) @ ((x_reorg[1:s] - x_reorg[0]) - mu_available).T)).T
                mucondy = mu_missing + \
                    (sigma_z_nz @ (np.linalg.inv(sigma_nz_nz) @ ((y_reorg[1:s] - y_reorg[0]) - mu_available).T)).T
                mucondz = mu_missing + \
                    (sigma_z_nz @ (np.linalg.inv(sigma_nz_nz) @ ((z_reorg[1:s] - z_reorg[0]) - mu_available).T)).T
                sigcond = sigma_z_z - ((sigma_z_nz @ np.linalg.inv(sigma_nz_nz)) @ sigma_nz_z)
                # infer missing positions given conditional for each axis
                x_missing = x_reorg[0] + mucondx
                y_missing = y_reorg[0] + mucondy
                z_missing = z_reorg[0] + mucondz
                std = np.sqrt(np.diag(sigcond))
                # final inferred coordinates and the error
                inf_coords = np.squeeze(np.concatenate((x_missing.T, y_missing.T, z_missing.T), axis=1))
                std_xyz = np.asarray([std, std, std])
                # replace missing coordinates with the inferred ones
                tr[missing_idxs,:] = inf_coords
                stack_imp[n] = tr[:]

        stack_imp = CoordsArrayStack(stack_imp)

        return (stack_imp, std_xyz) if method == "jost" else stack_imp


    # TODO include plotting for imputed data, currently not possible due to bar plot code structure
    def plot_bc_efficiency(self, path_to_save: str = "", **kwargs):
        """
        Plot the barcode-wise detection efficiency for the dataset.

        This function computes the detection efficiency of each barcode across all traces
        and plots a bar chart. If `include_imputed` is True, it overlays the efficiency
        after naive imputation for comparison.

        Parameters
        ----------
        path_to_save : str, optional
            Path to save the plot. If an empty string is provided (default), the plot is not saved.

        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure.

        Examples
        --------
        >>> fig = coords_stack.plot_barcode_efficiency("figs/efficiency.png", include_imputed=True)
        >>> fig.show()
        """

        assert isinstance(path_to_save, str)

        wflag = self._set_figsave_flag(path_to_save)

        effs = self.efficiency(by="barcode")
        # if include_imputed:
        #     stack_imp = self.impute(method="naive")
        #     effs_imp = stack_imp.efficiency(by="barcode")

        fig = vzl.plot_bar(data={"barcode_id": np.arange(self.num_bc),"efficiency": effs},
                           x="barcode_id", y="efficiency",
                           xlabel="Barcode #", ylabel="Efficiency",
                           **kwargs)

        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig



class PWDistArrayStack(np.ndarray):
    """
    A stack of pairwise distance vectors (1D) with utilities for matrix operations and visualization.
    
    Used to represent and process collections of intra-trace pairwise distances in chromatin tracing or
    polymer modeling.

    Parameters
    ----------
    arr : np.ndarray
        A 1D or 2D array where each vector represents the pairwise distances (flattened) of a trace.
        Must correspond to the binomial coefficient (n choose 2) for some integer n ≥ 2.

    Attributes
    ----------
    num_items : int
        Number of traces (1 if input was 1D).
    num_bc : int
        Number of barcodes (points) per trace, inferred from vector length.

    Methods
    -------
    ensemble(method='mean', regularize=False)
        Perform ensemble operation across the stack (mean, median, etc.).
    proximity(threshold, hardness=1, keep_NaNs=False)
        Compute proximity from distances using a threshold and hardness.
    filter(min_efficiency=0.1)
        Filter items based on missing data.
    efficiency(by='pixel')
        Compute detection efficiency by pixel or by item.
    plot_pwds_avg(path_to_save, method='mean', form='triangle', cmap='Reds_r')
        Plot ensemble-average pairwise distance matrix.
    plot_proximity_avg(path_to_save, threshold, hardness=1, form='triangle', cmap='Reds')
        Plot ensemble-average proximity matrix.
    """

    # Initializing using __new__ instead of __init__ in line with Numpy
    def __new__(cls, arr:np.ndarray):
        """ Create a new PairwiseArrayStack instance from a distance matrix or vector."""
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1 or 2
    
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        s = arr.shape
        d = int(np.ceil(np.sqrt(s[-1] * 2)))
        if d * (d - 1) != s[-1] * 2:
            raise ValueError("Incompatible vector size. Must be a binomial coefficient" \
            "n choose 2 for some integer n >= 2.")
        
        obj = np.asarray(arr).view(cls)

        # add new attributes to the created instance
        obj.num_items = arr.shape[0]
        obj.num_bc = d

        return obj
    

    def __str__(self):
        return f"PairwiseArrayStack\n# items: {self.num_items}\n# barcodes: {self.num_bc}"


    # Since new attributes were added in __new__
    def __array_finalize__(self, obj):
        # reset the attribute from passed original object
        self.num_items = getattr(obj, "num_items", None)
        self.num_bc = getattr(obj, "num_bc", None)


    @staticmethod
    def _set_figsave_flag(fp:str) -> bool:
        """Set the flag to save a Figure based on path string length"""

        if len(fp.strip()) > 0:
            return True
        return False


    # TODO class method to check symmetric matrices in the stack
    @staticmethod
    def _make_pw_sq_matrix_symm(matrix:np.ndarray) -> np.ndarray:
        """
        Ensure the matrix is symmetric by copying upper triangle to lower (or averaging).

        Parameters
        ----------
        matrix : np.ndarray
            Square matrix.

        Returns
        -------
        np.ndarray
            Symmetric matrix.
        """

        assert matrix.ndim == 2

        if issymmetric(matrix):
            return matrix
        return matrix + matrix.T - np.diag(np.diag(matrix)) if (np.tril(matrix, k=-1) == 0).all() \
            else (matrix + matrix.T) / 2


    @staticmethod
    def _cov_from_D2_matrix(matrix:np.ndarray) -> np.ndarray:
        """
        Compute covariance matrix from mean-squared distance matrix assuming a Gaussian distribution.

        Parameters
        ----------
        matrix : np.ndarray
            Symmetric distance² matrix.

        Returns
        -------
        np.ndarray
            Covariance matrix.
        """

        assert matrix.ndim == 2
        assert issymmetric(matrix)
        num_bc = matrix.shape[0]

        cov = np.zeros((num_bc-1, num_bc-1))
        np.fill_diagonal(cov, matrix[0,1:])
        for i in range(num_bc-1):
            for j in range(i+1, num_bc-1):
                cov[i,j] = (cov[i,i] + cov[j,j] - matrix[i+1, j+1]) / 2
        
        cov = PWDistArrayStack._make_pw_sq_matrix_symm(cov)
        return cov


    @staticmethod
    def _D2_matrix_from_cov(cov:np.ndarray) -> np.ndarray:
        """
        Compute mean-squared distance matrix from covariance matrix assuming a Gaussian distribution.

        Parameters
        ----------
        cov : np.ndarray
            Covariance matrix.

        Returns
        -------
        np.ndarray
            Symmetric mean-squared distance matrix.
        """

        assert cov.ndim == 2
        N = cov.shape[0] + 1

        mean2 = np.zeros((N, N))    
        mean2[0,1:] = np.diag(cov)
        mean2 = PWDistArrayStack._make_pw_sq_matrix_symm(mean2)
        for i in range(1, N):
            for j in range(i+1, N):
                mean2[i,j] = -2 * cov[i-1, j-1] + cov[i-1, i-1] + cov[j-1, j-1]
                mean2[j,i] = mean2[i,j]
        np.fill_diagonal(mean2, 0)

        return mean2


    @staticmethod
    def _regz_pw_sq_matrix(matrix:np.ndarray, hack=False) -> np.ndarray:
        """
        Make matrix positive definite by adjusting negative eigenvalues (i.e., regularize).

        Parameters
        ----------
        matrix : np.ndarray
            Square matrix.
        hack : bool
            If True, forces smallest positive eigenvalue to small constant.

        Returns
        -------
        np.ndarray
            Regularized matrix.
        """

        assert matrix.ndim == 2
        assert issymmetric(matrix)

        eigenvals, eigenvecs = np.linalg.eig(matrix)
        if np.all(eigenvals > 0) and hack:
            eigenvals[np.argmin(eigenvals)] = 1e-06
        else:
            eigenvals[eigenvals < 0] = 1e-06
        matrix_reg = eigenvecs @ np.diag(eigenvals) @ np.linalg.inv(eigenvecs)
        # TODO without the following step, will the matrix be symmetric?
        matrix_reg = PWDistArrayStack._make_pw_sq_matrix_symm(matrix_reg)
        return matrix_reg


    # TODO finish the vectorised implementation for a 3D stack (now works for a 2D matrix)
    @staticmethod
    def _reorg_pw_sq_matrix(matrix:np.ndarray, idxs) -> np.ndarray:      
        """
        Reorder a matrix using a new set of indices.

        Parameters
        ----------
        matrix : np.ndarray
            Pairwise matrix to reorder.
        idxs : array-like
            New ordering of indices.

        Returns
        -------
        np.ndarray
            Reordered matrix.
        """

        assert matrix.ndim == 2
        # assert issymmetric(matrix)
        assert len(np.unique(idxs)) == len(idxs)
        assert np.all(np.logical_and(idxs >= 0, idxs < matrix.shape[0]))

        return matrix[:,idxs][idxs]


    def efficiency(self, by: str = "pixel") -> np.ndarray:
        """
        Compute detection efficiency by pixel or item.

        Parameters
        ----------
        by : str
            "pixel" for per-pixel efficiency through the stack, "item" for whole matrices.

        Returns
        -------
        np.ndarray
            Efficiencies in percent (0-100).
        """

        assert isinstance(by, str) and by in ["pixel", "item", "matrix"]

        is_detected = ~np.isnan(self)
        effs = np.around(np.mean(is_detected, axis=0 if by=="pixel" else 1) * 100, decimals=2)

        return effs


    def ensemble(self, method="mean", regularize=False) -> np.ndarray:
        """
        Perform an ensemble operation (mean, median, etc.) across the stack.

        Parameters
        ----------
        method : str
            One of 'mean', 'median', 'std', 'sum', 'max', 'min'.
        regularize : bool
            Whether to regularize the resulting matrix.

        Returns
        -------
        np.ndarray
            Ensemble average vector.
        """

        methods = {"mean": np.nanmean, "median": np.nanmedian, "std": np.nanstd,
                   "sum": np.nansum, "max": np.nanmax, "min": np.nanmin}
        assert method.strip().lower() in methods.keys()

        if self.num_items == 1:
            warnings.warn("Note: ensemble operation being performed on an array with only one item")
            return squareform(self._regz_pw_sq_matrix(squareform(self[0]))) if regularize \
                else self[0][:]
        
        ensemble_matrix = methods[method](self, axis=0)
        return squareform(self._regz_pw_sq_matrix(squareform(ensemble_matrix))) if regularize \
            else ensemble_matrix


    # NOTE JB's original function was different: data = 1 / (1 + (data / contact_threshold)**10)
    def proximity(self, threshold:Number, hardness=1, keep_NaNs=False) -> np.ndarray:
        """
        Convert distances into proximity values using a sigmoid function or hard threshold.

        Hardness describes sigmoid slope, 0 < hardness <= 1, where 1 means hard.
        Ideal value for softness visually appears to be 0.4. Values up to 0.2 may be too soft.

        Parameters
        ----------
        threshold : float
            Distance cutoff.
        hardness : float
            Controls sigmoid slope. 1 is hard threshold; lower is softer.
        keep_NaNs : bool
            Whether to retain NaNs.

        Returns
        -------
        np.ndarray
            Proximity values in [0, 1].
        """

        assert isinstance(threshold, Number)
        assert not ((self == 0) | (self == 1)).all()
        assert threshold <= np.nanmax(self)
        assert isinstance(hardness, Number) and 0 < hardness <= 1

        if hardness < 0.2:
            warnings.warn(f"Input hardness value of {hardness} may be too soft")
        
        mode = "hard" if hardness == 1 else "soft"

        if mode == "hard":
            if keep_NaNs:
                return np.where(np.isnan(self), np.nan, self <= threshold)
            return (self <= threshold).astype(int)

        out = 1 / (1 + np.exp(hardness * (self - threshold)))
        if keep_NaNs:
            return out
        return np.nan_to_num(out, nan=0, copy=True)


    def filter(self, min_efficiency: Number = 0.6, return_idxs_only=True):
        """
        Remove matrices with insufficient barcode detection.

        Parameters
        ----------
        min_efficiency : float
            Minimum proportion of non-NaN values to retain a matrix.

        Returns
        -------
        np.ndarray or PWDistArrayStack
            Filtered indices or indices and subset of the stack.
        """

        proportion_bins_not_missing = np.mean(~np.isnan(self), axis=1)
        idxs = np.argwhere(proportion_bins_not_missing > min_efficiency)
        
        if return_idxs_only:
            return idxs    
        out = PWDistArrayStack(np.squeeze(self[idxs, :]))
        return idxs, out


    # TODO for normalization of pairwise distances from JB's code, available in format_FOF_data.py
    def normalize(self, ref: np.ndarray, method: str = "MMSF") -> PWDistArrayStack:
        """
        Normalize the stack by different methods while comparing to a reference dataset

        MMSF: median-median scaling factor

        JB's method based on R(s) yet to be implemented here
        """
    
        method = method.strip().upper()

        if method == "MMSF":
            scaling_factor = np.nanmedian(ref) / np.nanmedian(self)

            return self[:] * scaling_factor


    def split_random(self, proportion: Number = 0.5, return_idxs_only=True):
        """
        Split the dataset two items randomly based on a proportion (default 0.5, i.e., 50%)

        Parameters
        --------
        proportion : Number
            split proportion 0 < proportion < 1
        """

        assert 0 < proportion < 1
        assert isinstance(return_idxs_only, bool)

        idxs_all = np.arange(self.num_items, dtype=int)
        idxs_0 = np.random.choice(idxs_all, size=int(proportion * len(idxs_all)), replace=False)
        idxs_1 = np.delete(idxs_all, idxs_0)
        if len(idxs_0) == 0 or len(idxs_1) == 0:
            warnings.warn("Number of indices was zero after the split, consider changing the proportion")
        idxs_out = (idxs_0, idxs_1)
        
        if return_idxs_only:
            return idxs_out
        out = (PWDistArrayStack(np.squeeze(self[idxs_0, :])), PWDistArrayStack(np.squeeze(self[idxs_1, :])))
        return idxs_out, out


    def plot_pixel_efficiency(self, path_to_save: str = "", form: str = "triangle"):
        """
        Plot the pixel-wise detection efficiency across the pairwise distance matrix stack.

        This function computes the efficiency of detection at each pixel (i.e., matrix position)
        across all items in the stack, and visualizes the result as a heatmap.

        Parameters
        ----------
        path_to_save : str, optional
            File path to save the figure. If empty (default), the plot is not saved.
        form : str, optional
            Type of plot to generate. Must be either "triangle" or "square". Default is "triangle".

        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure.

        Examples
        --------
        >>> fig = pwd_stack.plot_pixel_efficiency("figs/eff_map.svg", form="square")
        >>> fig.show()
        """

        assert isinstance(path_to_save, str)
        assert form in ["triangle", "square"]

        wflag = self._set_figsave_flag(path_to_save)

        map_form_fn = {"triangle": vzl.plot_matrices_triangle, "square": vzl.plot_matrices_square,}

        effs = self.efficiency(by="pixel")
        fig = map_form_fn[form](items=[effs],
                                cmap="magma",
                                cbar_label="efficiency %")

        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig
        

    def plot_pwds_avg(self, method: str = "mean",
                      path_to_save: str = "",
                      form: str = "triangle", cmap: str = "Reds_r"):
        """
        Plot the ensemble-averaged pairwise distance matrix across the stack.

        Parameters
        ----------
        path_to_save : str
            Path (with file extension) where the figure will be saved.
        method : str, default="mean"
            Aggregation method to compute ensemble average. Supported values: "mean", "median".
        form : str, default="triangle"
            Format of the plot: "triangle" for lower triangle heatmap, or "square" for full matrix heatmap.
        cmap : str, default="Reds_r"
            Matplotlib colormap to use for the plot.

        Returns
        -------
        Figure
            Saves the plot to the specified file path.

        Examples
        --------
        >>> stack.plot_pwds_avg("figs/avg_pwds.svg", method="mean", form="square", cmap="viridis")
        """

        assert isinstance(path_to_save, str)
        assert form in ["triangle", "square"]
        assert isinstance(method, str) and method in ["mean", "median"]

        wflag = self._set_figsave_flag(path_to_save)

        map_form_fn = {"triangle": vzl.plot_matrices_triangle, "square": vzl.plot_matrices_square,}

        matrix = self.ensemble(method=method)
        fig = map_form_fn[form](items=[matrix],
                                titles=[f"$n = ${self.num_items:,}"],
                                cmap=cmap,
                                cbar_label=f"{method} distance")
        
        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig
    

    def plot_pwds_random_examples(self, num_items: int = 9, random_seed: int = None,
                                  path_to_save: str = "", form: str = "triangle", cmap="Reds_r", include_cbar=True):
        """
        Plot a set of randomly-picked PWD maps from the dataset.

        Parameters
        --------
        num_items : Number
            number of items to plot
        random_seed : int or None
            random seed for reproducibility
        path_to_save : str
            where to save the plot, if unspecified, returns only the Figure

        Returns
        --------
        Figure
        """

        assert isinstance(num_items, int)
        assert isinstance(path_to_save, str)
        assert form in ["triangle", "square"]

        wflag = self._set_figsave_flag(path_to_save)

        map_form_fn = {"triangle": vzl.plot_matrices_triangle, "square": vzl.plot_matrices_square,}

        random_gen = np.random.default_rng(seed=random_seed)
        idxs = random_gen.integers(self.num_items, size=num_items)
        
        fig = map_form_fn[form](items=self[idxs, ...],
                                titles=[f"Index  {i:,}" for i in idxs],
                                cmap=cmap,
                                cbar_label="distance",
                                include_cbar=include_cbar)
        
        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig


    def plot_proximity_avg(self, threshold: Number, hardness: Number = 1, keep_NaNs: bool = False,
                           path_to_save: str = "", form: str = "triangle", cmap: str = "Reds"):
        """
        Plot the ensemble-averaged proximity matrix based on a distance threshold.

        Proximity values are computed using a sigmoidal transformation or hard thresholding, then averaged
        across all items in the stack.

        Parameters
        ----------
        path_to_save : str
            Path (with file extension) where the figure will be saved.
        threshold : float
            Distance threshold used to compute proximity.
        hardness : float, default=1
            Controls the slope of the sigmoid used for soft thresholding. 1 = hard threshold.
        form : str, default="triangle"
            Format of the plot: "triangle" for triangular heatmap, or "square" for full matrix heatmap.
        cmap : str, default="Reds"
            Matplotlib colormap to use for the plot.

        Returns
        -------
        None
            Saves the plot to the specified file path.

        Examples
        --------
        >>> stack.plot_proximity_avg("figs/avg_proximity.svg", threshold=0.4, hardness=0.5, form="triangle", cmap="coolwarm")
        """
        
        assert isinstance(path_to_save, str)
        assert form in ["triangle", "square"]

        wflag = self._set_figsave_flag(path_to_save)

        map_form_fn = {"triangle": vzl.plot_matrices_triangle, "square": vzl.plot_matrices_square,}

        prox_stack = self.proximity(threshold=threshold, hardness=hardness, keep_NaNs=keep_NaNs)
        prox_mean = np.nanmean(prox_stack, axis=0)
        fig = map_form_fn[form](items=[prox_mean],
                                titles=[f"$n = ${self.num_items:,}  |  $k = ${threshold} nm |  $h = ${hardness:.1f}"],
                                cmap=cmap,
                                cbar_label="mean proximity")

        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig
    
    
    def plot_proximity_avg_comparisons(self, thresholds:Iterable, hardnesses:Iterable,
                                        path_to_save: str = "", form: str = "triangle", cmap="Reds"):
        """
        Plot comparisons of proximity average maps computed using different parameters for threshold and hardness.
        """

        assert isinstance(thresholds, Iterable)
        assert isinstance(hardnesses, Iterable)
        assert len(thresholds) == len(hardnesses)
        assert isinstance(path_to_save, str)
        assert form in ["triangle", "square"]

        wflag = self._set_figsave_flag(path_to_save)

        map_form_fn = {"triangle": vzl.plot_matrices_triangle, "square": vzl.plot_matrices_square,}

        prox_avgs = [np.mean(self.proximity(threshold=thr, hardness=hrd),axis=0) \
                     for thr, hrd in zip(thresholds, hardnesses)]
        
        fig = map_form_fn[form](items=prox_avgs,
                                titles=[f"$n = ${self.num_items:,}  |  $k = ${thr}  |  $h = ${hrd:.1f}" \
                                        for thr, hrd in zip(thresholds, hardnesses)],
                                cmap=cmap,
                                cbar_label="mean proximity")
        
        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig
    

    def plot_proximity_random_examples(self, threshold: Number, hardness: Number,
                                       num_items: int = 9, random_seed: int = None,
                                       path_to_save: str = "", form: str = "triangle", cmap="Reds", include_cbar=True):
        """
        Plot a set of randomly-picked proximity maps from the dataset.

        Parameters
        --------
        num_items : Number
            number of items to plot
        random_seed : int or None
            random seed for reproducibility
        path_to_save : str
            where to save the plot, if unspecified, returns only the Figure

        Returns
        --------
        Figure
        """

        assert isinstance(num_items, int)
        assert isinstance(path_to_save, str)
        assert form in ["triangle", "square"]

        wflag = self._set_figsave_flag(path_to_save)

        map_form_fn = {"triangle": vzl.plot_matrices_triangle, "square": vzl.plot_matrices_square,}

        random_gen = np.random.default_rng(seed=random_seed)
        idxs = random_gen.integers(self.num_items, size=num_items)
        
        fig = map_form_fn[form](items=self.proximity(threshold, hardness)[idxs, ...],
                                titles=[f"Idx {i:,}  |  $k = ${threshold} nm  |  $h = ${hardness:.1f}" for i in idxs],
                                cmap=cmap,
                                cbar_label="proximity",
                                include_cbar=include_cbar)
        
        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig


    def plot_mean_proximity_proportion(self, thresholds:list, hardness=1, path_to_save: str = ""):
        """
        Plot the mean proportion of pixels in proximity (within each item) across the dataset given a range of 
        proximity thresholds.

        For each threshold value, this function computes a binary proximity map (or using a soft sigmoid),
        and averages the proportion of proximities across the entire stack. The result is plotted as a curve.

        Parameters
        ----------
        thresholds : list of float or single float
            One or more distance thresholds at which proximity will be computed.
        hardness : float, optional
            Controls the sharpness of the proximity thresholding. 
            A value of 1 results in hard cutoff, lower values apply a softer sigmoid (default: 1).
        path_to_save : str
            File path to save the plot. If empty, the plot will not be saved.

        Returns
        -------
        matplotlib.figure.Figure
            The generated matplotlib figure showing proximity proportion vs. threshold.

        Examples
        --------
        >>> fig = pwd_stack.plot_mean_proximity_proportion("figs/prox_prop.svg", thresholds=[0.2, 0.3, 0.4], hardness=0.5)
        >>> fig.show()
        """

        if isinstance(thresholds, Number):
            thresholds = np.asarray([thresholds])
        else:
            assert isinstance(thresholds, Iterable)

        wflag = self._set_figsave_flag(path_to_save)

        proportion_means = np.empty(len(thresholds))
        for i, thr in enumerate(thresholds):
            stack = self.proximity(threshold=thr, hardness=hardness, keep_NaNs=False)
            if hardness != 1:
                stack = np.where(stack >= 0.75, 1, 0)
            proportion_means[i] = np.mean(stack)

        fig = vzl.plot_line(x=thresholds, y=proportion_means,
                            xlabel="Proximity threshold", ylabel="Mean proximity proportion",
                            linestyle="-", linewidth=4,
                            marker="o", markersize=16)

        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig


    # TODO complete after normalization procedure is implemented
    # def plot_proximity_stats(self, threshold: Number):
    #     """
    #     Plot the contact histogram
    #     @param pwd_raw: (numpy array) contains all the raw pairwise distance maps flatten as 1D vector
    #     @param pwd_normalized: (numpy array) contains all the raw pairwise distance maps flatten as 1D vector
    #     @param thresh: (int) threshold for the contact in nm
    #     @param saving_folder: (str) folder where to save the plot
    #     @return:
    #     """

    #     raw_contacts = np.where(np.isnan(pwd_raw), np.nan, pwd_raw < thresh)
    #     raw_contacts = np.nansum(raw_contacts, axis=1)
    #     normalized_contacts = np.where(np.isnan(pwd_normalized), np.nan, pwd_normalized < thresh)
    #     normalized_contacts = np.nansum(normalized_contacts, axis=1)

    #     plt.figure(figsize=(8, 6))
    #     sns.histplot(raw_contacts, bins=50, kde=True, alpha=0.5)
    #     sns.histplot(normalized_contacts, bins=50, kde=True, alpha=0.5)
    #     plt.xlabel("Contact Values")
    #     plt.ylabel("Frequency")
    #     plt.title("Histogram of Contacts")
    #     plt.legend(["raw", "normalized"])
    #     plt.savefig(path.join(saving_folder, f'contact_stat_thresh_{int(thresh)}nm.png'), dpi=150
    #                 , bbox_inches='tight')
    #     plt.close()
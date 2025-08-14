#!/usr/bin/env python
"""
Data types for topic decomposition (:mod:`topic3d.datatypes`)
=====================================================

.. authors:: gsgautham98, jbfiche, marcnol

Class reference
-----------------
"""


# =============================================================================
# IMPORTS
# =============================================================================

from numbers import Number
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import issymmetric
import warnings
import visualization



# =============================================================================
# CLASSES
# =============================================================================


class CoordsArrayStack(np.ndarray):
    """ A stack of arrays of 3D coordinates and associated metadata. 
    This class is primarily designed to handle collections of coordinate traces from chromatin tracing experiments 
    or coarse-grained polymer simulations.

    Parameters
    ----------
    arr : np.ndarray
        A 2D or 3D NumPy array where the last dimension must be 3 (x, y, z coordinates). 
        Shape should be (N, M, 3) or (M, 3), where N is the number of traces and M is the number of barcodes.

    Attributes
    ----------
    num_items : int
        Number of traces in the stack.
    num_bc : int
        Number of barcodes (or points) per trace.

    Methods
    -------
    center_of_mass()
        Compute the center of mass (CoM) for each trace.
    pwds()
        Compute the pairwise distances within each trace.
    efficiency(by='barcode')
        Compute detection efficiency by barcode or by trace.
    filter(min_efficiency=0.1)
        Filter traces based on a minimum barcode detection efficiency.
    impute(method='naive')
        Impute missing coordinates using a naive or advanced probabilistic method.
    """ 

    # Initializing using __new__ instead of __init__ in line with Numpy
    def __new__(cls, arr:np.ndarray):
        """ Create a new instance of CoordsArrayStack from a NumPy array.

        Parameters
        ----------
        arr : np.ndarray
            Input array of shape (M, 3) or (N, M, 3), where M = number of barcodes and N = number of items/traces.

        Returns
        -------
        CoordsArrayStack
            A view of the array as a CoordsArrayStack instance.
        """

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


    def center_of_mass(self) -> np.ndarray:
        """ Compute the center of mass for each trace.

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

        return np.nanmean(self, axis=1)


    def pwds(self):
        """ Compute the pairwise distances between all barcodes within each trace.

        Returns
        -------
        PairwiseArrayStack
            An array of shape (N, M*(M-1)//2), containing pairwise distances per trace.

        Examples
        --------
        >>> coords.pwds()
        array([[0.0, 1.4, 3.2, ...], [...], ...])
        """

        pwds = np.empty((self.num_items, int((self.num_bc * (self.num_bc - 1)) / 2), ))
        for i in range(self.num_items):
            pwds[i] = pdist(self[i])
        
        pwds = PairwiseArrayStack(pwds)
        return pwds


    def efficiency(self, by="barcode") -> np.ndarray:
        """ Compute the detection efficiency of barcodes or traces.

        Parameters
        ----------
        by : str, optional
            Either "barcode" or "item"/"trace". Default is "barcode".

        Returns
        -------
        np.ndarray
            1D array of efficiencies as percentages (0–100), rounded to two decimals.

        Examples
        --------
        >>> coords.efficiency(by="trace")
        array([92.3, 100.0, 85.5, ...])
        """

        assert isinstance(by, str) and by in ["barcode", "item", "trace"]

        is_detected = ~(np.isnan(self).any(axis=-1))   # gives (num_items x num_bc) bool matrix of detection
        effs = np.around(np.mean(is_detected, axis=0 if by=="barcode" else 1) * 100, decimals=2)

        return effs
    

    def filter(self, min_efficiency=0.1):
        """ Filter out traces with low barcode detection efficiency.

        Parameters
        ----------
        min_efficiency : float
            Minimum fraction of barcodes that must be detected in a trace to keep it.

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

        num_bc_detected_by_item = np.sum(~np.isnan(self).any(axis=-1), axis=1)
        idxs = np.argwhere(num_bc_detected_by_item >= int(min_efficiency * self.num_bc))

        return CoordsArrayStack(np.squeeze(self[idxs]))


    def impute(self, method="naive"):
        """ Impute missing coordinates in traces using the specified method.

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

        if ~np.all(np.isnan(self)):
            warnings.warn("No NaNs to impute in the stack")
            return self
        
        if method == "naive":
            neighbor_means = (self[:, 2:] + self[:, :-2]) / 2   # compute the mean pos of all potential neighbors
            stack_imputed = self[:]
            is_missing = np.isnan(stack_imputed[:, 1:-1])   # check missing coordinates
            stack_imputed[:, 1:-1][is_missing] = neighbor_means[np.where(is_missing)]   # replace NaNs with mean pos

        elif method == "jost":
            stack_imputed = self[:]
            for n in range(self.num_items):
                tr = stack_imputed[n][:]
                mean2 = pdist(tr) ** 2
                assert mean2.ndim == 2
                assert tr.shape[-1] == 3
                assert mean2.shape[0] == self.num_bc
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
                mean2_reorg = PairwiseArrayStack._reorganize_pairwise_square_matrix(mean2, new_order)
                # get sigma matrix
                sigma = PairwiseArrayStack._cov_from_D2_matrix(mean2_reorg)
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
                std_xyz = np.array([std, std, std])
                # replace missing coordinates with the inferred ones
                tr[missing_idxs,:] = inf_coords
                stack_imputed[n] = tr[:]

        stack_imputed = CoordsArrayStack(stack_imputed)
        return stack_imputed


########################################################################



class PairwiseArrayStack(np.ndarray):
    """ A stack of pairwise distance vectors (1D) with utilities for covariance inference,
    matrix regularization, and visualization. Used to represent and process collections of intra-trace 
    pairwise distances in chromatin tracing or polymer modeling.

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

    Static Methods
    --------------
    _make_pairwise_square_matrix_symmetric(matrix)
        Symmetrize a square matrix.
    _cov_from_D2_matrix(matrix)
        Infer covariance matrix from mean-squared distances.
    _D2_matrix_from_cov(cov)
        Compute mean-squared distances from covariance.
    _normalize_minmax(arr)
        Normalize values in array to [0, 1].
    _rescale_minmax(arr, ref)
        Rescale arr to the range of reference array ref.
    _regularize_pairwise_square_matrix(matrix, hack=False)
        Make a matrix positive definite by eigenvalue correction.
    _reorganize_pairwise_square_matrix(matrix, idxs)
        Reorder rows/columns of matrix based on index list.
    """

    # Initializing using __new__ instead of __init__ in line with Numpy
    def __new__(cls, arr:np.ndarray):
        """ Create a new PairwiseArrayStack instance from a distance matrix or vector.

        Parameters
        ----------
        arr : np.ndarray
            Input array of shape (N, K) or (K,), where K = n*(n-1)//2 for some n.

        Returns
        -------
        PairwiseArrayStack
            The input viewed as a PairwiseArrayStack with inferred metadata.
        """
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


    # Since new attributes were added in __new__
    def __array_finalize__(self, obj):
        # reset the attribute from passed original object
        self.num_items = getattr(obj, "num_items", None)
        self.num_bc = getattr(obj, "num_bc", None)


    @staticmethod
    def _make_pairwise_square_matrix_symmetric(matrix:np.ndarray) -> np.ndarray:
        """ Ensure the matrix is symmetric by copying upper triangle to lower (or averaging).

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
        """ Compute covariance matrix from mean-squared distance matrix assuming a Gaussian distribution.

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
        N = matrix.shape[0]

        cov = np.zeros((N-1, N-1))
        np.fill_diagonal(cov, matrix[0,1:])
        for i in range(N-1):
            for j in range(i+1, N-1):
                cov[i,j] = (cov[i,i] + cov[j,j] - matrix[i+1, j+1]) / 2
        
        cov = PairwiseArrayStack._make_pairwise_square_matrix_symmetric(cov)
        return cov
    

    @staticmethod
    def _D2_matrix_from_cov(cov:np.ndarray) -> np.ndarray:
        """ Compute mean-squared distance matrix from covariance matrix assuming a Gaussian distribution.

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
        mean2 = PairwiseArrayStack._make_pairwise_square_matrix_symmetric(mean2)
        for i in range(1, N):
            for j in range(i+1, N):
                mean2[i,j] = -2 * cov[i-1, j-1] + cov[i-1, i-1] + cov[j-1, j-1]
                mean2[j,i] = mean2[i,j]
        np.fill_diagonal(mean2, 0)

        return mean2


    # TODO rewrite as method applied to both self and numpy.ndarrays
    @staticmethod
    def _normalize_minmax(arr:np.ndarray) -> np.ndarray:
        """ Normalize array values to [0, 1] using min-max normalization.

        Parameters
        ----------
        arr : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Normalized array.
        """

        arr_min = np.min(arr)
        arr_rescaled = (arr - arr_min) / (np.max(arr) - arr_min)
        
        return arr_rescaled


    # TODO rewrite as method applied to both self and numpy.ndarrays
    @staticmethod
    def _rescale_minmax(arr:np.ndarray, ref:np.ndarray) -> np.ndarray:
        """ Rescale arr to the range of ref using min-max normalization.

        Parameters
        ----------
        arr : np.ndarray
            Array to rescale.
        ref : np.ndarray
            Reference array to match range.

        Returns
        -------
        np.ndarray
            Rescaled array.
        """
        
        assert np.logical_not(np.isnan(arr).all())

        ref = ref[np.logical_not(np.isnan(ref))]
        new_min = np.min(ref)
        new_max = np.max(ref)

        arr_rescaled = (PairwiseArrayStack._normalize_minmax(arr) * (new_max - new_min)) + new_min

        return arr_rescaled


    @staticmethod
    def _regularize_pairwise_square_matrix(matrix:np.ndarray, hack=False) -> np.ndarray:
        """ Make matrix positive definite by adjusting negative eigenvalues.

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
        matrix_reg = PairwiseArrayStack._make_pairwise_square_matrix_symmetric(matrix_reg)  
        return matrix_reg


    # TODO finish the vectorised implementation for a 3D stack (now 2D matrix)
    @staticmethod
    def _reorganize_pairwise_square_matrix(matrix:np.ndarray, idxs) -> np.ndarray:      
        """ Reorder a matrix using a new set of indices.

        Parameters
        ----------
        matrix : np.ndarray
            Symmetric matrix to reorder.
        idxs : array-like
            New ordering of indices.

        Returns
        -------
        np.ndarray
            Reordered matrix.
        """

        assert matrix.ndim == 2
        assert issymmetric(matrix)
        assert len(np.unique(idxs)) == len(idxs)
        assert np.all(np.logical_and(idxs >= 0, idxs < matrix.shape[0]))

        return matrix[:,idxs][idxs]


    def ensemble(self, method="mean", regularize=False) -> np.ndarray:
        """ Compute an ensemble operation (mean, median, etc.) across the stack.

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
            return squareform(self._regularize_pairwise_square_matrix(squareform(self.data[0]))) \
                if regularize else self.data[0][:]
        
        ensemble_matrix = methods[method](self.data[:], axis=0)
        return squareform(self._regularize_pairwise_square_matrix(squareform(ensemble_matrix))) \
            if regularize else ensemble_matrix


    # NOTE jb's original function was different: data = 1 / (1 + (data / contact_threshold)**10)
    def proximity(self, threshold:Number, hardness=1, keep_NaNs=False):
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

        result = 1 / (1 + np.exp(hardness * (self - threshold)))
        if keep_NaNs:
            return result
        return np.nan_to_num(result, nan=0, copy=True)


    # TODO verify that the function works correctly
    def filter(self, min_efficiency: Number = 0.1) -> np.ndarray:
        """ Filter matrices with insufficient barcode detection.

        Parameters
        ----------
        min_efficiency : float
            Minimum proportion of non-NaN values to retain a matrix.

        Returns
        -------
        np.ndarray
            Filtered subset of the stack.
        """

        proportion_bins_not_missing = np.mean(~np.isnan(self), axis=1)
        idxs = np.argwhere(proportion_bins_not_missing > min_efficiency)

        return np.squeeze(self[idxs])
    

    def efficiency(self, by: str = "pixel"):
        """ Compute detection efficiency by pixel or item.

        Parameters
        ----------
        by : str
            "pixel" for per-bin efficiency, "item" for whole matrices.

        Returns
        -------
        np.ndarray
            Efficiencies in percent (0–100).
        """

        assert isinstance(by, str) and by in ["pixel", "item", "matrix"]

        is_detected = ~np.isnan(self)
        effs = np.around(np.mean(is_detected, axis=0 if by=="pixel" else 1) * 100, decimals=2)

        return effs


    def plot_pwds_avg(self, path_to_save: str,
                      method: str = "mean",
                      form: str = "triangle", cmap: str = "Reds_r"):
        """ Plot the ensemble-averaged pairwise distance matrix across the stack.

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
        None
            Saves the plot to the specified file path.

        Examples
        --------
        >>> stack.plot_pwds_avg("figs/avg_pwds.svg", method="mean", form="square", cmap="viridis")
        """

        assert isinstance(path_to_save, str)
        assert form in ["triangle", "square"]
        assert isinstance(method, str) and method in ["mean", "median"]

        matrix = self.ensemble(method=method)
        if form == "triangle":
            visualization._plot_2d_triangle(matrix=matrix, path_to_save=path_to_save,
                                            cmap=cmap, label=f"{method} distance")
        else:
            visualization._plot_2d_square(matrix=matrix, path_to_save=path_to_save,
                                          cmap=cmap, label=f"{method} distance")


    def plot_proximity_avg(self, path_to_save: str,
                           threshold: Number, hardness: Number = 1,
                           form: str = "triangle", cmap: str = "Reds"):
        """ Plot the ensemble-averaged proximity matrix based on a distance threshold.

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
            Format of the plot: "triangle" for lower triangle heatmap, or "square" for full matrix heatmap.
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

        proximity_stack = self.proximity(threshold=threshold, hardness=hardness)
        matrix = np.nanmean(proximity_stack, axis=0)
        if form == "triangle":
            visualization._plot_2d_triangle(matrix=matrix, path_to_save=path_to_save, cmap=cmap, label="mean proximity")
        else:
            visualization._plot_2d_square(matrix=matrix, path_to_save=path_to_save, cmap=cmap, label="mean proximity")

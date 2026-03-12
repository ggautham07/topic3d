#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data types for chromatin-tracing proximity data (:mod:`topic3d.datatypes`)
=======================================================================

This module defines NumPy-backed containers used throughout *topic3d* to represent
single-cell 3D traces and their derived pairwise quantities.

The two most common representations are:

- :class:`~topic3d.datatypes.CoordsArrayStack`: stacks of 3D coordinates with shape
  ``(num_items, num_bc, 3)`` (or ``(num_bc, 3)`` for a single item).
- :class:`~topic3d.datatypes.PWDistArrayStack`: stacks of flattened pairwise-distance
  vectors with shape ``(num_items, num_bc * (num_bc - 1) // 2)``.

Both are implemented as subclasses of :class:`numpy.ndarray` so they can be passed
directly into NumPy/SciPy/scikit-learn routines, while still offering domain-specific
helpers for:

- quality control (detection efficiency, filtering),
- imputation of missing coordinates,
- conversion between distances and proximities,
- ensemble statistics and visualization.

Notes
-----
Missing values are represented with ``NaN``. Many methods use NaN-aware reductions
(e.g., :func:`numpy.nanmean`).

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
    Stack of 3D chromatin traces.

    A :class:`numpy.ndarray` subclass representing a collection of chromatin traces
    (experimental chromatin tracing or polymer simulations) as 3D coordinates.

    The underlying array has shape ``(num_items, num_bc, 3)``, where:

    - ``num_items`` is the number of single-cell traces / conformations,
    - ``num_bc`` is the number of barcodes (loci) per trace,
    - the last dimension corresponds to ``(x, y, z)``.

    Coordinates may contain ``NaN`` entries for missing loci.

    Attributes
    ----------
    num_items : int
        Number of items in the stack.
    num_bc : int
        Number of barcodes (loci) per item.

    See Also
    --------
    PWDistArrayStack
        Pairwise-distance representation derived from coordinates.

    Examples
    --------
    >>> stack = CoordsArrayStack(np.random.rand(100, 30, 3))
    >>> stack.num_items, stack.num_bc
    (100, 30)
    >>> pwds = stack.pwds()
    >>> eff = stack.efficiency(by="barcode")
    """

    # Initializing using __new__ instead of __init__ in line with Numpy
    def __new__(cls, arr: np.ndarray):
        """
        Create a :class:`~topic3d.datatypes.CoordsArrayStack` from a NumPy array.

        Parameters
        ----------
        arr : numpy.ndarray
            Coordinate array of shape ``(num_items, num_bc, 3)`` or ``(num_bc, 3)``.
            Missing coordinates can be encoded as NaNs.

        Returns
        -------
        CoordsArrayStack
            A view on ``arr`` with convenience attributes and methods.
        """

        assert isinstance(arr, np.ndarray)
        assert arr.ndim in (2, 3)
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


    def compute_CoM(self) -> np.ndarray:
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


    def compute_pwds(self) -> PWDistArrayStack:
        """
        Compute the pairwise distances between all barcodes within each item.

        Returns
        -------
        PWDistArrayStack
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


    def compute_efficiency(self, by: str = "barcode") -> np.ndarray:
        """
        Compute the detection efficiency of barcodes or traces.

        Parameters
        ----------
        by : {"barcode", "item", "trace"}, default="barcode"
            Whether to compute efficiency per barcode across items, or per item.

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
            Minimum detection efficiency (in percent, 0–100).
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

        effs = self.compute_efficiency(by=by)
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
        assert method in ["naive"]
        if np.all(~np.isnan(self)):
            warnings.warn("No NaNs to impute in the stack")
            return self
        
        stack_imp = np.copy(self)
        
        if method == "naive":
            neighbor_means = (self[:, 2:] + self[:, :-2]) / 2   # compute the mean pos of all potential neighbors
            is_missing = np.isnan(stack_imp[:, 1:-1])   # check missing coordinates
            stack_imp[:, 1:-1][is_missing] = neighbor_means[np.where(is_missing)]   # replace NaNs with mean pos

        stack_imp = CoordsArrayStack(stack_imp)

        return stack_imp


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
        >>> fig = coords_stack.plot_bc_efficiency("figs/efficiency.png", include_imputed=True)
        >>> fig.show()
        """

        assert isinstance(path_to_save, str)

        wflag = self._set_figsave_flag(path_to_save)

        effs = self.compute_efficiency(by="barcode")
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


class _PWArrayStack(np.ndarray):
    """
    Base class for stacks of flattened pairwise vectors.

    The flattened representation uses the same convention as
    :func:`scipy.spatial.distance.squareform`: for a locus count ``num_bc``, each
    item is stored as a 1D vector of length ``num_bc * (num_bc - 1) // 2``.

    Subclasses provide semantics for what the vector encodes (e.g., distances or
    proximity values).

    Notes
    -----
    This class validates that the last dimension can be interpreted as
    ``n choose 2`` for some integer ``n >= 2``.
    """

    # Initializing using __new__ instead of __init__ in line with Numpy
    def __new__(cls, arr:np.ndarray):
        """Create a new instance from a pairwise-distance vector stack."""
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
        return f"PWArrayStack\n# items: {self.num_items}\n# barcodes: {self.num_bc}"


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


    def compute_efficiency(self, by: str = "pixel") -> np.ndarray:
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


    def compute_ensemble_measure(self, method="mean", regularize=False) -> np.ndarray:
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

        effs = self.compute_efficiency(by="pixel")
        fig = map_form_fn[form](items=[effs],
                                cmap="magma",
                                cbar_label="efficiency %")

        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig
    

    def plot_ensemble_measure(self, method: str = "mean",
                              path_to_save: str = "",
                              form: str = "triangle", cmap: str = "Reds", cbar_label: str = None):
        """
        Plot the ensemble-averaged matrix across the stack.

        Parameters
        ----------
        path_to_save : str
            Path (with file extension) where the figure will be saved.
        method : str, default="mean"
            Aggregation method to compute ensemble measure. Supported values same as for `compute_ensemble_measure`
        form : str, default="triangle"
            Format of the plot: "triangle" for lower triangle heatmap, or "square" for full matrix heatmap.
        cmap : str, default="Reds_r"
            Matplotlib colormap to use for the plot.
        cbar_label : str, default=None
            If string is passed, string. If None, will be method + "value"

        Returns
        -------
        Figure
            Saves the plot to the specified file path.

        Examples
        --------
        >>> stack.plot_ensemble_avg("figs/avg_pwds.svg", method="mean", form="square", cmap="viridis")
        """

        assert isinstance(path_to_save, str)
        assert form in ["triangle", "square"]
        if cbar_label is None:
            cbar_label = ""
        else:
            cbar_label = str(cbar_label)

        wflag = self._set_figsave_flag(path_to_save)

        map_form_fn = {"triangle": vzl.plot_matrices_triangle, "square": vzl.plot_matrices_square,}

        matrix = self.compute_ensemble_measure(method=method)
        fig = map_form_fn[form](items=[matrix],
                                titles=[f"$n = ${self.num_items:,}"],
                                cmap=cmap,
                                cbar_label=f"{method} value" if cbar_label is None else cbar_label)
        
        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig
    

    def plot_random_examples(self, num_items: int = 9, random_seed: int = None,
                             path_to_save: str = "", form: str = "triangle", cmap="Reds",
                             include_cbar=True, cbar_label: str = None):
        """
        Plot a set of randomly-picked items from the dataset.

        Parameters
        --------
        num_items : Number
            number of items to plot
        random_seed : int or None
            random seed for reproducibility
        path_to_save : str
            where to save the plot, if unspecified, returns only the Figure
        cbar_label : str, default=None
            If string is passed, string. If None, will be empty

        Returns
        --------
        Figure
        """

        assert isinstance(num_items, int)
        assert isinstance(path_to_save, str)
        assert form in ["triangle", "square"]
        if cbar_label is None:
            cbar_label = ""

        wflag = self._set_figsave_flag(path_to_save)

        map_form_fn = {"triangle": vzl.plot_matrices_triangle, "square": vzl.plot_matrices_square,}

        random_gen = np.random.default_rng(seed=random_seed)
        idxs = random_gen.integers(self.num_items, size=num_items)
        
        fig = map_form_fn[form](items=self[idxs, ...],
                                titles=[f"Index  {i:,}" for i in idxs],
                                cmap=cmap,
                                include_cbar=include_cbar,
                                cbar_label="",)
        
        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig



class PWDistArrayStack(_PWArrayStack):
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
    plot_ensemble_avg(path_to_save, method='mean', form='triangle', cmap='Reds_r')
        Plot ensemble-average pairwise distance matrix.
    plot_proximity_avg(path_to_save, threshold, hardness=1, form='triangle', cmap='Reds')
        Plot ensemble-average proximity matrix.
    """

    # Initializing using __new__ instead of __init__ in line with Numpy
    def __new__(cls, arr:np.ndarray):
        """Create a new instance from a pairwise-distance vector stack."""
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
        return f"PWArrayStack\n# items: {self.num_items}\n# barcodes: {self.num_bc}"


    # Since new attributes were added in __new__
    def __array_finalize__(self, obj):
        # reset the attribute from passed original object
        self.num_items = getattr(obj, "num_items", None)
        self.num_bc = getattr(obj, "num_bc", None)


    # NOTE JB's original function was different: data = 1 / (1 + (data / contact_threshold)**10)
    def compute_proximity(self, threshold:Number, hardness=1, keep_NaNs=False) -> np.ndarray:
        """
        Convert distances into proximity values using a sigmoid function or hard threshold.

        Hardness describes the sigmoid slope: ``0 < hardness <= 1``. ``hardness=1`` gives a hard cutoff.
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
            stack = self.compute_proximity(threshold=thr, hardness=hardness, keep_NaNs=False)
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
    

    def plot_proximity_avg_comparisons(self, thresholds: Iterable, hardnesses: Iterable,
                                        path_to_save: str = "", form: str = "triangle", cmap="Reds"):
        """
        Compare ensemble-average proximity maps across parameter settings.

        This is a convenience wrapper around
        :meth:`topic3d.datatypes.PWDistArrayStack.proximity` followed by averaging
        across items.

        Parameters
        ----------
        thresholds : Iterable[float]
            Distance thresholds ``k`` used to convert pairwise distances to proximities.
        hardnesses : Iterable[float]
            Hardness values ``h`` (``0 < h <= 1``) controlling the steepness of the
            distance-to-proximity sigmoid (``h=1`` is a hard cutoff).
        path_to_save : str, optional
            If provided, save the figure to this path.
        form : {{"triangle", "square"}}, default="triangle"
            Whether to plot the lower triangle or the full square matrix.
        cmap : str, default="Reds"
            Matplotlib colormap.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
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



class PWProxArrayStack(_PWArrayStack):
    """
    Stack of flattened pairwise *proximity* vectors.

    This container is analogous to :class:`~topic3d.datatypes.PWDistArrayStack`,
    but stores proximity values (typically in ``[0, 1]``) instead of distances.
    """

    # Initializing using __new__ instead of __init__ in line with Numpy
    def __new__(cls, arr:np.ndarray):
        """Create a new instance from a pairwise-distance vector stack."""
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
        return f"PWArrayStack\n# items: {self.num_items}\n# barcodes: {self.num_bc}"


    # Since new attributes were added in __new__
    def __array_finalize__(self, obj):
        # reset the attribute from passed original object
        self.num_items = getattr(obj, "num_items", None)
        self.num_bc = getattr(obj, "num_bc", None)
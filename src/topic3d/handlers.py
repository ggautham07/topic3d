#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Handlers for topic decomposition input data and results (:mod:`topic3d.handlers`)
============================================================================

This module provides classes and functions for loading, parsing, processing, and analyzing
spatial genomics data using topic modeling approaches.

Classes
-------
- RawData : Handles loading and preprocessing of raw trace data from `.csv` (FOF) or `.npy` files.
- ProcessedData : Lightweight container for storing processed trace data and derived attributes.
- DecompositionResult : Container for topic decomposition results, with support for evaluation,
  visualization, and clustering of learned components.

Authors
-------
jbfiche, ggautham07, marcnol

"""


# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations
from os import path
import warnings
from io import StringIO
from numbers import Number

from itertools import combinations
import joblib as jl

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from .datatypes import CoordsArrayStack, PWDistArrayStack
from . import visualization as vzl



# =============================================================================
# CLASSES
# =============================================================================


class RawData:
    """
    Handler for raw spatial genomics data in NumPy (.npy) or Fish Omics Format (.csv) files.

    This class supports:
    - Loading and parsing single-cell spatial trace data.
    - Interfacing with raw NumPy arrays or structured FOF (Fish Omics Format) CSVs.
    - Extracting metadata, coordinates, barcode bins, and trace IDs.
    - Converting raw input into a `ProcessedData` object for downstream analysis.
    - Computing basic genomic and 3D spatial statistics.

    Attributes
    ----------
    library : pd.DataFrame or None
        Full FOF-formatted data table if a `.csv` file is loaded, otherwise None.
    traces : CoordsArrayStack or None
        Parsed coordinate array stack (shape: N_traces × N_barcodes × 3) for XYZ positions.
    num_traces : int or None
        Number of traces (typically one per nucleus or region).
    num_bc : int or None
        Number of barcodes per trace.
    bc_ids : np.ndarray or None
        Indices or identifiers for barcodes.
    source : list of str or None
        File path(s) from which data was loaded.
    
    Methods
    -------
    load(path_to_file, metadata=None, extract_items=True)
        Load raw data from a .csv (FOF) or .npy (NumPy) file.
    process()
        Convert raw traces and metadata into a `ProcessedData` object.
    compute_3d_vs_genomic_dists(max_realistic_3d_dist, experiment, axis="X")
        Relate 3D distances to genomic distances between barcodes.
    compute_inter_loci_genomic_dist()
        Compute genomic distances between consecutive barcode loci.
    plot_3d_dist_scaling(dist_df, dim, saving_filename, normalized=False)
        Visualize scaling of 3D distance vs genomic distance across axes.

    Notes
    -----
    - For `.npy` input, some metadata (e.g., barcode bins) must be provided via `metadata`.
    - FOF `.csv` files contain more comprehensive information and are recommended.
    - Traces are expected to be in 3D (XYZ); other formats may not be supported.
    - Use `ProcessedData` for downstream computations and visualization.

    Examples
    --------
    >>> raw = RawData()
    >>> raw.load("trace_data.csv")
    >>> pdata = raw.process()
    >>> df = raw.compute_3d_vs_genomic_dists(600, experiment=1)
    >>> raw.plot_3d_dist_scaling(df, dim={"X": 0, "Y": 1, "Z": 2}, saving_filename="scaling.png")
    """

    def __init__(self):
        self.library = None
        self.traces = None
        self.num_traces = None
        self.num_bc = None
        self.bc_ids = None
        self.source = None


    def __str__(self):
        out = ["RawData object"]
        avail_fof = False
        if self.library is not None and isinstance(self.library, pd.DataFrame):
            avail_fof = True
            out.append(f"Fish Omics Format (FOF) data: {'available' if avail_fof else 'unavailable'}")
        avail_tr = False
        if self.traces is not None and isinstance(self.traces, np.ndarray):
            avail_tr = True
            out.append(f"Traces array: {'available' if avail_tr else 'unavailable'}")
        if self.num_traces is not None:
                out.append(f"Number of traces: {self.num_traces}")
        if self.num_bc is not None:
                out.append(f"Number of barcodes: {self.num_bc}")
        out = "\n".join(out)
        return out
    

    @staticmethod
    def _extract_header_row_from_fof(full_seq:str, sub_seq:str) -> list:
        """
        Internal function used to extract header rows from Fish Omics Format (FOF) data
        
        Finds the first occurrence of sub_seq in full_seq and splits it at the first newline
        """

        return full_seq[full_seq.index(sub_seq):].split("\n", maxsplit=1)
    

    def _extract_traces_from_fof(self) -> np.ndarray:
        """
        Extract trace data from a loaded FOF-CT table and reformat into a 3D NumPy array.

        Returns
        -------
        np.ndarray
            Array of shape (num_traces, num_barcodes, 3) containing X, Y, Z coordinates.
        
        Examples
        --------
        >>> traces = raw_data._extract_traces_from_fof()
        """

        traces = np.empty((self.num_traces, self.num_bc, 3))
        traces[:] = np.nan
        # create the trace from the coordinates
        for j, curr_trace_id in enumerate(self.trace_ids):
            curr_data = self.library[self.library["Trace_ID"] == curr_trace_id]
            barcodes = [self.bc_ids[x] for x in curr_data["Chrom_Start"].to_list()]
            coords = curr_data[["X", "Y", "Z"]].to_numpy()
            traces[j, barcodes] = coords[:]
        
        return traces
    
    
    # TODO integrate with main load function, if possible including similar version for FOF data 
    # (code available from Marcelo)
    @staticmethod
    def _merge_numpy_trace_files(*args) -> np.ndarray:
        """
        Merge multiple NumPy trace arrays into a single array with shape (N, M, 3).

        Parameters
        ----------
        *args : str
            Paths to .npy files. Each must contain a 3D array of shape (n_traces, n_barcodes, 3).

        Returns
        -------
        np.ndarray
            A new NumPy array with the merged traces. Empty values are filled with NaNs.
        
        Examples
        --------
        >>> merged = RawData._merge_numpy_trace_files("a.npy", "b.npy", "c.npy")
        """

        # Get the datasets
        fps = [fp for fp in args if path.exists(fp) and fp.endswith(".npy")]
        datasets = [np.load(fp, allow_pickle=True) for fp in fps]

        # New dataset has the maximum number of barcodes from initial dataset list
        num_bc_max = None
        num_items = 0
        for ds in datasets:
            assert ds.ndim == 3
            assert ds.shape[-1] == 3
            num_items += ds.shape[0]
            if num_bc_max is None:
                num_bc_max = ds.shape[1]
                continue
            if ds.shape[0] > num_bc_max:
                num_bc_max = ds.shape[0]
        
        # Filling the data from each dataset, rest is NaN
        out = np.zeros(num_items, num_bc_max, 3)
        out[out == 0] = np.nan
        start = 0
        for ds in datasets:
            assert np.all(out[start:ds.shape[0], 0:ds.shape[1], :] == np.nan)
            out[start:ds.shape[0], 0:ds.shape[1], :] = ds[:]
            start += ds.shape[0]
        
        return out


    # TODO path_to_file can also take a list of files with the same extension/format, and merge the data
    # NOTE _merge_numpy_trace_files and marcnol's FOF data handler code to be used for merging
    def load(self, path_to_file:str, metadata:dict = None, extract_items:bool = True) -> None:
        """
        Load trace data from a file (CSV for FOF format or NPY for NumPy arrays).

        Parameters
        ----------
        path_to_file : str
            Path to the trace file. Must end with `.csv` (FOF-CT) or `.npy` (NumPy).
        metadata : dict, optional
            Metadata dictionary required for loading NPY files. Should contain keys like
            "trace_ids", "bc_ids", and "bc_bins".
        extract_items : bool, optional
            If True (default), extract traces directly from FOF table.

        Returns
        -------
        None
            Sets internal attributes such as `traces`, `bc_bins`, `trace_ids`, etc.

        Raises
        ------
        Exception
            If the file extension is not supported.
        
        Examples
        --------
        >>> raw_data.load("dataset.csv")
        >>> raw_data.load("dataset.npy", metadata={"bc_ids": np.arange(30)})
        """

        assert path.isfile(path_to_file)
        self.source = [path_to_file]

        if path_to_file.endswith(".npy"):
            self.traces = CoordsArrayStack(np.load(path_to_file, allow_pickle=True))
            assert self.traces.data.shape[-1] == 3
            self.num_traces, self.num_bc, _ = self.traces.shape
            self.bc_ids = np.arange(self.num_bc)
            if isinstance(metadata, dict):
                self.trace_ids = metadata.get("trace_ids", None)
                self.bc_ids = metadata.get("bc_ids", np.arange(self.num_bc))
                self.bc_bins = metadata.get("bc_bins", None)
                if self.bc_bins is not None:
                    self.bc_bins_sizes = self.bc_bins[:, 1] - self.bc_bins[:, 0]
                    self.bc_bins_contiguity = np.all(self.bc_bins[1:, 0] - self.bc_bins[:-1, 1] == 0)
                for v in vars(self):
                    if v is None and v not in ["library", "traces", "num_traces", "num_bc", "bc_ids"]:
                        del v


        elif path_to_file.endswith(".csv"):

            # load the data and organize them as a dataframe
            contents = open(path_to_file, mode="r").read()  # load all contents as string
            table_header_raw, table_data_raw = self._extract_header_row_from_fof(contents,
                                                                "##columns")  # split the header from the trace data
            colnames = table_header_raw[table_header_raw.index("(")+1:table_header_raw.index(")")].split(",")
            colnames = [c.strip() for c in colnames]  # get column names
            data = pd.read_csv(StringIO(table_data_raw), sep=",", header=None)  # get trace data
            data.columns = colnames  # assign column names to trace table

            # Locating other key metadata
            assembly = self._extract_header_row_from_fof(contents, "##genome_assembly")[0].split("=")[1]
            unit = self._extract_header_row_from_fof(contents, "##XYZ_unit")[0].split("=")[1]

            # Variable assignment
            self.library = data.copy()
            self.unit = unit
            self.assembly = assembly

            self.trace_ids = np.unique(self.library["Trace_ID"])
            self.num_traces = len(self.trace_ids)

            chrom_starts = np.sort(self.library.Chrom_Start.unique())
            chrom_ends = np.sort(self.library.Chrom_End.unique())
            assert len(chrom_starts) == len(chrom_ends)
            self.bc_bins = np.hstack((chrom_starts[..., np.newaxis], chrom_ends[..., np.newaxis]))
            self.num_bc = len(self.bc_bins)
            self.bc_ids = np.arange(self.num_bc)
            self.bc_bins_sizes = self.bc_bins[:, 1] - self.bc_bins[:, 0]
            self.bc_bins_contiguity = np.all(self.bc_bins[1:, 0] - self.bc_bins[:-1, 1] == 0)   # TODO contiguity array

            # Extraction of traces
            if extract_items:
                self.traces = CoordsArrayStack(self._extract_traces_from_fof())
        
        else:
            raise Exception("Invalid file extension. Only npy and csv files accepted")


    def process(self) -> ProcessedData:
        """
        Process the trace data and return it as a `ProcessedData` object.

        This method extracts metadata and computes:
        - Barcode detection efficiency,
        - Pairwise distances for all traces,
        - Source and trace-level attributes,
        and packages them into a `ProcessedData` container.

        Returns
        -------
        ProcessedData
            A container object with processed trace data, metadata, and derived features.

        Raises
        ------
        AssertionError
            If `self.traces` is None.
        """

        assert self.traces is not None, "Traces object `ProcessedData.traces` is required for processing"

        pdata = ProcessedData()
        pdata.traces = self.traces
        pdata.bc_effs = self.traces.efficiency(by="barcode")
        pdata.pwds = self.traces.pwds()
        pdata.metadata = {"num_traces": self.num_traces,
                          "num_bc": self.num_bc,
                          "bc_ids": self.bc_ids,
                          "source": self.source}
        if self.library is not None:
            pdata.metadata.update({"bc_bins": self.bc_bins,
                                   "bc_bins_sizes": self.bc_bins_sizes,
                                   "bc_bins_contiguity": self.bc_bins_contiguity})

        return pdata
    
    
    def compute_3d_vs_genomic_dists(self, max_realistic_3d_dist:Number, experiment:int, axis="X") -> pd.DataFrame:
        """
        Compute the relationship between 3D Euclidean distances and genomic distances between barcodes.

        For each pair of barcodes, their genomic distance is computed based on their bin positions,
        and their 3D distance is obtained from the pairwise distance matrix. Unrealistic 3D distances
        above a given threshold are discarded. Distances are then binned by genomic separation,
        and the median 3D distance per bin is returned.

        Parameters
        ----------
        max_realistic_3d_dist : Number
            Maximum 3D distance (in nm) considered realistic; larger values are discarded as outliers.
        experiment : int
            Identifier for the experiment (used to annotate the output).
        axis : str, optional
            Axis label (e.g., "X", "Y", "Z") to annotate the output. Default is "X".

        Returns
        -------
        pd.DataFrame
            DataFrame containing binned genomic distances and corresponding median 3D distances.
            Includes log-transformed columns and counts per bin.

            Columns:
            - 'genomic distance (kbp)'
            - 'log10 genomic dist (kbp)'
            - 'median euclidean distance (nm)'
            - 'log10 median dist (nm)'
            - 'n_data'
            - 'experiment'
            - 'axis'

        Raises
        ------
        None

        Notes
        -----
        - Genomic distances are computed as the difference of mean bin positions.
        - Output can be used for plotting dose–response-like curves in chromatin modeling.
        """

        pw_dists_3d = self.traces.pwds()
        pw_dists_genomic = np.asarray([np.mean(self.bc_bins[j]) - np.mean(self.bc_bins[i]) \
                               for i, j in combinations(self.bc_ids, 2)])

        # compile all the distance maps and aggregate genetic and euclidean distances
        pw_dists_genomic_3d_combined = []
        for i in range(self.num_traces):
            merged = np.column_stack((pw_dists_genomic, pw_dists_3d[i, :]))
            pw_dists_genomic_3d_combined.append(merged)

        pw_dists_genomic_3d_combined = np.asarray(pw_dists_genomic_3d_combined)
        pw_dists_genomic_3d_combined = np.reshape(pw_dists_genomic_3d_combined,
                                                       (self.num_traces * len(pw_dists_genomic), 2))
        pw_dists_genomic_3d_combined[pw_dists_genomic_3d_combined[:, 1] > max_realistic_3d_dist, 1] = np.nan

        # to make the plot easier to read, data are binned
        bins_genomic = np.linspace(start=np.min(pw_dists_genomic),
                                   stop=np.max(pw_dists_genomic),
                                   num=30)
        dists_3d_vs_genomic = np.zeros((len(bins_genomic)-1, 3))
        for bin in range(len(bins_genomic) - 1):
            idx = np.where((pw_dists_genomic_3d_combined[:, 0] >= bins_genomic[bin]) & \
                           (pw_dists_genomic_3d_combined[:, 0] < bins_genomic[bin+1]))
            dists_3d_vs_genomic[bin, :] = [bins_genomic[bin], 
                                           np.nanmedian(pw_dists_genomic_3d_combined[idx, 1]), 
                                           len(idx[0])]

        results = pd.DataFrame()
        results["genomic distance (kbp)"] = dists_3d_vs_genomic[:, 0]
        results["log10 genomic dist (kbp)"] = np.log10(dists_3d_vs_genomic[:, 0])
        results["median euclidean distance (nm)"] = dists_3d_vs_genomic[:, 1]
        results["log10 median dist (nm)"] = np.log10(dists_3d_vs_genomic[:, 1])
        results["n_data"] = dists_3d_vs_genomic[:, 2]
        results["experiment"] = [f'exp_{experiment}'] * (len(bins_genomic) - 1)
        results["axis"] = [axis] * (len(bins_genomic) - 1)

        return results
    

    def compute_inter_loci_genomic_dist(self) -> np.ndarray:
        """
        Compute the genomic distances between consecutive barcodes in the trace.

        This function calculates the difference in mean genomic bin positions between
        each consecutive pair of barcodes. It assumes that `self.bc_bins` contains the
        genomic bin ranges (e.g., [start, end]) for each barcode.

        Returns
        -------
        np.ndarray
            1D array of genomic distances (in bin units) between each pair of consecutive barcodes.

        Raises
        ------
        AssertionError
            If `self.bc_bins` is not defined.

        Warns
        -----
        UserWarning
            If barcode bins are not contiguous (`self.bc_bins_contiguity=False`), which may affect interpretation.

        Examples
        --------
        >>> distances = handler.compute_inter_loci_genomic_dist()
        >>> print(distances)
        array([2000., 1500., 1800., ...])
        """

        assert self.bc_bins
        if not self.bc_bins_contiguity:
            warnings.warn("Note: discontiguous segments supplied")

        bins_mean_pos = np.mean(self.bc_bins, axis=1)

        return bins_mean_pos[1:] - bins_mean_pos[:-1]
    

    @staticmethod
    def plot_3d_dist_scaling(dist_df:pd.DataFrame, dim:dict, saving_filename:str, normalized=False):
        """
        Plot the scaling of 3D distance vs genomic distance in log-log space, for raw or normalized data.

        Parameters
        ----------
        dist_df : pd.DataFrame
            A dataframe containing histogrammed distances with columns including 
            "log10 genomic dist (kbp)", "log10 median dist (nm)" or "log10 normalized (nm)", 
            and "experiment".
        dim : dict
            Dictionary mapping axis labels (e.g., "X", "Y", "Z") to subplot indices (0, 1, 2).
        saving_filename : str
            Path to save the generated figure (including extension).
        normalized : bool, optional
            Whether to use normalized 3D distances for plotting (default is False).

        Returns
        -------
        None
            Saves a multi-panel scatterplot of distance scaling to the specified file.

        Examples
        --------
        >>> RawData.plot_dist_scaling(dist_df, dim={"X": 0, "Y": 1, "Z": 2}, saving_filename="scaling.png")
        """
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axs = plt.subplots(nrows=3, ncols=1)
        for ax_name, ax in dim.items():
            if normalized:
                df = dist_df.loc[dist_df["axis"] == ax_name, ["log10 genomic dist (kbp)",
                                                              "log10 normalized (nm)",
                                                              "experiment"]]
                sns.scatterplot(data=df, x="log10 genomic dist (kbp)", y="log10 normalized (nm)",
                                hue="experiment", ax=axs[ax])
            else:
                df = dist_df.loc[dist_df["axis"] == ax_name, ["log10 genomic dist (kbp)",
                                                            "log10 median dist (nm)",
                                                            "experiment"]]
                sns.scatterplot(data=df, x="log10 genomic dist (kbp)", y="log10 median dist (nm)",
                                hue="experiment", ax=axs[ax])
        fig.set_size_inches(8, 10)
        plt.savefig(saving_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)



class ProcessedData:
    """
    Container for processed data including 3D coordinates, pairwise distances, and metadata.

    Attributes
    ----------
    traces : CoordsArrayStack or None
        Stack of trace coordinates (if available).
    pwds : PWDistArrayStack or None
        Stack of pairwise distances (if available).
    metadata : dict or None
        Metadata dictionary describing the data (e.g., barcode bins, efficiencies).

    Methods
    -------
    load(path_to_file: str)
        Load processed data from an HDF5 file.
    save(path_to_file: str, replace: bool = True)
        Save current data and metadata to an HDF5 file.
    """

    def __init__(self):
        self.traces = None
        self.pwds = None
        self.metadata = None

 
    def __str__(self):
        out = ["ProcessedData object"]
        avail_tr = False
        if self.traces is not None and isinstance(self.traces, np.ndarray):
            avail_tr = True
            out.append(f"Traces array: {'available' if avail_tr else 'unavailable'}")
        avail_pwds = False
        if self.pwds is not None and isinstance(self.pwds, np.ndarray):
            avail_pwds = True
            out.append(f"PWDists array: {'available' if avail_pwds else 'unavailable'}")
        out = "\n".join(out)
        return out


    def load(self, path_to_file:str) -> None:
        """
        Load processed data (traces and/or pairwise distances) from an HDF5 file.

        Parameters
        ----------
        path_to_file : str
            Path to the HDF5 file to load.

        Returns
        -------
        None
            Sets `traces`, `pwds`, and `metadata` attributes.

        Raises
        ------
        AssertionError
            If the file is not HDF5 or is missing both traces and distances.

        Examples
        --------
        >>> pdata = ProcessedData()
        >>> pdata.load("processed.h5")
        """

        assert path_to_file.endswith(".h5")
        assert path.isfile(path_to_file)
        with h5py.File(path_to_file, mode="r") as handler:
            datasets = {x:y[:] for x, y in handler.items()}
            attributes = {x:y[:] for x, y in handler.attrs.items()}
        assert ("traces" in datasets.keys()) or ("pwds" in datasets.keys())
        self.traces = datasets.get("traces", None)
        if self.traces is not None:
            self.traces = CoordsArrayStack(self.traces)
        self.traces_imputed = datasets.get("traces_imputed", None)
        if self.traces_imputed is not None:
            self.traces_imputed = CoordsArrayStack(self.traces_imputed)
        self.pwds = datasets.get("pwds", None)
        if self.pwds is not None:
            self.pwds = PWDistArrayStack(self.pwds)
        self.pwds_norm = {}
        self.pwds_norm["MMSF"] = datasets.get("pwds_norm-MMSF", None)
        self.pwds_norm["R(s)"] = datasets.get("pwds_norm-R(s)", None)
        self.pwds_imputed_norm = {}
        self.pwds_imputed_norm["MMSF"] = datasets.get("pwds_imputed_norm-MMSF", None)
        self.pwds_imputed_norm["R(s)"] = datasets.get("pwds_imputed_norm-R(s)", None)
        for norm_key in self.pwds_norm.keys():
            if self.pwds_norm[norm_key] is not None:
                self.pwds_norm[norm_key] = PWDistArrayStack(self.pwds_norm[norm_key])
            if self.pwds_imputed_norm[norm_key] is not None:
                self.pwds_imputed_norm[norm_key] = PWDistArrayStack(self.pwds_imputed_norm[norm_key])
        self.metadata = attributes


    def save(self, path_to_file: str, replace: bool = True) -> None:
        """
        Save current traces, pairwise distances, and metadata to an HDF5 file.

        Parameters
        ----------
        path_to_file : str
            Destination file path for the HDF5 file.
        replace : bool, optional
            If False, raises an error if the file already exists (default is True).

        Returns
        -------
        None

        Raises
        ------
        FileExistsError
            If file exists and `replace` is set to False.

        Examples
        --------
        >>> pdata.save("output.h5")
        """

        if path.isfile(path_to_file) and not replace:
            raise FileExistsError("File already exists and replace is set to False")
        attr_labels = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        print("Attempting to save the following: ", *attr_labels, end="\n")
        with h5py.File(path_to_file, mode="w") as handler:
            for lbl in attr_labels:
                attr = getattr(self, lbl)
                try:
                    print(lbl, type(attr), attr.dtype)
                    handler.create_dataset(name=lbl, data=attr)
                    print(f"Saved {lbl} successfully as a dataset")
                except AttributeError:
                    pass
                except TypeError:
                    handler.attrs[lbl] = attr
                    print(f"Saved {lbl} successfully in metadata")
                except not (TypeError or AttributeError) as err:
                    print(f"Unexpected error occurred while saving {lbl}", err)
                    continue


# TODO
class DecompositionResult:
    """
    Handler for LDA-based decomposition results and downstream analyses.

    This class stores and processes the results of topic modeling (e.g., via Latent Dirichlet Allocation)
    and provides convenience methods for evaluating component quality, plotting, and clustering.

    Attributes
    ----------
    model : LatentDirichletAllocation
        The trained LDA model from which components were derived.
    decomposition : np.ndarray
        The document-topic matrix (samples × topics).
    components : np.ndarray
        The topic-word matrix (topics × features).
    components_norm : np.ndarray
        Normalized version of the components matrix where each row sums to 1.
    num_components : int
        Number of topics (components).
    num_items : int
        Number of samples in the dataset.
    metadata : dict
        Optional metadata associated with the decomposition (e.g., training/testing indices).
    data_training : np.ndarray or PWDistArrayStack, optional
        Training data used to compute coherence scores.
    data_testing : np.ndarray or PWDistArrayStack, optional
        Testing data reserved for evaluation.

    Methods
    -------
    save(dir_to_save)
        Save the decomposition model, matrices, and metadata to disk.
    reconstruction(normalize=False)
        Reconstruct the original data from the decomposition and components.
    representativity(threshold=0.2)
        Compute the fraction of samples where each component is active.
    sort_components(threshold=None, return_freqs=False)
        Sort components by representativity across samples.
    divergence(path_to_save=None)
        Compute and optionally plot the Jensen-Shannon divergence between components.
    coherence(method="all", num_top_pixels=None)
        Compute coherence metrics (NPMI, cosine, uMass) for components.
    plot_components(normalized=False, ordered=False, ...)
        Plot component matrices as heatmaps (triangle or square form).
    plot_decomposition(...)
        Plot the full decomposition matrix as a heatmap.
    embed_decomposition(...)
        Embed the decomposition matrix in 2D using PCA, UMAP, or t-SNE with optional clustering.
    """

    def __init__(self, model:LatentDirichletAllocation, decomposition:np.ndarray, **kwargs):
        
        self.model = model
        self.decomposition = decomposition
        self.components = model.components_
        self.components_norm = self.components / np.sum(self.components, axis=1)[:, np.newaxis]
        self.num_components = len(self.components)
        self.num_items = len(self.decomposition)
        self.metadata = kwargs.copy()
        
        # # TODO store original dataset indices of the training and testing sets using self.data_idxs = kwargs.get()
        # self.data_training = self.metadata.get("data_training", None)
        # if self.data_training is not None:
        #     assert isinstance(self.data_training, (PWDistArrayStack, np.ndarray))
        # self.data_testing = self.metadata.get("data_testing", None)
        # if self.data_testing is not None:
        #     assert isinstance(self.data_testing, (PWDistArrayStack, np.ndarray))


    def save(self, dir_to_save:str):
        """
        Save the decomposition results to disk.

        This includes:
        - components and decomposition matrices in HDF5 format
        - model in joblib format
        - metadata as HDF5 attributes

        Parameters
        ----------
        dir_to_save : str
            Path to directory where files will be saved.

        Returns
        -------
        None

        Examples
        --------
        >>> model.save("results/experiment_01/")
        """
        
        with h5py.File(path.join(dir_to_save, "results.h5"), mode="w") as handler:
            handler.create_dataset(name="components", data=self.components)
            handler.create_dataset(name="decomposition", data=self.decomposition)
            handler.create_dataset(name="data_idxs/training", data=self.data_training)
            handler.create_dataset(name="data_idxs/testing", data=self.data_testing)
            handler.attrs = self.metadata
        
        jl.dump(self.model, path.join(dir_to_save, "model_lda.jl"))


    @staticmethod
    def _set_figsave_flag(fp:str) -> bool:
        """Set the flag to save a Figure based on path string length"""

        if len(fp.strip()) > 0:
            return True
        return False


    def reconstruction(self, normalize=False):
        """
        Reconstruct proximity vectors from the components and decomposition matrices.

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize each component vector to sum to 1 (default is False).

        Returns
        -------
        np.ndarray
            Reconstructed data matrix of shape (n_samples, n_features).

        Examples
        --------
        >>> rec = model.reconstruction(normalize=True)
        """

        if normalize:
            return self.decomposition @ self.components_norm
        return self.decomposition @ self.components
    

    # TODO multiple methods
    @staticmethod
    def _extract_threshold(decomposition, method=None):
        if method is None:
            return (1 / decomposition.shape[-1]) + 0.1


    def representativity(self, threshold: float = None) -> np.ndarray:
        """
        Compute the representativity of each component across the dataset.

        Representativity is the proportion of samples in which each component's 
        activation exceeds a given threshold.

        Parameters
        ----------
        threshold : float, optional
            Activation threshold to count a component as "present" (default will be $\\frac{1}{n_c} + 0.1$).

        Returns
        -------
        np.ndarray
            Array of shape (n_components,) with representativity scores between 0 and 1.

        Examples
        --------
        >>> model.representativity(threshold=0.3)
        array([0.85, 0.92, 0.74, ...])
        """
        
        if threshold is None:
            threshold = self._extract_threshold(self.decomposition)
        else:
            assert isinstance(threshold, Number)

        return np.mean(self.decomposition >= threshold, axis=0)


    @staticmethod
    def _compute_coherence_umass(components, data_used, num_top_pixels=10, randomize_data=False):
        """
        Compute the uMass coherence score for each component.

        The uMass coherence measures how frequently the top pixels (features) 
        of a component co-occur in the data.

        Parameters
        ----------
        components : np.ndarray
            Component matrix of shape (n_components, n_features).
        data_used : np.ndarray
            Binarized data matrix used to compute co-occurrence frequencies.
        num_top_pixels : int, optional
            Number of top pixels to use from each component (default is 10).
        randomize_data : bool, optional
            If True, select random pixels instead of top-k (default is False).

        Returns
        -------
        np.ndarray
            uMass coherence scores for each component, shape (n_components,).

        Examples
        --------
        >>> _compute_coherence_umass(components, data_used)
        array([0.31, 0.26, 0.38, ...])
        """

        num_components = len(components)
        coherence_umass = np.zeros((num_components,))
        data_used = np.asarray(data_used)
        data_used_freq = np.sum(data_used, axis=0)

        EPS = 10**-5    # value of epsilon is kept constant

        for k in range(num_components):
            comp = components[k, :]
            if randomize_data:
                idx = np.random.permutation(comp.shape[0])
            else:
                idx = np.argsort(comp)[::-1]
            idxs_top = idx[:num_top_pixels]

            c_umass = 0
            for m in range(1, num_top_pixels):
                for n in range(0, m):
                    d_mn = np.sum((data_used[:, idxs_top[m]] == 1) & (data_used[:, idxs_top[n]] == 1))
                    d_nn = data_used_freq[idxs_top[n]]
                    if d_nn > 0:
                        c_umass += np.log2((d_mn + EPS) / d_nn)
            coherence_umass[k] = 2 / (num_top_pixels * (num_top_pixels - 1)) * c_umass

        return coherence_umass


    @staticmethod
    def _compute_coherence_npmi_cos(components, data_used, num_top_pixels=10, randomize_data=False):
        """
        Compute both the NPMI and cosine coherence scores for each component.

        Parameters
        ----------
        components : np.ndarray
            Component matrix of shape (n_components, n_features).
        data_used : np.ndarray
            Binarized data matrix for co-occurrence calculation.
        num_top_pixels : int, optional
            Number of top pixels per component to use for computing coherence (default is 10).
        randomize_data : bool, optional
            If True, use random top pixels instead of sorted ones.

        Returns
        -------
        tuple of np.ndarray
            - NPMI coherence scores, shape (n_components,)
            - Cosine coherence scores, shape (n_components,)

        Examples
        --------
        >>> _compute_coherence_npmi_cos(components, data_used)
        (array([...]), array([...]))
        """
        
        num_components = len(components)
        coherence_npmi = np.zeros((num_components,))
        coherence_cos = np.zeros((num_components,))
        data_used = np.asarray(data_used)
        data_used_probs = np.mean(data_used, axis=0)

        EPS = 10**-5    # value of epsilon is kept constant

        for k in range(num_components):
            comp = components[k, :]
            if randomize_data:
                idx = np.random.permutation(comp.shape[0])
            else:
                idx = np.argsort(comp)[::-1]  # return the bin index in descending order
            idxs_top = idx[:num_top_pixels]

            # Compute the NPMI for all pairs of selected bins
            npmi = np.zeros((num_top_pixels, num_top_pixels))
            for m in range(num_top_pixels):
                for n in range(m+1, num_top_pixels):
                    i, j = idxs_top[m], idxs_top[n]
                    co_occurrence = np.mean((data_used[:, i] >= 0.75) & (data_used[:, j] >= 0.75))
                    p_mn = co_occurrence
                    p_nn = data_used_probs[idx[n]]
                    p_mm = data_used_probs[idx[m]]
                    if co_occurrence > 0:
                        pmi = np.log2((p_mn + EPS) / (p_nn * p_mm))
                        npmi[m, n] = - pmi / np.log2(p_mn + EPS)
                        npmi[n, m] = npmi[m, n]
            c_npmi = np.sum(npmi[np.triu_indices(num_top_pixels, k=1)])

            # For each selected pixel, compute the norm of the NPMI computed for all the pixels (one vs all)
            # Required for the calculation of the cosine similarity.
            norm = np.linalg.norm(npmi, axis=1)
            c_cos = 0
            for m in range(0, num_top_pixels, 1):
                for n in range(m + 1, num_top_pixels):
                    c_cos += np.dot(npmi[m, :], npmi[n, :]) / (norm[m] * norm[n])

            coherence_cos[k] = 2 / (num_top_pixels * (num_top_pixels - 1)) * c_cos
            coherence_npmi[k] = 2 / (num_top_pixels * (num_top_pixels - 1)) * c_npmi

        return coherence_npmi, coherence_cos


    # NOTE vectorised version of jb's original code, NEEDS VALIDATION
    def sort_components(self, threshold: Number = None, return_freqs: bool = False):
        """
        Sort components by representativity based on a weight threshold.

        Parameters
        ----------
        threshold : float, optional
            Minimum weight above which a component is considered present. If None, uses the model's default threshold.
        return_freqs : bool, optional
            Whether to return the frequency of each component across all samples.

        Returns
        -------
        np.ndarray
            Sorted indices of components (most to least representative).
        np.ndarray, optional
            Frequencies of each component if `return_freqs` is True.

        Examples
        --------
        >>> idxs = model.sort_components(threshold=0.3)
        >>> idxs, freqs = model.sort_components(threshold=0.3, return_freqs=True)
        """

        if threshold is None:
            threshold = self._extract_threshold(self.decomposition)
        else:
            assert isinstance(threshold, Number)
        
        reprez = self.representativity(threshold)
        idxs = np.argsort(reprez)[::-1]
        freqs = reprez / self.num_items

        if return_freqs:
            return idxs, freqs
        return idxs


    # TODO figure out a way to have training or testing data ready to compute coherence
    # def coherence(self, method: str = "all", num_top_pixels : int = None):
    #     """
    #     Compute coherence metrics (NPMI, cosine, uMass) for topic components.

    #     Parameters
    #     ----------
    #     method : str, optional
    #         Which coherence metric to compute: "npmi-cos", "umass", or "all" (default).
    #     num_top_pixels : int, optional
    #         Number of top pixels to use per component. Defaults to 10% of component length.

    #     Returns
    #     -------
    #     tuple
    #         Coherence scores depending on selected method:
    #         - If "npmi-cos": (npmi, cosine)
    #         - If "umass": (umass,)
    #         - If "all": ((npmi, cosine), umass)

    #     Examples
    #     --------
    #     >>> model.coherence(method="npmi-cos")
    #     """

    #     assert isinstance(method, str) and method.lower() in ["npmi-cos", "umass", "all"]
    #     assert self.data_training is not None
    #     if num_top_pixels is None:
    #         num_top_pixels = self.components.shape[1] // 10
    #     else:
    #         isinstance(num_top_pixels, int)

    #     out = []
    #     if method in ["npmi-cos", "all"]:
    #         out.append(self._compute_coherence_npmi_cos(components=self.components,
    #                                                     data_used=self.data_training,
    #                                                     num_top_pixels=num_top_pixels,
    #                                                     randomize_data=False))
    #     if method in ["umass", "all"]:
    #         out.append(self._compute_coherence_umass(components=self.components,
    #                                                  data_used=self.data_training,
    #                                                  num_top_pixels=num_top_pixels,
    #                                                  randomize_data=False))

    #     return tuple(out)


    # TODO returning mean divergence for decomposition, sorting the matrix by least to most distant?
    def divergence(self, metric="jensenshannon", path_to_save:str = ""):
        """
        Compute the pairwise divergence between all components.

        Optionally, plot and save the divergence matrix as a heatmap.

        Parameters
        ----------
        metric : str, optional
            a metric accepted by `scipy.spatial.distance.pdist`, default 'jensenshannon'
        path_to_save : str, optional
            File path to save the plotted heatmap (must end with .svg or .png).

        Returns
        -------
        np.ndarray
            Pairwise divergence matrix as a condensed distance vector or square matrix.

        Examples
        --------
        >>> div = model.divergence()
        >>> model.divergence("divergence_plot.svg")
        """

        wflag = self._set_figsave_flag(path_to_save)

        pw_divergence = pdist(self.components, metric=metric)

        if wflag:
            import matplotlib.pyplot as plt
            assert path_to_save.endswith((".svg", ".png"))
            pw_divergence = squareform(pw_divergence)
            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(8)
            im = ax.imshow(pw_divergence, cmap="Blues")
            cbar = fig.colorbar(im, shrink=0.85)
            cbar.set_label(f"{metric} distance", labelpad=6)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ticks_labels = np.arange(self.num_components)
            ax.set_xticks(ticks=ticks_labels, labels=ticks_labels);
            ax.set_yticks(ticks=ticks_labels, labels=ticks_labels);
            ax.tick_params("x", labelsize=10)
            ax.tick_params("y", labelsize=10)
            ax.set_xlabel("Component #", labelpad=10)
            ax.set_ylabel("Component #", labelpad=10)
            plt.tight_layout()
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)

        return pw_divergence


    # TODO add common_cbar in plotting
    def plot_components(self, normalized: bool = True, ordered: bool = False, 
                        path_to_save: str = "", form: str = "triangle", cmap: str = "Reds", **kwargs):
        """
        Plot all component matrices as triangular or square heatmaps.

        Parameters
        ----------
        normalized : bool, optional
            Whether to plot normalized component weights (default: False).
        ordered : bool, optional
            Whether to sort components by representativity before plotting (default: False).
        path_to_save : str, optional
            Path to save the figure. If empty, the figure is not saved.
        form : str, optional
            "triangle" or "square" heatmap style.
        cmap : str, optional
            Colormap to use for visualization (default: "Reds").

        Returns
        -------
        plt.Figure
            The matplotlib figure object.

        Examples
        --------
        >>> model.plot_components(normalized=True, ordered=True, path_to_save="components.svg")
        """

        wflag = self._set_figsave_flag(path_to_save)

        map_form_fn = {"triangle": vzl.plot_matrices_triangle, "square": vzl.plot_matrices_square,}

        reprez = self.representativity(threshold=kwargs.get("threshold", (1 / self.num_components) + 0.1))
        
        if ordered:
            idxs = self.sort_components(threshold=kwargs.get("threshold", (1 / self.num_components) + 0.1))
            reprez = reprez[idxs]
            items_to_plot = self.components_norm[idxs,:] if normalized else self.components[idxs,:]
        else:
            items_to_plot = self.components_norm if normalized else self.components

        idxs_comps = np.arange(self.num_components)
        titles = [f"T{i}  |  $\\rho = ${r*100:.2f}%" \
                  for i, r in zip(idxs_comps[idxs] if ordered else idxs_comps, reprez)]
        fig = map_form_fn[form](items=items_to_plot,
                                titles=titles,
                                cmap=cmap, cbar_label="norm. pseudocount" if normalized else "pseudocount")

        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig


    def plot_decomposition(self, ordered: bool = True,
                           path_to_save: str = "", cmap: str = "Blues"):
        """
        Plot the decomposition matrix as a heatmap (samples x components).

        Parameters
        ----------
        ordered : bool, optional
            Whether to sort components before plotting (not currently implemented).
        path_to_save : str, optional
            Path to save the figure.
        cmap : str, optional
            Colormap for the heatmap (default: "Blues").

        Returns
        -------
        plt.Figure
            The matplotlib figure object.

        Examples
        --------
        >>> model.plot_decomposition(path_to_save="decomposition.svg")
        """

        import seaborn as sns

        wflag = self._set_figsave_flag(path_to_save)

        fig = plt.figure(figsize=(8, 10), dpi=300)
        ax = fig.add_subplot(1, 1, 1)

        reprez = self.representativity(threshold=(1 / self.num_components) + 0.1)
        idxs = np.argsort(reprez)[::-1]

        sns.heatmap(self.decomposition[:, idxs] if ordered else self.decomposition, cmap=cmap, ax=ax)
        plt.xlabel("Topic #")
        plt.ylabel("Proximity matrix #")

        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig


    def embed_decomposition(self,
                            dir_to_save: str,
                            method: str = "UMAP",
                            cluster: bool = False,
                            save_models: bool = True,
                            **kwargs):
        """
        Embed the decomposition matrix using PCA, UMAP, or t-SNE and optionally cluster the result.

        Parameters
        ----------
        dir_to_save : str
            Directory path to save the embedding plot and models.
        method : str, optional
            Embedding method to use: "PCA", "UMAP", or "TSNE" (default: "UMAP").
        cluster : bool, optional
            Whether to perform clustering using HDBSCAN on the embedding (default: False).
        save_models : bool, optional
            Whether to save the embedding and clustering models (default: True).
        **kwargs : dict
            Additional parameters for embedding (e.g., `num_neighbors`, `min_dist`, etc.).

        Returns
        -------
        pd.DataFrame
            Embedded coordinates with optional cluster labels.

        Examples
        --------
        >>> df = model.embed_decomposition("results/", method="umap", cluster=True,
        ...                                num_neighbors=15, min_dist=0.1)
        """

        assert isinstance(dir_to_save, str) and path.isdir(dir_to_save)
        assert isinstance(method, str)
        assert isinstance(cluster, bool)

        import matplotlib.pyplot as plt
        import seaborn as sns

        method = method.strip().lower()
        assert method in ["pca", "umap", "tsne"]
        method_label = method.upper()   # for labeling method in plots and DataFrame columns

        num_components_embedding = kwargs.get("num_components", 2)

        if method == "umap":
            try:
                from umap import UMAP
                if cluster:
                    from hdbscan import HDBSCAN
            except ImportError:
                raise
            # UMAP parameter handling
            try:
                metric = kwargs.get("metric", "cosine")
                num_neighbors_embedding = kwargs["num_neighbors"]
                min_dist_embedding = kwargs["min_dist"]
                assert isinstance(num_neighbors_embedding, int)
                assert isinstance(min_dist_embedding, float)
            except KeyError:
                print(f"Parameter values for {method} not found, supply using a dictionary")
                raise
            except AssertionError:
                print("Parameter values not suitable")
                raise
            # UMAP model training and data embedding
            reducer = UMAP(
                n_neighbors=num_neighbors_embedding,
                min_dist=min_dist_embedding,
                metric=metric,
                n_components=num_components_embedding,
                random_state=42,
            )
            embedding = reducer.fit_transform(self.decomposition)

        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(self.decomposition)

        else:
            reducer = TSNE(n_components=2, random_state=42)
            embedding = reducer.fit_transform(self.decomposition)

        # Saving the embedding model as joblib file
        if save_models:
            jl.dump(reducer, path.join(dir_to_save, f"model_embedding_decomposition_{method}.jl"))

        # Prepare df for plotting
        embedding_df = pd.DataFrame(embedding, columns=[f"{method_label}_{i}" \
                                                        for i in range(1, num_components_embedding+1)])

        # Plotting the results
        fig, ax = plt.subplots(dpi=300)
        fig.set_figwidth(8)
        fig.set_figheight(8)
        sns.scatterplot(data=embedding_df,
                x=embedding_df.columns[0], y=embedding_df.columns[1],
                s=10, alpha=0.4, color="#216eff", edgecolor="none",
                legend=False,
                ax=ax)
        ax.set_title(method)
        fig.tight_layout()
        fig.savefig(path.join(dir_to_save, f"embedding_decomposition_{method}.png"))
        plt.close(fig)
        
        if cluster:

            import hdbscan

            # HDBSCAN parameters for the pooled components
            min_cluster_size = kwargs.get("min_cluster_size", self.decomposition.shape[0] // 40)
            min_samples = kwargs.get("min_samples", 1)
            cluster_selection_method = kwargs.get("cluster_selection_method", "eom")
            alpha = kwargs.get(alpha, 1.0)
            
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=cluster_selection_method,
                alpha=alpha,
                prediction_data=True
            )
            clusterer = clusterer.fit(embedding)
            embedding_df["cluster"] = clusterer.labels_

            # Saving the clusterer model as joblib file
            if save_models:
                jl.dump(clusterer, path.join(dir_to_save, f"model_embedding_decomposition_{method}_clusterer.jl"))

            clusters_unique = sorted(list(np.unique(clusterer.labels_).astype(int)))
            clusters_unique.append(clusters_unique.pop(0))
            clusters_unique = np.asarray(clusters_unique)
            num_clusters = len(clusters_unique)     # noise is included as a cluster

            embedding_clusters_df = embedding_df[embedding_df["cluster"] != -1].copy()
            embedding_noise_df = embedding_df[embedding_df["cluster"] == -1].copy()

            palette = sns.color_palette("tab20", n_colors=num_clusters) \
                if num_clusters <= 15 else sns.color_palette("hls", n_colors=num_clusters)
            cluster_to_color = dict(zip(clusters_unique, palette))
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
            data=embedding_clusters_df,
            x=embedding_df.columns[0], y=embedding_df.columns[1],
            hue="cluster",
            palette=cluster_to_color,
            s=60,
            legend=False
            )
            sns.scatterplot(
                data=embedding_noise_df,
                x=embedding_df.columns[0], y=embedding_df.columns[1],
                color="gainsboro",
                s=60,
                label="Noise"
            )
            # Annotate cluster numbers
            centroids = embedding_clusters_df.groupby("cluster")[[f"{method_label}_1", f"{method_label}_2"]].mean()
            for cluster_id, row in centroids.iterrows():
                plt.text(row[f"{method_label}_1"]+1, row[f"{method_label}_2"]+1, str(cluster_id),
                        fontsize=10, weight="bold", ha="center", va="center", color="black",
                        bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.3"))
            plt.tight_layout()
            plt.savefig(path.join(dir_to_save, f"embedding_decomposition_{method}_clustered.png"), dpi=300)
            plt.close()

        return embedding_df
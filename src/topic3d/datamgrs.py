#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data managers for topic decomposition input data (:mod:`topic3d.datamgrs`)
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

import h5py
import numpy as np
import pandas as pd

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
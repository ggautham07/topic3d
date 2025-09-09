#!/usr/bin/env python
"""
Full object handlers for topic decomposition data and results (:mod:`topic3d.handlers`)
=====================================================

.. authors:: jbfiche, ggautham07, marcnol

Function reference
------------------
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
    Data handler for raw data file types: .npy, .csv for Fish Omics Format (FOF) are accepted
    
    Create an instance of the object and use the method `load(path_to_file:str)` to load the data
    
    Data can be processed to produce a `ProcessedData` object with the `process()` method
    
    Note: the highest amount of information can only be captured from a FOF file
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
        """Extract traces from a FOF-CT Dataframe"""

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
        Merge multiple Numpy trace files.

        Loads the datasets and stacks them together in a single dataset. The total number of items is the sum of 
        the number of items of each dataset and the maximum number of barcodes (other coordinates are replaced by 
        NaNs)

        Parameters
        --------
        args : str
            paths to Numpy files

        Returns
        --------
        out : np.ndarray
            merged traces array

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
        Open a trace file; can either be a FOF-CT trace core file or a Numpy file.
        Former will be formatted as a pandas dataframe. Latter directly as CoordsStackArray object

        Parameters
        --------
        path_to_file : str
            path to the raw data file
        metadata : dict
            metadata to support data from Numpy. Ignored for FOF data.
            Accepted keys are "trace_ids", "bc_ids", "bc_bins". Others will be added soon
        extract_items : bool
            if True (which is default), extract traces from FOF data and set traces object
        
        Returns
        --------
        None
            class attributes are set automatically
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
        """Process the trace data and return everything as a Processed data object"""

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
        Using the genomic distance between the barcodes, compute the median 3d distance as a function of the genomic
        distance.

        @param pairwise_dist: (numpy array) pairwise distances for each individual trace
        @param gen_dist: (numpy array) 1D array indicating the genomic distance between all pairs of barcodes
        @param gen_dist_bins: (int) list of genomic distance used to build the histogram
        @param dist_threshold: (float) maximum distance above which the distance are considered unrealistic
        @param experiment: (int) indicate the number of the experiment that is being processed
        @param axis: (str) indicate which axis is being processed
        @return:
        dist: (pandas dataframe) contains all the data for the histogram
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
        Compute the genomic distance separating two consecutive barcodes (from the experiment)

        @param lib: (pandas dataframe) coordinates of the barcodes
        @param n_bins: (int) total number of barcodes of the library
        @return: locus_dist: (pandas dataframe) contains the genomic distance in kbp
        """

        assert self.bc_bins
        if not self.bc_bins_contiguity:
            warnings.warn("Note: discontiguous segments supplied")

        bins_mean_pos = np.mean(self.bc_bins, axis=1)

        return bins_mean_pos[1:] - bins_mean_pos[:-1]
    

    @staticmethod
    def plot_log_normalization_graph(dist_df:pd.DataFrame, dim:dict, saving_filename:str, normalized_data=False):
        """
        Plot in log-log scale the distance versus the genomic distance for the raw data and after normalization shift.

        @param dist_df: (pandas dataframe) contains all the histogram distance data (Euclidean and genomic)
        @param dim: (dict) dictionary indicating the name of the axis
        @param saving_filename: (str) path where to save the figure
        @param normalized_data: (bool) indicate whether the raw or normalized data should be used
        """
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axs = plt.subplots(nrows=3, ncols=1)
        for ax_name, ax in dim.items():
            if normalized_data:
                df = dist_df.loc[dist_df["axis"] == ax_name, ['log10 genomic dist (kbp)',
                                                            'log10 normalized (nm)',
                                                            'experiment']]
                sns.scatterplot(data=df, x='log10 genomic dist (kbp)', y='log10 normalized (nm)',
                                hue="experiment", ax=axs[ax])
            else:
                df = dist_df.loc[dist_df["axis"] == ax_name, ['log10 genomic dist (kbp)',
                                                            'log10 median dist (nm)',
                                                            'experiment']]
                sns.scatterplot(data=df, x='log10 genomic dist (kbp)', y='log10 median dist (nm)',
                                hue="experiment", ax=axs[ax])
        fig.set_size_inches(8, 10)
        plt.savefig(saving_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)



class ProcessedData:
    """
    Handler for processed data, both loading and saving
    
    Methods for plotting, computations can be performed on CoordsArrayStack and PWDistsArrayStack
    
    Metadata includes barcode list, their genomic bins and detection efficiencies
    """

    def __init__(self):
        self.traces = None
        self.pwds = None
        self.bc_effs = None
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
        assert path_to_file.endswith(".h5")
        assert path.isfile(path_to_file)
        with h5py.File(path_to_file, mode="r") as handler:
            datasets = {x:y[:] for x, y in handler.items()}
            attributes = {x:y[:] for x, y in handler.attrs.items()}
        self.traces = datasets.get("traces", None)
        if self.traces is not None:
            self.traces = CoordsArrayStack(self.traces)
        self.pwds = datasets.get("pwds", None)
        if self.pwds is not None:
            self.pwds = PWDistArrayStack(self.pwds)
        self.bc_effs = datasets.get("bc_effs", None)
        self.metadata = attributes


    def save(self, path_to_file: str, replace: bool = True) -> None:
        if path.isfile(path_to_file) and not replace:
            raise FileExistsError("File already exists and replace is set to False")
        attr_labels = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        print("Saving the following attributes: ", attr_labels)     # TODO change the printing to cleaner format
        with h5py.File(path_to_file, mode="w") as handler:
            for lbl in attr_labels:
                attr = getattr(self, lbl)
                try:
                    print(lbl, type(attr), attr.dtype)
                    handler.create_dataset(name=lbl, data=attr)
                    print(f"Saved {lbl} successfully")
                except AttributeError:
                    pass
                except TypeError:
                    handler.attrs[lbl] = attr
                    print(f"Saved {lbl} successfully as attribute")


#TODO
class DecompositionResult:
    """
    Decomposition results handler, loads and saves results, used to quickly perform further analyses on them
    
    Methods for post-processing described to compute representativity, reconstruction, coherence and divergence
    
    Embedding and clustering of decomposition
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
        Save the decomposition results to a joblib file for the model, JSON file for metadata and a 
        HDF5 file for everything else
        """
        
        with h5py.File(path.join(dir_to_save, "results.h5"), mode="w") as handler:
            handler.create_dataset(name="components", data=self.components)
            handler.create_dataset(name="decomposition", data=self.decomposition)
            handler.create_dataset(name="data_idxs/training", data=self.data_training)
            handler.create_dataset(name="data_idxs/testing", data=self.data_testing)
            handler.attrs = self.metadata
        
        jl.dump(self.model, path.join(dir_to_save, "model_lda.jl"))


    def reconstruction(self, normalize=False):
        """
        Reconstruct individual contact vectors using the captured components and the decomposition matrix.
        """

        if normalize:
            components_norm = self.components / self.components.sum(axis=1)[:, np.newaxis]
            return self.decomposition @ components_norm
        return self.decomposition @ self.components
    
    
    # TODO automatic extraction of threshold from a weights distribution, preferably multiple options
    @staticmethod
    def _extract_threshold(X):
        return np.quantile(X, 10)

    def representativity(self, threshold=None):
        if threshold is None:
            threshold = self._extract_threshold(self.decomposition)
        return np.mean(self.decomposition >= threshold, axis=0)
    
       
    @staticmethod
    def _compute_coherence_umass(components, data_used, num_top_pixels=10, randomize_data=False):
        """
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

        num_components = len(components)
        coherence_npmi = np.zeros((num_components,))
        coherence_cos = np.zeros((num_components,))
        data_used = np.asarray(data_used)
        num_data_used = data_used.shape[0]
        data_used_freq = np.sum(data_used, axis=0)

        EPS = 10**-5    # value of epsilon is kept constant

        for k in range(num_components):
            comp = components[k, :]
            if randomize_data is False:
                idx = np.argsort(comp)[::-1]  # return the bin index in descending order
            else:
                idx = np.random.permutation(comp.shape[0])
            idxs_top = idx[:num_top_pixels]

            # Compute the NPMI for all pairs of selected bins
            npmi = np.zeros((num_top_pixels, num_top_pixels))
            for m in range(num_top_pixels):
                for n in range(m+1, num_top_pixels):
                    i, j = idxs_top[m], idxs_top[n]
                    co_occurrence = np.sum((data_used[:, i] >= 0.75) & (data_used[:, j] >= 0.75))
                    p_mn = co_occurrence / num_data_used
                    p_nn = data_used_freq[idx[n]] / num_data_used
                    p_mm = data_used_freq[idx[m]] / num_data_used
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
        Sort the components captured from a decomposition by its representativity.

        Parameters
        --------
        threshold: float
            minimum weight threshold above which the component is considered relevant (or representative)
        return_freqs: bool
            return the frequencies of each component sorted by representativity

        Returns
        --------
        freqs : numpy.ndarray
            frequency of finding each component within the data (after sorting)
        idxs : numpy.ndarray
            indices used to sort the components by representativity
        """

        if threshold is None:
            threshold = self._extract_threshold(self.decomposition)
        else:
            assert isinstance(threshold, Number)
        
        representativity = np.sum(self.decomposition > threshold, axis=0)
        idxs = np.argsort(representativity)[::-1]
        freqs = representativity / self.num_items

        if return_freqs:
            return idxs, freqs
        return idxs
    

    # TODO returning mean divergence for decomposition, sorting the matrix by least to most distant?
    def divergence(self, path_to_save:str = None):

        pw_divergence = pdist(self.components, metric="jensenshannon")
        
        if path_to_save is not None:
            import matplotlib.pyplot as plt
            assert isinstance(path_to_save, str)
            assert path_to_save.endswith((".svg", ".png"))
            pw_divergence = squareform(pw_divergence)
            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(8)
            im = ax.imshow(pw_divergence, cmap="Blues")
            cbar = fig.colorbar(im, shrink=0.85)
            cbar.set_label("Jensen-Shannon divergence", labelpad=6)
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
            fig.savefig(path_to_save, dpi=300)
        
        return pw_divergence


    def coherence(self, method: str = "all", num_top_pixels : int = None):

        assert isinstance(method, str) and method.lower() in ["npmi-cos", "umass", "all"]
        assert self.data_training is not None
        if num_top_pixels is None:
            num_top_pixels = self.components.shape[1] // 10
        else:
            isinstance(num_top_pixels, int)

        out = []
        if method in ["npmi-cos", "all"]:
            out.append(self._compute_coherence_npmi_cos(components=self.components,
                                                        data_used=self.data_training,
                                                        num_top_pixels=num_top_pixels,
                                                        randomize_data=False))
        if method in ["umass", "all"]:
            out.append(self._compute_coherence_umass(components=self.components,
                                                     data_used=self.data_training,
                                                     num_top_pixels=num_top_pixels,
                                                     randomize_data=False))

        return tuple(out)


    # TODO add common_cbar in plotting
    def plot_components(self, normalized: bool = False, ordered: bool = False, 
                        path_to_save: str = "", form: str = "triangle", cmap: str = "Reds"):

        wflag = self._set_figsave_flag(path_to_save)

        map_form_fn = {"triangle": vzl.plot_matrices_triangle, "square": vzl.plot_matrices_square,}

        reps = np.mean(self.decomposition > 0.3, axis=0)

        if ordered:
            idxs = self.sort_components(threshold=0.3)
            items_to_plot = self.components_norm[idxs,:] if normalized else self.components[idxs,:]
        else:
            items_to_plot = self.components_norm if normalized else self.components

        titles = [f"T{i}  |  $\\rho = ${r}" for i, r in enumerate(reps)]
        fig = map_form_fn[form](items=items_to_plot,
                                titles=titles,
                                cmap=cmap, cbar_label="norm. pseudocount" if normalized else "pseudocount")

        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig
    

    # TODO
    def plot_decomposition(self, path_to_save: str = "", cmap: str = "Reds"):
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(1, 1, 1)
        # sns.heatmap(lda_decomposition_ordered, cmap="coolwarm", ax=ax)
        # ax.get_xaxis().set_visible(False)
        # ax.get_xaxis().set_visible(True)
        # saving_filename = path.join(training_path, f'topic_decomposition.png')
        # plt.savefig(saving_filename, dpi=150, bbox_inches='tight')
        # plt.close(fig)
        return


    def embed_decomposition(self,
                            dir_to_save: str,
                            method: str = "UMAP",
                            cluster: bool = False,
                            save_models: bool = True,
                            **kwargs):

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
        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(8)
        sns.scatterplot(data=embedding_df,
                x=embedding_df.columns[0], y=embedding_df.columns[1],
                s=10, alpha=0.4, color="#216eff", edgecolor="none",
                legend=False,
                ax=ax)
        ax.set_title(method)
        fig.tight_layout()
        fig.savefig(path.join(dir_to_save, f"embedding_decomposition_{method}.png"), dpi=300)
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
                        fontsize=10, weight='bold', ha='center', va='center', color='black',
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))
            plt.tight_layout()
            plt.savefig(path.join(dir_to_save, f"embedding_decomposition_{method}_clustered.png"), dpi=300)
            plt.close()

        return embedding_df
#!/usr/bin/env python
"""
Full object handlers for topic decomposition data and results (:mod:`topic3d.handlers`)
=====================================================

.. authors:: jbfiche, gsgautham98, marcnol

Function reference
------------------
"""


# =============================================================================
# IMPORTS
# =============================================================================

from numbers import Number
from os import path
from io import StringIO
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib as jl
from datatypes import CoordsArrayStack, PairwiseArrayStack



# =============================================================================
# CLASSES
# =============================================================================


class RawData:
    """ Data handler for raw data file types: .npy, .csv for Fish Omics Format (FOF) are accepted
    
    Create an instance of the object and use the method `load(path_to_file:str)` to load the data
    
    Data can be processed to produce a `ProcessedData` object with the `process()` method
    
    Note: the highest amount of information can only be captured from a FOF file
    """

    def __init__(self):
        self.library = None
        self.unit = None
        self.assembly = None
        self.trace_ids = None
        self.traces = None
        self.num_traces = None
        self.num_bc = None
        self.bc_ids = None
        self.bc_bins = None


    def __str__(self):
        return (f"FOF-CT data: {self.library is not None}\n"
                f"Trace array: {self.traces is not None}\n"
                f"Genome assembly: {self.assembly}\n"
                f"Measurement unit: {self.unit}\n"
                f"Number of barcodes: {self.num_bc}\n"
                f"Number of traces: {self.num_traces}\n"
                )
    

    @staticmethod
    def _extract_header_row_from_fof(full_seq:str, sub_seq:str) -> list:
        """ Internal function used to extract header rows from Fish Omics Format (FOF) data
        Finds the first occurrence of sub_seq in full_seq and splits it at the first newline
        """

        return full_seq[full_seq.index(sub_seq):].split("\n", maxsplit=1)
    

    def _extract_traces_from_fof(self):
        """Extract traces from FOF-CT Dataframe"""

        traces = np.empty((self.num_traces, self.num_bc, 3))
        traces[:] = np.nan
        # create the trace from the coordinates
        for j, curr_trace_id in enumerate(self.trace_ids):
            curr_data = self.library[self.library["Trace_ID"] == curr_trace_id]
            barcodes = [self.bc_ids[x] for x in curr_data["Chrom_Start"].to_list()]
            coords = curr_data[["X", "Y", "Z"]].to_numpy()
            traces[j, barcodes] = coords[:]
        
        return traces


    # TODO path_to_file can also take a list of files with the same extension/format, and merge the data
    def load(self, path_to_file:str):
        """ Open a trace file; can either be a FOF-CT trace core file or a Numpy file.
        Former will be formatted as a pandas dataframe.
        @param file_path: (str) path to the file
        @return: data: (pd dataframe) contains the XYZ localizations and barcode positions for all localizations
        @return: unit: (str) indicate which unit was used for the data
        """

        assert path.isfile(path_to_file)

        if path_to_file.endswith(".npy"):
            self.traces = CoordsArrayStack(np.load(path_to_file, allow_pickle=True))
            assert self.traces.data.shape[-1] == 3
            del self.library

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
            self.library = data
            self.unit = unit
            self.assembly = assembly

            self.trace_ids = np.unique(self.library["Trace_ID"])
            self.num_traces = len(self.trace_ids)

            chrom_starts = np.unique(data["Chrom_Start"])
            chrom_ends = np.unique(data["Chrom_End"])
            assert len(chrom_starts) == len(chrom_ends)
            self.num_bc = len(chrom_starts)
            self.bc_ids = list(range(self.num_bc))
            self.bc_bins = dict(zip(self.bc_ids, zip(chrom_starts, chrom_ends)))

            # Extraction of traces
            self.traces = CoordsArrayStack(self._extract_traces_from_fof())
        
        else:
            raise Exception("Invalid file extension. Only npy and csv files accepted")


    def process(self):
        """Process the trace data and return everything as a Processed data object"""

        assert self.traces is not None
        prc_data = ProcessedData()
        prc_data.traces = self.traces
        prc_data.bc_effs = self.traces.efficiency(by="barcode")
        prc_data.pwds = self.traces.pwds()
        if self.library is None:
            prc_data.metadata = None
        else:
            self.bc_bin_sizes = [end - start for start, end in self.bc_bins.values()]
            prc_data.metadata = {"bc_bins": self.bc_bins,
                                 "bc_bin_sizes": self.bc_bin_sizes}
        return prc_data
    
    
    def compute_3d_vs_genomic_dists(self, max_realistic_3d_dist : Number):
        """Using the genomic distance between the barcodes, compute the median 3d distance as a function of the genomic
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
        bins_genomic = np.linspace(start=np.min(pw_dists_genomic), stop=np.max(pw_dists_genomic)+1, num=20)
        dists_3d_vs_genomic = np.zeros((len(bins_genomic), 3))
        for bin in range(len(bins_genomic) - 1):
            idx = np.where((pw_dists_genomic_3d_combined[:, 0] >= bins_genomic[bin]) & \
                           (pw_dists_genomic_3d_combined[:, 0] < bins_genomic[bin+1]))
            dists_3d_vs_genomic[bin, :] = [bins_genomic[bin], 
                                           np.nanmedian(pw_dists_genomic_3d_combined[idx, 1]), 
                                           len(idx[0])]

        # convert to dataframe
        results = pd.DataFrame()
        results["genomic distance"] = dists_3d_vs_genomic[:, 0]
        results["log10 genomic dist"] = np.log10(dists_3d_vs_genomic[:, 0])
        results["median euclidean distance"] = dists_3d_vs_genomic[:, 1]
        results["log10 median dist"] = np.log10(dists_3d_vs_genomic[:, 1])
        results["n_data"] = dists_3d_vs_genomic[:, 2]

        return results



class ProcessedData:
    """ Handler for processed data, both loading and saving
    
    Methods for plotting, computations can only be performed on RawData objects or separately using array functions
    
    Metadata includes barcode list, their genomic bins and detection efficiencies
    """

    def __init__(self):
        self.traces = None
        self.pwds = None
        self.bc_effs = None
        self.metadata = None


    def __str__(self):
        return (f"traces: {self.traces is not None}\n"
                f"pwds: {self.pwds is not None}\n"
                f"bc_effs: {self.bc_effs is not None}\n"
                f"metadata: {self.metadata}\n"
                )


    def load(self, path_to_file:str):
        # assert path.isfile(path_to_file)
        assert path_to_file.endswith(".h5")
        with h5py.File(path_to_file, mode="r") as handler:
            datasets = {x:y[:] for x, y in handler.items()}
            attributes = {x:y[:] for x, y in handler.attrs.items()}
        self.traces = datasets.get("traces", None)
        if self.traces is not None:
            self.traces = CoordsArrayStack(self.traces)
        self.pwds = PairwiseArrayStack(datasets.get("pwds", None))
        self.bc_effs = datasets.get("bc_effs", None)
        self.metadata = attributes


    def save(self, path_to_file:str, replace=True):
        if path.isfile(path_to_file) and not replace:
            raise FileExistsError("File already exists and replace is set to False")
        attr_labels = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        print(attr_labels)
        with h5py.File(path_to_file, mode="w") as handler:
            for al in attr_labels:
                attr = getattr(self, al)
                try:
                    print(al, type(attr), attr.dtype)
                    handler.create_dataset(name=al, data=attr)
                    print(f"Saved {al} successfully")
                except AttributeError:
                    pass
                except TypeError:
                    handler.attrs[al] = attr
                    print(f"Saved {al} successfully as attribute")
    

    def plot_contacts_avg(self, threshold):
        contacts = np.where(np.isnan(self.pwds), np.nan, self.pwds < threshold)
        contacts_ensemble = squareform(np.nanmean(contacts, axis=0))
        np.fill_diagonal(contacts_ensemble, 1)
        vmin = np.nanmin(contacts_ensemble)
        vmax = np.quantile(contacts_ensemble, 0.99)
        if np.isnan(vmax):
            vmax = np.nanmax(contacts_ensemble)
        plt.figure(figsize=(8, 6))
        im = plt.imshow(contacts_ensemble, cmap="Reds", vmin=vmin, vmax=vmax)
        plt.colorbar(label="contact freq")
        plt.tight_layout()
        return im
    
    # TODO NOTE standardise use of sns for all plots to maintain consistency
    def plot_3d_vs_genomic_dists():
        """Plot the 3D distance as a function of the genomic distance"""

    # TODO
    def plot_pwds_distribution():
        """"""
    
    # TODO
    def plot_bc_effs():
        """"""



class DecompositionResult:
    """ Decomposition results handler, loads and saves results, used to quickly perform further analyses on them
    
    Methods for post-processing described to compute representativity, reconstruction, coherence and divergence
    
    Embedding and clustering of decomposition
    """

    def __init__(self, model:LatentDirichletAllocation, decomposition:np.ndarray, **kwargs):
        self.model = model
        self.decomposition = decomposition
        self.components = model.components_
        self.num_components = len(self.components)
        self.metadata = {}
        
        # TODO store original dataset indices of the training and testing sets using self.data_idxs = kwargs.get()
        self.data_training = kwargs.get("data_training", None)
        if self.data_training is not None:
            assert isinstance(self.data_training, (PairwiseArrayStack, np.ndarray))
        self.data_testing = kwargs.get("data_testing", None)
        if self.data_testing is not None:
            assert isinstance(self.data_testing, (PairwiseArrayStack, np.ndarray))


    # TODO JSON file for the metadata, optional additional copy in HDF5 file
    def save(self, path_to_save_dir:str):
        """Save the decomposition results to a joblib file for the model, json file for metadata and a 
        HDF5 file for everything else"""
        
        with h5py.File(path.join(path_to_save_dir, "results.h5"), mode="w") as handler:
            handler.create_dataset(name="components", data=self.components)
            handler.create_dataset(name="decomposition", data=self.decomposition)
            handler.create_dataset(name="data_training", data=self.data_training)
            handler.create_dataset(name="data_testing", data=self.data_testing)
            handler.attrs = self.metadata
        
        import joblib
        joblib.dump(self.model, path.join(path_to_save_dir, "model_lda.jl"))


    def reconstruction(self):
        """Reconstruct single-cell maps using components and decomposition"""
        
        return self.decomposition @ self.components
    
    
    @staticmethod
    def _compute_coherence_umass(components, data_used, num_top_pixels, randomize_data=False):
        """ Compute the umass coherence for a selected set of topics and raw data. The calculation is performed by selecting
        the top words of each topic. A default value of 10 words (or bins) is indicated.

        @param num_top_pixel: (int) number of top-words (bins) to select for computing the coherence
        @param n_topics: (int) number of topics used for LDA decomposition
        @param topics: (numpy array) list all the topics processed during LDA decomposition
        @param raw_data: (numpy array) contain the contact vectors for each single cells (previously selected for LDA)
        @param random: (boolean) indicate whether the data should be randomized (in order to compute a baseline)
        @return coherence_umass: (numpy array) contain the coherence values computed for each topic
        """

        num_topics = len(components)
        coherence_umass = np.zeros((num_topics,))
        data_frequency = np.sum(data_used, axis=0)

        for n_topic in range(num_topics):
            topic = components[n_topic, :]
            if randomize_data:
                idx = np.random.permutation(topic.shape[0])
            else:
                idx = np.argsort(topic, )[::-1]  # return the bin index in descending order
            top_indices = idx[:num_top_pixels]
            c_umass = 0
            for m in range(1, num_top_pixels):
                for n in range(0, m):
                    d_mn = np.sum((data_used[:, top_indices[m]] == 1) & (data_used[:, top_indices[n]] == 1))
                    d_nn = data_frequency[top_indices[n]]
                    if d_nn > 0:
                        c_umass += np.log2((d_mn + 10 ** -6) / d_nn)
            coherence_umass[n_topic] = 2 / (num_top_pixels * (num_top_pixels - 1)) * c_umass

        return coherence_umass


    @staticmethod
    def _compute_coherence_npmi_cos(components, data_used, num_top_pixels=10, randomize_data=False):
        """ Compute coherence for a selected set of topics and raw data using two methods. The calculation is based on the
        normalized pointwise mutual information and is performed by selecting the top words of each topic. A default value
        of 10 words (or bins) is indicated.

        @param num_top_pixels: (int) number of top-words (bins) to select for computing the coherence
        @param n_topics: (int) number of topics used for LDA decomposition
        @param topics: (numpy array) list all the topics processed during LDA decomposition
        @param raw_data: (numpy array) contain the contact vectors for each single cells (previously selected for LDA)
        @param random: (boolean) indicate whether the data should be randomized (in order to compute a baseline)
        @return coherence_npmi: (numpy array) contain the npmi coherence values computed for each topic
                coherence_cos: (numpy array) contain the coherence values computed for each topic based on cosine similarity
        """

        num_topics = len(components)
        coherence_npmi = np.zeros((num_topics,))
        coherence_cos = np.zeros((num_topics,))
        n_data = data_used.shape[0]
        data_frequency = np.sum(data_used, axis=0)

        for k in range(num_topics):
            topic = components[k, :]
            if randomize_data is False:
                idx = np.argsort(topic, )[::-1]  # return the bin index in descending order
            else:
                idx = np.random.permutation(topic.shape[0])

            top_idx = idx[:num_top_pixels]

            # compute the NPMI for all pairs of selected bins
            npmi = np.zeros((num_top_pixels, num_top_pixels))
            for m in range(num_top_pixels):
                for n in range(m+1, num_top_pixels):
                    i, j = top_idx[m], top_idx[n]
                    co_occurrence = np.sum((data_used[:, i] >= 0.8) & (data_used[:, j] >= 0.8))
                    p_mn = co_occurrence / n_data
                    p_nn = data_frequency[idx[n]] / n_data
                    p_mm = data_frequency[idx[m]] / n_data
                    if co_occurrence > 0:
                        pmi = np.log2((p_mn + 10 ** -6) / (p_nn * p_mm))
                        npmi[m, n] = - pmi / np.log2(p_mn + 10**-6)
                        npmi[n, m] = npmi[m, n]

            c_npmi = np.sum(npmi[np.triu_indices(num_top_pixels, k=1)])
            # for each selected bins (top words), compute the norm of the NPMI computed for all the bins (one top bin versus
            # all). This is required for the calculation of the cosine similarity.
            norm = np.linalg.norm(npmi, axis=1)
            c_cos = 0
            # based on the previous calculation, compute the NPMI and COS coherence
            for m in range(0, num_top_pixels, 1):
                for n in range(m + 1, num_top_pixels):
                    # c_npmi += npmi[m, n]
                    c_cos += np.dot(npmi[m, :], npmi[n, :]) / (norm[m] * norm[n])

            coherence_cos[k] = 2 / (num_top_pixels * (num_top_pixels - 1)) * c_cos
            coherence_npmi[k] = 2 / (num_top_pixels * (num_top_pixels - 1)) * c_npmi

        return coherence_cos, coherence_npmi


    # TODO automatic extraction of threshold from a weights distribution, preferably multiple options
    @staticmethod
    def _extract_threshold(X):
        return X


    # TODO adding automatic threshold extraction using internal function
    def representativity(self, threshold=None):
        if threshold is None:
            threshold = self._extract_threshold(self.decomposition)
        return np.mean(self.decomposition > threshold, axis=0)
    

    # TODO returning mean divergence for decomposition, sorting the matrix by least to most distant?
    def divergence(self, path_to_save:str = None):
        """Compute the Jensen-Shannon divergence between each pair of components captured from a training
        and prepare a pairwise matrix plot if `path_to_save` is provided
        """

        pw_divergence = pdist(self.components, metric="jensenshannon")
        
        if path_to_save is not None:
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


    # TODO a more optimum way of structuring the function, using a dict of keys and methods, maybe?
    def coherence(self, method: str = "all"):
        """Compute the coherence of the decomposition
        
        Using NPMI, cosine or UMASS, or all of them
        """

        assert isinstance(method, str) and method.lower() in ["npmi-cos", "umass", "all"]
        assert self.data_training is not None

        if method == "npmi-cos":
            return self._compute_coherence_npmi_cos(components=self.components,
                                                    data_used=self.data_training,
                                                    num_top_pixels=self.components.shape[1]//10,
                                                    randomize_data=False)
        elif method == "umass":
            return self._compute_coherence_umass(components=self.components,
                                                 data_used=self.data_training,
                                                 num_top_pixels=self.components.shape[1]//10,
                                                 randomize_data=False)
        else:
            coherence_npmi_cos = self._compute_coherence_npmi_cos(components=self.components,
                                                                  data_used=self.data_training,
                                                                  num_top_pixels=self.components.shape[1]//10,
                                                                  randomize_data=False)
            coherence_umass = self._compute_coherence_umass(components=self.components,
                                                            data_used=self.data_training,
                                                            num_top_pixels=self.components.shape[1]//10,
                                                            randomize_data=False)
            return coherence_npmi_cos, coherence_umass


    # TODO t-SNE method, embedding plot with clusters needs to be saved, HDBSCAN parameter handling and model saving
    def embed_decomposition(self, path_to_save_dir: str, method: str = "UMAP", cluster: bool = False, **kwargs):
        """Embed the decomposition using Principal Component Analysis (PCA), UMAP, or t-SNE and cluster"""

        assert isinstance(path_to_save_dir, str) and path.isdir(path_to_save_dir)
        assert isinstance(method, str)
        assert isinstance(cluster, bool)

        method = method.strip().lower()
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

        # Prepare df for plotting
        embedding_df = pd.DataFrame(embedding, columns=[f"{method_label}_{i}" \
                                                        for i in range(1, num_components_embedding+1)])
        # Saving the model as joblib file
        jl.dump(reducer, path.join(path_to_save_dir, f"model_decomposition_{method}.jl"))

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
        fig.savefig(path.join(path_to_save_dir, f"embedding_decomposition_{method}.png"), dpi=300)
        plt.close(fig)
        
        if cluster:
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
            plt.savefig(path.join(path_to_save_dir, f"embedding_decomposition_{method}_clustered.png"), dpi=300)
            plt.close()

        return embedding_df
#!/usr/bin/env python
"""
Full object handler for topic decomposition data and results (:mod:`topic3d.handlers`)
=====================================================

.. authors:: gsgautham98, jbfiche, marcnol

Function reference
------------------
"""

from os import path
from io import StringIO
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.linalg import issymmetric
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib as jl
from hdbscan import HDBSCAN

from datatypes import CoordsArrayStack, PairwiseArrayStack

####################


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


    def __str__(self):
        return (f"FOF-CT data: {self.library is not None}\n"
                f"Trace array: {self.traces is not None}\n"
                f"Genome assembly: {self.assembly}\n"
                f"Measurement unit: {self.unit}\n"
                f"Number of barcodes: {self.num_bc}\n"
                f"Number of traces: {self.num_traces}\n"
                )
    
    @staticmethod
    def _findnsplit(full_seq:str, sub_seq:str) -> list:
        """ Find the first occurrence of a string within another and split into two at a newline"""
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
            table_header_raw, table_data_raw = self._findnsplit(contents,
                                                                "##columns")  # split the header from the trace data
            colnames = table_header_raw[table_header_raw.index("(")+1:table_header_raw.index(")")].split(",")
            colnames = [c.strip() for c in colnames]  # get column names
            data = pd.read_csv(StringIO(table_data_raw), sep=",", header=None)  # get trace data
            data.columns = colnames  # assign column names to trace table

            # Locating other key metadata
            assembly = self._findnsplit(contents, "##genome_assembly")[0].split("=")[1]
            unit = self._findnsplit(contents, "##XYZ_unit")[0].split("=")[1]

            # Variable assignment
            self.library = data
            self.unit = unit
            self.assembly = assembly

            self.trace_ids = np.unique(self.library["Trace_ID"])
            self.num_traces = len(self.trace_ids)

            starts = np.unique(data["Chrom_Start"])
            self.bc_ids = dict(zip(starts, range(len(starts))))
            self.num_bc = len(starts)

            # Extraction of traces
            self.traces = CoordsArrayStack(self._extract_traces_from_fof())
        
        else:
            raise Exception("Invalid file extension. Only npy and csv files accepted")


    def process(self):
        """Process the trace data and return everything as a Processed data object"""

        assert self.traces is not None
        prc_data = ProcessedData()
        prc_data.traces = self.traces
        prc_data.bc_effs = self.traces.bc_efficiency()
        prc_data.pwds = self.traces.pwds()
        prc_data.bc_list = self.bc_ids
        if self.library is None:
            prc_data.metadata = None
        else:
            prc_data.metadata = {}

        return prc_data
    
    
    def _compute_3d_vs_genomic_distance(pairwise_dist, gen_dist, gen_dist_bins, dist_threshold=1., experiment=None,
                                    axis='X'):
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
        n_maps = pairwise_dist.shape[0]

        # compile all the distance map and aggregate genetic and euclidean distances
        merged_list = []
        for i in range(n_maps):
            merged = np.column_stack((gen_dist, pairwise_dist[i, :]))
            merged_list.append(merged)

        merged_dist = np.array(merged_list)
        merged_dist = np.reshape(merged_dist, (n_maps * len(gen_dist), 2))
        merged_dist[merged_dist[:, 1] > dist_threshold, 1] = np.nan

        # to make the plot easier to read, data are binned
        gen_bin = np.linspace(start=np.min(gen_dist), stop=np.max(gen_dist)+1, num=gen_dist_bins)
        d_vs_gen_dist = np.zeros((len(gen_bin), 3))
        for bin in range(len(gen_bin) - 1):
            idx = np.where((merged_dist[:, 0] >= gen_bin[bin]) & (merged_dist[:, 0] < gen_bin[bin+1]))
            d_vs_gen_dist[bin, :] = [gen_bin[bin], np.nanmedian(merged_dist[idx, 1]), len(idx[0])]

        # convert to dataframe
        dist = pd.DataFrame()
        dist['genomic distance (kbp)'] = d_vs_gen_dist[:, 0]
        dist['log10 genomic dist (kbp)'] = np.log10(d_vs_gen_dist[:, 0])
        dist['median euclidean distance (nm)'] = d_vs_gen_dist[:, 1]
        dist['log10 median dist (nm)'] = np.log10(d_vs_gen_dist[:, 1])
        dist['n_data'] = d_vs_gen_dist[:, 2]
        dist['experiment'] = [f'exp_{experiment}'] * len(gen_bin)
        dist['axis'] = [axis] * len(gen_bin)
        return dist
        

    def _compute_inter_loci_genomic_dist(lib, n_bins):
        """ Compute the genomic distance separating two consecutive barcodes (from the experiment)

        @param lib: (pandas dataframe) coordinates of the barcodes
        @param n_bins: (int) total number of barcodes of the library
        @return: locus_dist: (pandas dataframe) contains the genomic distance in kbp
        """
        locus_dist = pd.DataFrame()
        mean_pos = (lib["Chrom_End"] + lib["Chrom_Start"]) / 2

        for bin in range(n_bins-1):
            d = mean_pos[bin + 1] - mean_pos[bin]
            d = int(d /1000)
            if bin == 0:
                d_interloci = [d]
                locus = [f'locus_{bin}_{bin+1}']
            else:
                d_interloci = np.concatenate((d_interloci, [d]), axis=0)
                locus = locus + [f'locus_{bin}_{bin+1}']

        locus_dist['locus'] = locus
        locus_dist['genomic distance (kbp)'] = d_interloci
        return locus_dist


    def _compute_efficiency_map(pwd, path_to_save):
        """ Compute efficiency map.

        @param pwd: (numpy array) pairwise distance maps
        @param root: (str) root folder
        @param tag_exp: (str) indicate the type of experiment that is analyzed
        """
        n_maps = pwd.shape[0]
        efficiency = np.zeros((pwd.shape[1],))
        for n in range(pwd.shape[1]):
            efficiency[n] = np.sum(~np.isnan(pwd[:, n])) * 100 / n_maps
        efficiency = squareform(efficiency)

        plt.figure(figsize=(8, 6))
        plt.imshow(efficiency, cmap="hot")
        plt.colorbar(label="efficiency")
        plt.tight_layout()
        plt.title(f'{tag} n={pwd.shape[0]}')
        fig_name = f"Efficiency_{tag_exp}.png"
        fig_path = path.join(root, "barcode_efficiency", fig_name)
        plt.savefig(fig_path, format="png")


####################


class ProcessedData:

    def __init__(self):
        self.traces = None
        self.pwds = None
        self.bins = None
        self.bc_effs = None
        self.metadata = None
        self.bc_list = None

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
    
    
    def plot_end_to_end_dist():
        """"""


    def plot_pwds_distribution():
        """"""
    

    def plot_bc_effs():
        """"""

    

####################


class DecompositionResult:
    """ Decomposition results handler, reads and writes results
    
    Methods for post-processing to compute coherence, 
    """

    def __init__(self, model:LatentDirichletAllocation, decomposition:np.ndarray):
        self.model = model
        self.decomposition = decomposition
        self.components = model.components_
            

    def save(self, path_to_dir:str):
        with h5py.File(path.join(path_to_dir, "results.h5"), mode="w") as handler:
            handler.create_dataset(name="components", data=self.components)
            handler.create_dataset(name="decomposition", data=self.decomposition)
        import joblib
        joblib.dump(self.model, path.join(path_to_dir, "model.jl"))

    
    def jsd_cluster(self, clusters, components_by_cluster):
        cluster_pairs = list(combinations(clusters, 2))
        div = np.empty(len(cluster_pairs))
        for i, (c0, c1) in enumerate(cluster_pairs):
            div[i] = np.mean(cdist(components_by_cluster[c0], components_by_cluster[c1], metric="jensenshannon"))
        div = squareform(div)
        np.fill_diagonal(div, [np.mean(pdist(components_by_cluster[cluster], metric="jensenshannon")) for cluster in clusters])

        fig, ax = plt.subplots()
        fig.set_figwidth(10)
        fig.set_figheight(8)
        im = ax.imshow(div, cmap="Blues")
        cbar = fig.colorbar(im, shrink=0.85)
        cbar.set_label("mean Jensen-Shannon divergence", labelpad=6)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xticks(np.arange(len(clusters)), labels=clusters);
        ax.set_yticks(np.arange(len(clusters)), labels=clusters);
        ax.tick_params("x", labelsize=10)
        ax.tick_params("y", labelsize=10)
        ax.set_xlabel("Cluster #", labelpad=10)
        ax.set_ylabel("Cluster #", labelpad=10)
        plt.tight_layout()
        fig.savefig(f"{dir_results}/{tag}_jsd_heatmap.png", dpi=300)

        assert issymmetric(div)


    def embed(self, method="UMAP", cluster=False, **kwargs):

        if method.lower().strip() == "umap":
        
            # UMAP parameters list
            metric = "cosine"
            n_umap_components = 2
            # min_dist_all = [0.01, 0.1, 1]
            min_dist_all = [0.2, 0.4, 0.6, 0.8]
            # num_neighbors_all = [2, 22, 42, 62, 82]
            num_neighbors_all = [4, 8, 12, 16, 20]
            param_sets = list(product(min_dist_all, num_neighbors_all))

            fig, axes = plt.subplots(nrows=len(min_dist_all), ncols=len(num_neighbors_all), sharex=True, sharey=True)
            fig.set_figwidth(len(num_neighbors_all) * 4)
            fig.set_figheight(len(min_dist_all) * 4)

            for i, (min_dist, num_neighbors) in enumerate(param_sets):

                reducer = umap.UMAP(
                    n_neighbors=num_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    n_components=2,
                    random_state=42,
                )
                reducer.fit(components_all)
                jl.dump(reducer, path.join(dir_results,
                                        "optimization",
                                        f"{tag_wt}-{tag_mut}-combined-{num_components_wt}-{num_components_mut}_UMAP-components" \
                                        f"_model_md={min_dist}_nn={num_neighbors}.jl"))
                embedding = reducer.transform(components_all)
                embedding_df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])

                sns.scatterplot(data=embedding_df,
                                x="UMAP_1", y="UMAP_2",
                                s=10, alpha=0.4, color="#216eff", edgecolor="none",
                                legend=False,
                                ax=axes.flat[i])
                axes.flat[i].set_title(f"min_dist={min_dist}, num_neighbors={num_neighbors}")

            fig.tight_layout()
            fig.savefig(path.join(dir_results,
                                "optimization",
                                f"{tag_wt}-{tag_mut}-combined-{num_components_wt}-{num_components_mut}_UMAP-components_grid_1.png"),
                                dpi=300)
            plt.close(fig)

        elif method.lower() == "pca":
            reducer = PCA(n_components=2, perplexity=40, random_state=42)
            embedding = reducer.fit_transform(components_all)
            embedding_df = pd.DataFrame(embedding, columns=["C1", "C2"])
            jl.dump(reducer, path.join(dest_folder, f"{tag}_{num_components}_TSNE-components_model.jl"))


        # HDBSCAN parameters for the pooled components
        min_size = 50
        min_sample = 1      # increase it to break apart long clusters
        method = "leaf"     # "eom"  # "leaf"
        alpha = 1.0     # strength of distance scaling — higher = stricter merging
        
        if cluster:
            clusterer = HDBSCAN(
                min_cluster_size=min_size,
                min_samples=min_sample,
                cluster_selection_method=method,
                alpha=alpha,
                prediction_data=True
            )
            clusterer = clusterer.fit(embedding)
            embedding_df["cluster"] = clusterer.labels_

            clusters_unique = sorted(list(np.unique(clusterer.labels_).astype(int)))
            clusters_unique.append(clusters_unique.pop(0))
            clusters_unique = np.asarray(clusters_unique)
            num_clusters = len(clusters_unique)     # noise is included as a cluster


    
    def consensus(self, method="jsd"):
        """"""
        return
    

    def umass_coherence(n_topics, topics, raw_data, n_top_words=10, random=False):
        """ Compute the umass coherence for a selected set of topics and raw data. The calculation is performed by selecting
        the top words of each topic. A default value of 10 words (or bins) is indicated.

        @param n_top_words: (int) number of top-words (bins) to select for computing the coherence
        @param n_topics: (int) number of topics used for LDA decomposition
        @param topics: (numpy array) list all the topics processed during LDA decomposition
        @param raw_data: (numpy array) contain the contact vectors for each single cells (previously selected for LDA)
        @param random: (boolean) indicate whether the data should be randomized (in order to compute a baseline)
        @return coherence_umass: (numpy array) contain the coherence values computed for each topic
        """
        coherence_umass = np.zeros((n_topics,))
        data_frequency = np.sum(raw_data, axis=0)

        for n_topic in range(n_topics):
            topic = topics[n_topic, :]
            if random is False:
                idx = np.argsort(topic, )[::-1]  # return the bin index in descending order
            else:
                idx = np.random.permutation(topic.shape[0])

            top_indices = idx[:n_top_words]
            c_umass = 0
            for m in range(1, n_top_words):
                for n in range(0, m):
                    d_mn = np.sum((raw_data[:, top_indices[m]] == 1) & (raw_data[:, top_indices[n]] == 1))
                    d_nn = data_frequency[top_indices[n]]
                    if d_nn > 0:
                        c_umass += np.log2((d_mn + 10 ** -6) / d_nn)

            coherence_umass[n_topic] = 2 / (n_top_words * (n_top_words - 1)) * c_umass

        return coherence_umass


    def npmi_coherence(n_topics, topics, raw_data, n_top_words=10, random=False):
        """ Compute coherence for a selected set of topics and raw data using two methods. The calculation is based on the
        normalized pointwise mutual information and is performed by selecting the top words of each topic. A default value
        of 10 words (or bins) is indicated.

        @param n_top_words: (int) number of top-words (bins) to select for computing the coherence
        @param n_topics: (int) number of topics used for LDA decomposition
        @param topics: (numpy array) list all the topics processed during LDA decomposition
        @param raw_data: (numpy array) contain the contact vectors for each single cells (previously selected for LDA)
        @param random: (boolean) indicate whether the data should be randomized (in order to compute a baseline)
        @return coherence_npmi: (numpy array) contain the npmi coherence values computed for each topic
                coherence_cos: (numpy array) contain the coherence values computed for each topic based on cosine similarity
        """
        coherence_npmi = np.zeros((n_topics,))
        coherence_cos = np.zeros((n_topics,))
        n_data = raw_data.shape[0]
        data_frequency = np.sum(raw_data, axis=0)

        for k in range(n_topics):
            topic = topics[k, :]
            if random is False:
                idx = np.argsort(topic, )[::-1]  # return the bin index in descending order
            else:
                idx = np.random.permutation(topic.shape[0])

            top_idx = idx[:n_top_words]

            # compute the NPMI for all pairs of selected bins
            npmi = np.zeros((n_top_words, n_top_words))
            for m in range(n_top_words):
                for n in range(m+1, n_top_words):
                    i, j = top_idx[m], top_idx[n]
                    co_occurrence = np.sum((raw_data[:, i] >= 0.8) & (raw_data[:, j] >= 0.8))
                    p_mn = co_occurrence / n_data
                    p_nn = data_frequency[idx[n]] / n_data
                    p_mm = data_frequency[idx[m]] / n_data
                    if co_occurrence > 0:
                        pmi = np.log2((p_mn + 10 ** -6) / (p_nn * p_mm))
                        npmi[m, n] = - pmi / np.log2(p_mn + 10**-6)
                        npmi[n, m] = npmi[m, n]

            c_npmi = np.sum(npmi[np.triu_indices(n_top_words, k=1)])

            # for each selected bins (top words), compute the norm of the NPMI computed for all the bins (one top bin versus
            # all). This is required for the calculation of the cosine similarity.
            norm = np.linalg.norm(npmi, axis=1)
            # norm = np.zeros((n_top_words,))
            # for n in range(n_top_words):
            #     norm[n] = np.sqrt(np.sum(np.square(npmi[n, :])))

            c_cos = 0
            # based on the previous calculation, compute the NPMI and COS coherence
            for m in range(0, n_top_words, 1):
                for n in range(m + 1, n_top_words):
                    # c_npmi += npmi[m, n]
                    c_cos += np.dot(npmi[m, :], npmi[n, :]) / (norm[m] * norm[n])

            coherence_cos[k] = 2 / (n_top_words * (n_top_words - 1)) * c_cos
            coherence_npmi[k] = 2 / (n_top_words * (n_top_words - 1)) * c_npmi

        return coherence_cos, coherence_npmi


    def reconstruction(self):
        return self.decomposition @ self.components

    @staticmethod
    def _extract_threshold(X):
        return X

    def representativity(self, threshold=None):
        if threshold is None:
            threshold = _extract_threshold(self.decomposition)
        return np.mean(self.decomposition > threshold, axis=0)
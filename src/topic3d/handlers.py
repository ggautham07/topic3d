import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib as jl
import matplotlib.pyplot as plt
from numbers import Number
from os import path
from scipy.spatial.distance import squareform, pdist
from . import visualization as vzl


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
            # # TODO store original dataset indices of the training and testing sets using self.data_idxs = kwargs.get()
            # handler.create_dataset(name="data_idxs/training", data=self.data_training)
            # handler.create_dataset(name="data_idxs/testing", data=self.data_testing)
            for k, v in self.metadata.items():
                try:
                    handler.attrs[k] = v
                except TypeError:
                    handler.attrs[k] = str(v)
                else:
                    raise
        
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
    def plot_components(self, normalized: bool = False, ordered: bool = False, 
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
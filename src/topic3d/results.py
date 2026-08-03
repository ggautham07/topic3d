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
from scipy.linalg import issymmetric
from . import visualization as vzl


# TODO
class Topic3DResult:
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
        self.CFMs = model.components_
        self.CFMs_norm = self.CFMs / np.sum(self.CFMs, axis=1)[:, np.newaxis]
        self.num_CFMs = len(self.CFMs)
        self.num_items = len(self.decomposition)
        self.metadata = kwargs.copy()


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
            handler.create_dataset(name="CFMs", data=self.CFMs)
            handler.create_dataset(name="decomposition", data=self.decomposition)
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


    def reconstruct_data(self, normalize=False):
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
            return self.decomposition @ self.CFMs_norm
        return self.decomposition @ self.CFMs
  

    @staticmethod
    def _extract_representativity_threshold(decomposition, method="gt", value=None):

        assert method in ["gt", "x"]

        default_value = {"gt": 0.1, "x": 2}
        if value is None:
            value = default_value[method]

        prob = 1 / decomposition.shape[-1]

        if method == "gt":
            return prob + value
        return prob * value


    def compute_CFM_representativity(self, weight_threshold: float = None) -> np.ndarray:
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
        
        if weight_threshold is None:
            weight_threshold = self._extract_representativity_threshold(self.decomposition)
        else:
            assert isinstance(weight_threshold, Number)

        return np.mean(self.decomposition >= weight_threshold, axis=0)


    @staticmethod
    def _compute_coherence_umass_legacy(components, data_used, num_top_pixels=None, randomize_data=False):
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
    def _compute_coherence_npmi_legacy(CFM: np.ndarray,
                                       data: np.ndarray,
                                       num_top_pixels: int = None,
                                       randomize_data: bool = False,
                                       return_cos: bool = False):
        """
        Compute the NPMI and cosine coherence scores from the proximity data for each CFM.

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

        data_ = np.asarray(data)
        data_probs = np.mean(data_, axis=0)

        assert (num_top_pixels is None) or isinstance(num_top_pixels, Number)
        var_num_top_pixels = True if num_top_pixels is None else False

        EPS = 10**-5    # value of epsilon is kept constant

        if randomize_data:
            idxs = np.random.permutation(CFM.shape[0])
        else:
            idxs = np.argsort(CFM)[::-1]  # return the bin index in descending order
        if var_num_top_pixels:
            num_top_pixels = np.sum(CFM > 1, dtype=int)
        idxs_top = idxs[:num_top_pixels]

        # Compute the NPMI for all pairs of selected bins
        npmi = np.zeros((num_top_pixels, num_top_pixels))
        for m in range(num_top_pixels):
            for n in range(m+1, num_top_pixels):
                i, j = idxs_top[m], idxs_top[n]
                co_occurrence = np.mean((data_[:, i] == 1) & (data_[:, j] == 1))
                p_mn = co_occurrence
                p_nn = data_probs[idxs_top[n]]
                p_mm = data_probs[idxs_top[m]]
                if co_occurrence > 0:
                    pmi = np.log2((p_mn + EPS) / (p_nn * p_mm))
                    npmi[m, n] = - pmi / np.log2(p_mn + EPS)
                    npmi[n, m] = npmi[m, n]
        c_npmi = np.sum(npmi[np.triu_indices(num_top_pixels, k=1)])
        coherence_npmi = 2 / (num_top_pixels * (num_top_pixels - 1)) * c_npmi

        if return_cos:
            # For each selected pixel, compute the norm of the NPMI computed for all the pixels (one vs all)
            # Required for the calculation of the cosine similarity.
            norm = np.linalg.norm(npmi, axis=1)
            c_cos = 0
            for m in range(0, num_top_pixels, 1):
                for n in range(m + 1, num_top_pixels):
                    c_cos += np.dot(npmi[m, :], npmi[n, :]) / (norm[m] * norm[n])
            coherence_cos = 2 / (num_top_pixels * (num_top_pixels - 1)) * c_cos

            return coherence_npmi, coherence_cos

        return coherence_npmi


    @staticmethod
    def _compute_pmi_matrix(X: np.ndarray, gamma: float = 1.0):

        EPS = 10**-5

        # If X is proximity data, compute cooccurrence matrix
        assert X.ndim == 2
        if (X.shape[0] == X.shape[1]) and issymmetric(X):
            cooccur = X[:]
        else:    
            num_items = X.shape[0]
            cooccur = X.T @ X

        # Get probabilities for PMI computation from cooccurrence matrix
        pij = cooccur / num_items
        pij_safe = np.clip(pij, a_min=EPS, a_max=1)     # to ensure postive non-zero values during computations
        p = np.diag(pij)
        pi_pj = p[:, None] * p[None, :]

        # To take only values where probabilities are > 0, rest will be NaN later
        valid = pi_pj > 0

        # Compute the PMI for all pairs
        pmi = np.full_like(pij, np.nan, dtype=float)
        pmi[valid] = np.log2(pij_safe[valid] / pi_pj[valid])

        # Initialize NPMI matrix and assign values to edge case not covered by the usual computation
        npmi = np.full_like(pij, np.nan, dtype=float)
        npmi[(pij == 0) & valid] = -1

        # Compute the NPMI
        mask = valid & (pij > 0) & (pij < 1)
        npmi[mask] = pmi[mask] / -np.log2(pij_safe[mask])

        # Return values after raising to power in case differences need to be emphasized
        return pmi**gamma, npmi**gamma
    

    # NOTE NPMI coherence could be vectorized to compute for multiple CFMs, but seems unnecessary
    @staticmethod
    def _compute_coherence_npmi(npmi_mat: np.ndarray,
                                CFM: np.ndarray,
                                num_top_px: int = -1,
                                ignore_nans: bool = True,
                                return_cosine: bool = False):

        assert isinstance(npmi_mat, np.ndarray) and issymmetric(npmi_mat)

        # Take only pixels with an unnormalized pseudocount value >> 0
        valid = np.nonzero(CFM >= 1)[0]
        if len(valid) == 0:
            return np.nan

        # Take all valid pixel indices if -1, otherwise, the top pixels based on pseudocount values
        if num_top_px == -1:
            idxs = valid[:]
        else:
            sort_idxs = np.argsort(CFM)[::-1][:num_top_px]
            idxs = np.intersect1d(valid, sort_idxs)
        num_top_px_true = len(idxs)

        # Extract NPMI submatrix based on indices
        npmi_submat = npmi_mat[np.ix_(idxs, idxs)]
        upper = np.triu_indices(num_top_px_true, k=1)

        # Compute the NPMI coherence
        if ignore_nans:
            coherence_npmi = np.nanmean(npmi_submat[upper])
        else:
            coherence_npmi = 2 / (num_top_px_true * (num_top_px_true - 1)) * np.nansum(npmi_submat[upper])

        # Compute the cosine coherence
        if return_cosine:
            norm = np.linalg.norm(npmi_submat, axis=1)
            cos = 0
            for m in range(0, num_top_px_true):
                for n in range(m+1, num_top_px_true):
                    cos += np.dot(npmi_submat[m, :], npmi_submat[n, :]) / (norm[m] * norm[n])
            if ignore_nans:
                coherence_cos = np.nanmean(cos)
            else:
                coherence_cos = 2 / (num_top_px_true * (num_top_px_true - 1)) * cos

            return coherence_npmi, coherence_cos

        return coherence_npmi


    def compute_coherence(self,
                          data: np.ndarray,
                          num_top_px: int = -1,
                          gamma: float = 1.0,
                          ignore_nans: bool = True):
        """
        Compute coherence of CFMs using NPMI and cosine.

        Parameters
        ----------
        data : ndarray
            Proximity data to check coherence with (usually training data)
        num_top_px : int, optional
            Number of top pixels to use per CFM. Defaults to all valid pixels for each CFM
        gamma : float, optional
            Power to raise computed NPMI coherence values (default=1), used to emphasize differences
        ignore_nans : bool, optional
            Ignore NaNs in the computed NPMI values during coherence estimation 

        Returns
        -------
        (coherence_npmi, coherence_cos) : tuple
            NPMI and cosine coherence averaged over all CFMs

        Examples
        --------
        >>> model.compute_coherence(num_top_px=10)
        """

        # Compute normalized pointwise mutual information from entire proximity data
        _, npmi_mat = self._compute_pmi_matrix(X=data, gamma=gamma)

        # Compute the coherence using the above matrix for all the CFMs
        coherence_npmi_all = np.empty(self.num_CFMs)
        coherence_cos_all = np.copy(coherence_npmi_all)
        for i in range(self.num_CFMs):
            cc_npmi, cc_cos = self._compute_coherence_npmi(npmi_mat=npmi_mat,
                                                           CFM=self.CFMs[i],
                                                           num_top_px=num_top_px,
                                                           ignore_nans=ignore_nans,
                                                           return_cosine=True)
            coherence_npmi_all[i] = cc_npmi
            coherence_cos_all[i] = cc_cos

        # Average over number of CFMs without ignoring NaNs (seems more suitable for aggregation)
        coherence_npmi = np.nansum(coherence_npmi_all) / self.num_CFMs
        coherence_cos = np.nansum(coherence_cos_all) / self.num_CFMs

        return coherence_npmi, coherence_cos


    def order_CFMs(self, by="representativity", weight_threshold: Number = None):
        """
        Sort components by representativity (based on a weight threshold, the default) or unnormalized pseudocount.

        Parameters
        ----------
        weight_threshold : float, optional
            Minimum weight above which a component is considered present. If None, uses the model's default threshold.

        Returns
        -------
        np.ndarray
            Sorted indices of components (most to least representative).

        Examples
        --------
        >>> idxs = model.sort_components(threshold=0.3)
        """

        assert by in ["representativity", "pseudocount"]

        # Get representativity/max. pseudocount for all CFMs
        if by == "representativity":
            if weight_threshold is None:
                weight_threshold = self._extract_representativity_threshold(self.decomposition)
            else:
                assert isinstance(weight_threshold, Number)
            assert 0 < weight_threshold <= 1        
            vals = self.compute_CFM_representativity(weight_threshold)
        else:
            vals = np.max(self.CFMs, axis=1)

        # Sort in descending order
        idxs = np.argsort(vals)[::-1]

        return idxs


    # TODO returning mean divergence for decomposition, sorting the matrix by least to most distant?
    def compute_CFM_divergence(self, metric="jensenshannon", path_to_save:str = ""):
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

        pw_divergence = squareform(pdist(self.CFMs, metric=metric))

        if wflag:
            import matplotlib.pyplot as plt
            assert path_to_save.endswith((".svg", ".png"))
            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(8)
            im = ax.imshow(pw_divergence, cmap="Blues")
            cbar = fig.colorbar(im, shrink=0.85)
            cbar.set_label(f"{metric} distance", labelpad=6)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ticks_labels = np.arange(self.num_CFMs)
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
    def plot_CFMs(self, ordered: bool = False, normalized: bool = False,
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

        reprez = self.compute_CFM_representativity(weight_threshold=kwargs.get("weight_threshold", (1 / self.num_CFMs) + 0.1))
        
        if ordered:
            idxs = self.order_CFMs(weight_threshold=kwargs.get("threshold", (1 / self.num_CFMs) + 0.1))
            reprez = reprez[idxs]
            items_to_plot = self.CFMs_norm[idxs,:] if normalized else self.CFMs[idxs,:]
        else:
            items_to_plot = self.CFMs_norm if normalized else self.CFMs

        idxs_comps = np.arange(self.num_CFMs)
        titles = [f"CFM {i}  |  $\\rho = ${r*100:.2f}%" \
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

        from matplotlib.ticker import MaxNLocator
        import seaborn as sns

        wflag = self._set_figsave_flag(path_to_save)

        fig = plt.figure(figsize=(8, 10), dpi=300)
        ax = fig.add_subplot(1, 1, 1)

        if ordered:
            reprez = self.compute_CFM_representativity(weight_threshold=(1 / self.num_CFMs) + 0.1)
            idxs_reprez_desc = np.argsort(reprez)[::-1]
            weights_reorg_init = self.decomposition[:, idxs_reprez_desc]
            idxs_maxval_by_trace = np.argmax(weights_reorg_init, axis=1)
            weights_ordered = np.empty((0, self.num_CFMs), dtype=weights_reorg_init.dtype)
            idxs_traces_new_order = [] 
            for i in range(len(idxs_reprez_desc)):
                selection = weights_reorg_init[idxs_maxval_by_trace == i, :]
                idxs_selection_ordered = np.argsort(selection[:, i])[::-1]
                weights_ordered = np.concatenate((weights_ordered, selection[idxs_selection_ordered, :]),
                                                 axis=0)
                idxs_traces_new_order.extend(idxs_selection_ordered)

            def format_fn(tick_val, tick_pos):
                if int(tick_val) in range(len(idxs_traces_new_order)):
                    return idxs_traces_new_order[int(tick_val)]
                else:
                    return ""

        sns.heatmap(weights_ordered if ordered else self.decomposition,
                    vmin=0, vmax=1,
                    cmap=cmap,
                    ax=ax,
                    xticklabels=idxs_reprez_desc if ordered else "auto",
                    cbar_kws={"shrink": 0.6, "label": "probability"})
        if ordered:
            yticks = ax.get_yticks()
            labels = [idxs_traces_new_order[int(round(t))] for t in yticks]
            ax.set_yticklabels(labels)
        #     ax.yaxis.set_major_formatter(format_fn)
        #     ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
        plt.xlabel("CFM #", labelpad=10)
        plt.ylabel("Trace #", labelpad=10)

        if wflag:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig


    def plot_decomposition_mult(self,
                                sources: np.ndarray,
                                cmaps: dict = None,
                                path_to_save: str = ""):
        """
        Plot an ordered decomposition matrix with multiple data sources.        
        """

        from itertools import chain
        import seaborn as sns

        wflag = self._set_figsave_flag(path_to_save)

        reprez = self.compute_CFM_representativity(self.decomposition, weight_threshold=(1/self.num_CFMs)+0.1)
        idxs_reprez_desc = np.argsort(reprez)[::-1]
        decomposition_reorg = self.decomposition[:, idxs_reprez_desc]
        idxs_maxval_by_trace = np.argmax(decomposition_reorg, axis=1)
        decomposition_ord = np.empty((0, self.num_CFMs), dtype=decomposition_reorg.dtype)
        sources_ord = np.empty(0, dtype=sources.dtype)
        for i in range(len(idxs_reprez_desc)):
            selection = decomposition_reorg[idxs_maxval_by_trace == i, :]
            sources_filt = sources[idxs_maxval_by_trace == i]
            for s in np.unique(sources):
                idxs_ord = (np.argsort(selection[sources_filt == s, i])[::-1])
                decomposition_ord = np.concatenate((decomposition_ord,
                                                    selection[sources_filt == s][idxs_ord, :]), axis=0)
                sources_ord = np.concatenate(sources_ord,
                                             sources_filt[sources_filt == s][idxs_ord])

        default_cmaps = ["Greys", "Purples", "Blues", "Greens", "Oranges", "Reds"]
        if cmaps is None:
            sources_unique = np.unique(sources)
            cmaps = {src: cmap for src, cmap in zip(sources_unique, default_cmaps)}

        fig = plt.figure(figsize=(6, 4))
        fig.set_layout_engine("tight")
        fig.set_dpi(600)
        ax = sns.heatmap(decomposition_ord, cmap="Greys", vmin=0, vmax=1, cbar=False)
        for src, cmap in cmaps.items():
            mask = np.zeros_like(decomposition_ord, dtype=bool)
            mask[~(sources_ord == src)] = True
            sns.heatmap(decomposition_ord, mask=mask, cmap=cmap, vmin=0, vmax=1, ax=ax, cbar=True)
        ax.set_xticks(ticks=np.arange(idxs_reprez_desc.shape[0])+0.5, labels=idxs_reprez_desc, fontsize=6);
        ax.get_yaxis().set_visible(False)

        if wflag:
            fig.savefig(path_to_save,
                        bbox_inches="tight")
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

        from scipy.spatial.distance import squareform, pdist
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
                assert isinstance(min_dist_embedding, Number)
            except KeyError:
                print(f"Parameter values for {method} not found, supply using a dictionary")
                raise
            except AssertionError:
                print("Parameter values not suitable")
                raise
            # UMAP model training and data embedding
            try:
                reducer = UMAP(
                    n_neighbors=num_neighbors_embedding,
                    min_dist=min_dist_embedding,
                    metric=metric,
                    n_components=num_components_embedding,
                    random_state=42,
                )
                embedding = reducer.fit_transform(self.decomposition)
            except ValueError:
                reducer = UMAP(
                    n_neighbors=num_neighbors_embedding,
                    min_dist=min_dist_embedding,
                    metric="precomputed",
                    n_components=num_components_embedding,
                    random_state=42,
                )
                embedding = reducer.fit_transform(squareform(pdist(self.decomposition, metric=metric)))

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
        component_reprez = np.argmax(self.decomposition, axis=1)
        embedding_df["representative"] = [f"CFM #{i}" for i in component_reprez]

        # Plotting the results
        fig, ax = plt.subplots(dpi=300)
        fig.set_figwidth(8)
        fig.set_figheight(8)
        sns.scatterplot(data=embedding_df,
                x=embedding_df.columns[0], y=embedding_df.columns[1],
                s=10, alpha=0.4, hue="representative", edgecolor="none",
                legend=True,
                ax=ax)
        ax.set_title(method)
        fig.tight_layout()
        fig.savefig(path.join(dir_to_save, f"embedding_decomposition_{method}.png"))
        plt.close(fig)
        
        if cluster:

            # HDBSCAN parameters for the pooled components
            min_cluster_size = kwargs.get("min_cluster_size", self.decomposition.shape[0] // 40)
            min_samples = kwargs.get("min_samples", 1)
            cluster_selection_method = kwargs.get("cluster_selection_method", "eom")
            alpha = kwargs.get("alpha", 1.0)
            
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
                plt.text(row[f"{method_label}_1"]+0.5, row[f"{method_label}_2"]+0.5, str(cluster_id),
                         fontsize=10, weight="bold", ha="center", va="center", color="black",
                         bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.3"))
            plt.tight_layout()
            plt.savefig(path.join(dir_to_save, f"embedding_decomposition_{method}_clustered.png"), dpi=300)
            plt.close()

        return embedding_df


class Topic3DModelPool():
    
    def __init__(self, models: list[str, LatentDirichletAllocation]):
        self.models = []
        for md in models:
            if isinstance(md, str):
                assert path.isfile(md)
                md_ = jl.load(md)
                assert isinstance(md_, LatentDirichletAllocation)
                self.models.append(md_)
            elif isinstance(md, LatentDirichletAllocation):
                self.models.append(md)
        self.CFMs_by_model = [model.components_ for model in self.models]
        self.num_CFMs_by_model = np.asarray([len(CFMs) for CFMs in self.CFMs_by_model])


    @staticmethod
    def _get_model_perplexity(model):
        return model.bound_
    

    def filter(self, value, by="num_CFMs"):
        """
        Filter the model pool by a specific parameter value and return filtered models
        """

        mapping = {
            "num_CFMs": "n_components",
            "max_iter": "max_iter",
            "random_state": "random_state",
            "learning_method": "learning_method"
        }
        
        models = [md for md in self.models if getattr(md, mapping[by]) == value]

        return Topic3DModelPool(models=models)


    def rank_models(self, by="perplexity", **kwargs):
        """
        Rank models by perplexity or coherence.

        If by is coherence, pass the following kwargs.

        data : ndarray
            Proximity data to check coherence with (usually training data)
        num_top_px : int, optional
            Number of top pixels to use per CFM. Defaults to all valid pixels for each CFM
        gamma : float, optional
            Power to raise computed NPMI coherence values (default=1), used to emphasize differences
        ignore_nans : bool, optional
            Ignore NaNs in the computed NPMI values during coherence estimation 

        
        """

        assert isinstance(by, str) and by in ["coherence", "perplexity"]

        # Handling parameters for coherence-based ranking
        if by == "coherence":
            data = kwargs["data"]
            assert isinstance(data, np.ndarray)     # TODO common proximity data checking function, or FOF loader everywhere
            num_top_px = kwargs.get("num_top_px", -1)
            gamma = kwargs.get("gamma", 1)
            ignore_nans = kwargs.get("ignore_nans", True)

        # Computing values for coherence-based ranking
        if by == "coherence":
            # Compute normalized pointwise mutual information from entire proximity data
            _, npmi_mat = Topic3DResult._compute_pmi_matrix(X=data, gamma=gamma)
            scores = []
            for model in self.models:
            # Compute the coherence using the above matrix for all the CFMs
                coherence = np.empty(len(model.components_))
                for i in range(len(model.components_)):
                    cc_npmi = Topic3DResult._compute_coherence_npmi(npmi_mat=npmi_mat,
                                                                    CFM=model.components_[i],
                                                                    num_top_px=num_top_px,
                                                                    ignore_nans=ignore_nans,
                                                                    return_cosine=False)
                    coherence[i] = cc_npmi
                # Average over number of CFMs without ignoring NaNs (seems more suitable for aggregation)
                scores.append(np.nansum(coherence) / len(model.components_))

        # Computing them for perplexity-based ranking
        elif by == "perplexity":
            scores = np.asarray([self._get_model_perplexity(model) for model in self.models])

        # NOTE needs updating if other methods are included later
        sort_idxs = np.argsort(scores) if by=="perplexity" else np.argsort(scores)[::-1] 

        return np.asarray(self.models)[sort_idxs]


    def find_best_model(self, by="perplexity", **kwargs) -> LatentDirichletAllocation:
        """
        Find the best model by coherence or perplexity.
        """

        if by == "coherence":
            return self.rank_models(by=by, **kwargs)[0]
        elif by == "perplexity":
            return self.rank_models(by=by)[0]


    def compute_coherence_vs_num_CFMs(self,
                                      data: np.ndarray,
                                      gamma: float = 1.0,
                                      num_top_px: int = -1,
                                      ignore_nans: bool = True):
        """
        Compute the NPMI coherence as a function of number of CFMs
        """
        
        num_CFMs_unique = np.unique(self.num_CFMs_by_model)

        _, npmi_mat = Topic3DResult._compute_pmi_matrix(X=data, gamma=gamma)

        coherence_by_n = {n: [] for n in num_CFMs_unique}

        for n in num_CFMs_unique:
            model_idxs = np.nonzero(self.num_CFMs_by_model == n)[0]
            for i in model_idxs:
                cc_npmi = []
                for comp in self.models[i].components_:
                    cc_npmi.append(Topic3DResult._compute_coherence_npmi(npmi_mat=npmi_mat,
                                                                         CFM=comp,
                                                                         num_top_px=num_top_px,
                                                                         ignore_nans=ignore_nans,
                                                                         return_cosine=False))
                coherence_by_n[n].append(np.nansum(cc_npmi) / n)

        return coherence_by_n
    

    def rank_num_CFMs(self, by="perplexity", **kwargs):
        """
        Rank models by coherence or perplexity
        """
        
        assert isinstance(by, str) and by in ["coherence", "perplexity"]

        # Handling parameters for coherence-based ranking
        if by == "coherence":
            data = kwargs["data"]
            assert isinstance(data, np.ndarray)     # TODO common proximity data checking function, or FOF loader everywhere
            num_top_px = kwargs.get("num_top_px", -1)
            gamma = kwargs.get("gamma", 1)
            ignore_nans = kwargs.get("ignore_nans", True)

        num_CFMs_unique = np.unique(self.num_CFMs_by_model)

        if by == "coherence":
            coherence_by_n = self.compute_coherence_vs_num_CFMs(data=data,
                                                                gamma=gamma,
                                                                num_top_px=num_top_px,
                                                                ignore_nans=ignore_nans)
            scores = [np.mean(coherence_by_n[n]) for n in num_CFMs_unique]

        elif by == "perplexity":
            scores = []
            for n in num_CFMs_unique:
                model_idxs = np.nonzero(self.num_CFMs_by_model == n)[0]
                scores.append(np.mean([self._get_model_perplexity(self.models[i]) \
                                       for i in model_idxs]))

        for n, score in zip(num_CFMs_unique, scores):
            print(f"{n} CFMs: {score:.3f}")

        # NOTE will need updating if other methods are included
        sort_idxs = np.argsort(scores) if by=="perplexity" else np.argsort(scores)[::-1]
        
        return num_CFMs_unique[sort_idxs]


    def find_best_num_CFMs(self, by="perplexity", **kwargs):

        ranking = self.rank_num_CFMs(by=by, **kwargs)
        means_empty = self.compute_mean_empty_vs_num_CFMs()
        for n_CFMs in ranking:
            if means_empty[n_CFMs] == 0:
                return n_CFMs


    def plot_mean_perplexity_vs_num_CFMs(self, path_to_save: str = ""):
        
        num_CFMs_by_model_unique = np.unique(self.num_CFMs_by_model)
        perplexities = np.asarray([model.bound_ for model in self.models])
        perplexities_means = []
        perplexities_stds = []
        for n in num_CFMs_by_model_unique:
            perps = perplexities[self.num_CFMs_by_model == n]
            perplexities_means.append(np.mean(perps))
            perplexities_stds.append(np.std(perps))
        
        fig, ax = plt.subplots()
        ax.errorbar(x=num_CFMs_by_model_unique,
                    y=perplexities_means,
                    yerr=perplexities_stds,
                    capsize=2.5)
        ax.set_xlabel("# CFMs")
        ax.set_ylabel("mean perplexity")
        
        if isinstance(path_to_save, str) and len(path_to_save) > 0:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig


    def plot_mean_coherence_vs_num_CFMs_legacy(self,
                                               data_used: np.ndarray,
                                               method: str = "npmi",
                                               path_to_save: str = ""):
        
        num_CFMs_by_model_unique = np.unique(self.num_CFMs_by_model)

        fig, ax = plt.subplots()
        num_top_px = [5, 10, 15, 20, 30]

        for n_px in num_top_px:

            coherence_all = []
            for CFMs in self.CFMs_by_model:
                coherence_by_CFM = []
                for cfm in CFMs:
                    c, _ = Topic3DResult._compute_coherence_npmi_legacy(CFM=cfm,
                                                                     data=data_used,
                                                                     num_top_pixels=n_px)
                    coherence_by_CFM.append(c)
                coherence_all.append(coherence_by_CFM)
            coherences_means = []
            coherences_stds = []
            for n in num_CFMs_by_model_unique:
                idxs_reqd = np.nonzero(self.num_CFMs_by_model == n)[0]
                cohrs = []
                for i in idxs_reqd:
                    cohrs.extend(coherence_all[i])
                coherences_means.append(np.mean(cohrs))
                coherences_stds.append(np.std(cohrs))

            ax.errorbar(x=num_CFMs_by_model_unique,
                        y=coherences_means,
                        yerr=coherences_stds,
                        capsize=2.5,
                        label=f"top {n_px} pixels")
        
        ax.set_xlabel("# CFMs")
        ax.set_ylabel("mean NPMI coherence")
        plt.legend()
        
        if isinstance(path_to_save, str) and len(path_to_save) > 0:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig
    

    # def plot_mean_coherence_vs_num_CFMs(self, data_used: np.ndarray, method: str = "npmi", path_to_save: str = ""):
        
    #     num_CFMs_by_model_unique = np.unique(self.num_CFMs_by_model)

    #     fig, ax = plt.subplots()
    #     coherence_all_rep = []
    #     coherence_all_norep = []
    #     for model, CFMs in zip(self.models, self.CFMs_by_model):
    #         decomp = model.transform(data_used)
    #         decomp_bin = decomp >= (1 / model.n_components + 0.1)
    #         coherence_by_CFM_rep = []
    #         coherence_by_CFM_norep = []
    #         for i in range(len(CFMs)):
    #             data_rep = data_used[decomp_bin[:,i] == 1]
    #             data_norep = data_used[decomp_bin[:,i] == 0]
    #             c_rep, _ = Topic3DResult._compute_coherence_npmi_legacy(CFM=CFMs[i],
    #                                                                  data=data_rep,)
    #             c_norep, _ = Topic3DResult._compute_coherence_npmi_legacy(CFM=CFMs[i],
    #                                                                    data=data_norep,)
    #             coherence_by_CFM_rep.append(c_rep)
    #             coherence_by_CFM_norep.append(c_norep)
    #         coherence_all_rep.append(coherence_by_CFM_rep)
    #         coherence_all_norep.append(coherence_by_CFM_norep)
    #     coherences_rep_means = []
    #     coherences_rep_stds = []
    #     coherences_norep_means = []
    #     coherences_norep_stds = []        
    #     for n in num_CFMs_by_model_unique:
    #         idxs_reqd = np.nonzero(self.num_CFMs_by_model == n)[0]
    #         cohrs_rep = []
    #         cohrs_norep = []
    #         for i in idxs_reqd:
    #             cohrs_rep.extend(coherence_all_rep[i])
    #             cohrs_norep.extend(coherence_all_norep[i])
    #         coherences_rep_means.append(np.mean(cohrs_rep))
    #         coherences_rep_stds.append(np.std(cohrs_rep))
    #         coherences_norep_means.append(np.mean(cohrs_norep))
    #         coherences_norep_stds.append(np.std(cohrs_norep))

    #     ax.errorbar(x=num_CFMs_by_model_unique,
    #                 y=coherences_rep_means,
    #                 yerr=coherences_rep_stds,
    #                 fmt="s-",
    #                 color="C0",
    #                 capsize=2.5,
    #                 label="representative")
    #     ax.errorbar(x=num_CFMs_by_model_unique,
    #                 y=coherences_norep_means,
    #                 yerr=coherences_norep_stds,
    #                 fmt="o-",
    #                 color="C7",
    #                 capsize=2.5,
    #                 label="non-representative")
        
    #     ax.set_xlabel("# CFMs")
    #     ax.set_ylabel("mean NPMI coherence")
    #     plt.legend()
        
    #     if isinstance(path_to_save, str) and len(path_to_save) > 0:
    #         fig.savefig(path_to_save, bbox_inches="tight")
    #         plt.close(fig)
    #     return fig


    @staticmethod
    def count_empty_CFMs(CFMs):

        counter = 0
        for l in range(len(CFMs)):
            if np.allclose(CFMs[l], 0, atol=1e-1):
                counter += 1
        
        return counter
    

    def compute_mean_empty_vs_num_CFMs(self):
        
        num_CFMs_by_model_unique = np.unique(self.num_CFMs_by_model)
        mean_empty = []
        for n in num_CFMs_by_model_unique:
            CFMs_by_model_reqd = [self.CFMs_by_model[i] for i in np.nonzero(self.num_CFMs_by_model == n)[0]]
            mean_empty.append(np.mean([self.count_empty_CFMs(CFMs) / len(CFMs) for CFMs in CFMs_by_model_reqd]))
        
        return dict(zip(num_CFMs_by_model_unique, mean_empty))


    def plot_mean_empty_vs_num_CFMs(self, path_to_save=""):

        result = self.compute_mean_empty_vs_num_CFMs()

        fig, ax = plt.subplots()
        ax.plot(result.keys(), result.values())
        ax.set_xlabel("# CFMs")
        ax.set_ylabel("mean empty CFMs")
        
        if isinstance(path_to_save, str) and len(path_to_save) > 0:
            fig.savefig(path_to_save, bbox_inches="tight")
            plt.close(fig)
        return fig

    
    # TODO options for thresholding to pull flat clusters
    def cluster_CFMs(self, metric: str = "cosine"):

        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

        CFMs_pooled = []
        CFMs_pooled.extend(self.CFMs_by_model)

        tree = linkage(CFMs_pooled, metric=metric)

        clusters = fcluster(tree, t=0.5)

        return clusters




    # TODO    
    # def embed_CFMs(method="UMAP"):
    #     return
    
    # def compute_consensus(self, metric="jensenshannon"):
    #     return

# TODO REFINE & SORT INTO METHODS FOR TOPIC3DMODELPOOL
# cfm_size_by_n_comps = {}

# for n, components in enumerate(components_all):

#     n_comps = components.shape[0]

#     sizes = []
#     for i in range(n_comps):
#         sizes.append(np.sum(components[i] > 0.5))

#     if n_comps not in cfm_size_by_n_comps.keys():
#         cfm_size_by_n_comps[n_comps] = []
#     cfm_size_by_n_comps[n_comps].extend(sizes)

# num_components_final = []
# sizes_means = []
# sizes_stes = []

# for n_comps, vals in cfm_size_by_n_comps.items():
#     vals = np.asarray(vals)
#     num_components_final.append(n_comps)
#     sizes_means.append(np.nanmean(vals))
#     sizes_stes.append(np.nanstd(vals) / np.sqrt(len(vals)))
# sort_idxs = np.argsort(num_components_final)
# num_components_final = np.asarray(num_components_final)[sort_idxs]
# sizes_means = np.asarray(sizes_means)[sort_idxs]
# sizes_stes = np.asarray(sizes_stes)[sort_idxs]
# plt.errorbar(num_components_final, sizes_means, sizes_stes)
# plt.xlabel("number of CFMs")
# plt.ylabel("size of CFM")
# sizes = [cfm_size_by_n_comps[n] for n in num_components_final]
# plt.boxplot(x=sizes, tick_labels=num_components_final);
# plt.xlabel("number of CFMs")
# plt.ylabel("CFM size (in px)")
# from itertools import chain

# coherences_merged = list(chain.from_iterable([vals for vals in coherence_by_n_comps.values()]))
# sizes_merged = list(chain.from_iterable([vals for vals in cfm_size_by_n_comps.values()]))
# n_CFMs_merged = list(chain.from_iterable([[n] * len(vals) for n, vals in coherence_by_n_comps.items()]))

# assert len(coherences_merged) == len(sizes_merged) == len(n_CFMs_merged)
# plt.scatter(sizes_merged, coherences_merged, c=n_CFMs_merged, s=6, cmap="Set2")
# plt.xlabel("CFM size (in px)")
# plt.ylabel("NPMI coherence")
# plt.colorbar(shrink=0.75, label="number of CFMs")



# num_CFMs = np.unique(model_pool.num_CFMs_by_model)
# reps_by_k = {n:[] for n in num_CFMs}
# for md in model_pool.models:
#     n = md.n_components
#     reps_cutoff = (1 / n) + 0.1
#     decomp = md.transform(data)
#     reps = np.mean(decomp >= reps_cutoff, axis=0)
#     reps_by_k[n].extend(reps)
# reps_by_k
# for n in reps_by_k.keys():
#     reps_by_k[n] = np.asarray(reps_by_k[n]).flatten()
#     print(reps_by_k[n].shape)
# fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
# fig.supxlabel("representativity")
# fig.supylabel("density")
# for n, ax in zip(num_CFMs, axs.flat):
#     r = np.sum(reps_by_k[n] < 0.05) / np.sum(reps_by_k[n] >= 0.05)
#     # r = np.mean(reps_by_k[n] < 0.05)
#     ax.hist(reps_by_k[n], bins=20, density=True)
#     ax.set_title(f"{n} CFMs, $r_5 = ${r:.2f}", fontsize=8)
# fig.set_layout_engine("tight")
# fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
# fig.supxlabel("representativity")
# fig.supylabel("counts")
# for n, ax in zip(num_CFMs, axs.flat):
#     ax.set_yscale("log")
#     ax.hist(reps_by_k[n], bins=20)
#     ax.set_title(f"{n} CFMs")
# fig.set_layout_engine("tight")
# fig, ax = plt.subplots()
# ax.set_xlabel("representativity")
# ax.set_ylabel("density")
# for n in num_CFMs:
#     ax.hist(reps_by_k[n], bins=20, density=True, alpha=0.5, label=f"{n} CFMs")
# fig.set_layout_engine("tight")
# plt.legend()
# CFMs_rep_mean = [np.mean(reps_by_k[n] >= 0.05) for n in num_CFMs]
# CFMs_rep_std = [np.std(reps_by_k[n] >= 0.05) for n in num_CFMs]
# plt.plot(num_CFMs, CFMs_rep_mean, "o-")
# plt.ylabel("mean representativity $\geq 5\%$")
# plt.xlabel("# CFMs")
#!/usr/bin/env python
"""
Methods for the post-processing of topic decomposition results (:mod:`topic3d.postprocess`)
=====================================================

.. authors:: jbfiche, ggautham07, marcnol

Function reference
------------------
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from os import path
from scipy.linalg import issymmetric
from itertools import combinations
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
from sklearn.decomposition import PCA
import joblib as jl
from . import visualization


def consensus(components:np.ndarray, 
              max_deviation: float = 0.33, min_cluster_size: int = None,
              return_hierarchy: bool = False, return_clusters: bool = True):
    """ Compute consensus representations of a set of components by computing the Jensen-Shannon divergence
    between them and clustering hierarchically. Components from undersized clusters are discarded.

    Parameters
    ----------
    components : np.ndarray
        2D array of shape (n_components, n_features) representing individual components.
    max_deviation : float, default=0.33
        Maximum allowed linkage distance (Jensen-Shannon divergence) for flat clustering.
    min_cluster_size : int, optional
        Minimum number of components per cluster to be retained. Defaults to n_components // 50.
    return_hierarchy : bool, default=False
        Whether to return the full linkage matrix.
    return_clusters : bool, default=True
        Whether to return the cluster labels of individual components.

    Returns
    -------
    components_avg : list of np.ndarray
        Consensus components (cluster means) for each retained cluster.
    hierarchy : np.ndarray, optional
        Linkage matrix from hierarchical clustering (if return_hierarchy is True).
    clusters : np.ndarray, optional
        Cluster labels for each component, with -1 for discarded ones (if return_clusters is True).

    Examples
    --------
    >>> avg, linkage_matrix, cluster_labels = consensus(components, return_hierarchy=True)
    >>> len(avg)
    12  # number of consensus components retained
    """

    assert isinstance(components, np.ndarray)
    assert isinstance(max_deviation, float)
    assert isinstance(min_cluster_size, int)
    if min_cluster_size is None:
        min_cluster_size = len(components) // 50
    
    # Build hierarchy and extract flat clusters
    hierarchy = linkage(components, metric="jensenshannon", method="average", optimal_ordering=True)
    clusters = fcluster(hierarchy, criterion="distance", t=max_deviation)  # TODO change max_deviation to dg height
    
    # Filter the resultant clusters by minimum cluster size
    clabels, cfreqs = np.unique(clusters, return_counts=True)
    clabels_to_remove = clabels[cfreqs < min_cluster_size]
    for i in range(len(clabels_to_remove)):
        clusters[clusters == clabels_to_remove[i]] = -1
    clabels, cfreqs = np.unique(clusters, return_counts=True)
    print({l:v for l, v in zip(clabels, cfreqs)})
    
    # Compute and plot the consensus component representation
    components_avg = [np.mean(components[clusters == c], axis=0) \
                      for c in clabels[1:]]
    # TODO plotting grid of triangular components
    # visualization.plot_pyramid_topic(components_avg)

    result = [components_avg]
    if return_hierarchy:
        result.append(hierarchy)
    if return_clusters:
        result.append(clusters)
    return tuple(result)


def compute_divergence_between_sets(components_by_cluster: dict, path_to_save:str):
    """ Compute and visualize the average pairwise Jensen-Shannon divergence between component sets 
    from different clusters.

    Parameters
    ----------
    components_by_cluster : dict
        Dictionary mapping cluster labels (str or int) to arrays of components (shape: n_items × n_features).
    path_to_save : str
        File path to save the resulting heatmap image (e.g., "output/divergence_heatmap.png").

    Returns
    -------
    pw_mean_div : np.ndarray
        Symmetric matrix of pairwise mean Jensen-Shannon divergences between component sets.

    Examples
    --------
    >>> divergence_matrix = compute_divergence_between_sets(components_by_cluster, "output/js_divergence.png")
    >>> divergence_matrix.shape
    (5, 5)
    """
    
    cluster_labels = list(components_by_cluster.keys())
    cluster_pairs = list(combinations(cluster_labels, 2))
    pw_mean_div = np.empty(len(cluster_pairs))
    for i, (c0, c1) in enumerate(cluster_pairs):
        pw_mean_div[i] = np.mean(cdist(components_by_cluster[c0],
                                              components_by_cluster[c1],
                                              metric="jensenshannon"))
    pw_mean_div = squareform(pw_mean_div)
    np.fill_diagonal(pw_mean_div, [np.mean(pdist(components_by_cluster[cluster], metric="jensenshannon")) \
                           for cluster in cluster_labels])
    assert issymmetric(pw_mean_div)

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(8)
    im = ax.imshow(pw_mean_div, cmap="Blues")
    cbar = fig.colorbar(im, shrink=0.85)
    cbar.set_label("mean Jensen-Shannon divergence", labelpad=6)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(np.arange(len(cluster_labels)), labels=cluster_labels);
    ax.set_yticks(np.arange(len(cluster_labels)), labels=cluster_labels);
    ax.tick_params("x", labelsize=10)
    ax.tick_params("y", labelsize=10)
    ax.set_xlabel("Cluster #", labelpad=10)
    ax.set_ylabel("Cluster #", labelpad=10)
    plt.tight_layout()
    fig.savefig(path_to_save, dpi=300)

    return pw_mean_div


def embed_components(components, path_to_save_dir: str,
                     method: str = "UMAP", cluster: bool = False,
                     plot_representations: bool = False, **kwargs):
    """ Embed high-dimensional component vectors into 2D using PCA or UMAP, optionally cluster them, 
    and visualize the results. Clustering uses HDBSCAN if enabled.

    Parameters
    ----------
    components : np.ndarray
        Input array of shape (n_components, n_features) to embed.
    path_to_save_dir : str
        Directory path to save figures and model files.
    method : str, default="UMAP"
        Dimensionality reduction method to use. Options: "UMAP", "PCA".
    cluster : bool, default=False
        Whether to perform clustering on the embedded data using HDBSCAN.
    plot_representations : bool, default=False
        Whether to plot averaged representations for each cluster (requires cluster=True).
    **kwargs : dict
        Additional parameters:
        - For UMAP:
            - num_neighbors : int
            - min_dist : float
            - metric : str, default="cosine"
        - For clustering:
            - min_cluster_size : int, optional
            - min_samples : int, optional
            - cluster_selection_method : str, optional
            - alpha : float, optional
        - num_components : int, optional
            Number of embedding dimensions (default=2).

    Returns
    -------
    embedding_df : pd.DataFrame
        DataFrame containing the 2D embedding and optional cluster assignments.

    Examples
    --------
    >>> embed_df = embed_components(components, "results/", method="UMAP", cluster=True,
    ...                             num_neighbors=15, min_dist=0.1, min_cluster_size=10)
    >>> embed_df.head()
         UMAP_1   UMAP_2  cluster
    0    -0.23     0.57         2
    1     0.42     0.91         2
    """

    assert isinstance(path_to_save_dir, str) and path.isdir(path_to_save_dir)
    assert isinstance(method, str)
    assert isinstance(cluster, bool)
    assert isinstance(plot_representations, bool)
    if plot_representations:
        assert cluster

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
        embedding = reducer.fit_transform(components)

    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(components)
        
    # for plotting with sns
    embedding_df = pd.DataFrame(embedding, columns=[f"{method_label}_{i}" \
                                                    for i in range(1, num_components_embedding+1)])
    # Saving the model as joblib file
    jl.dump(reducer, path.join(path_to_save_dir, f"model_components_{method}.jl"))

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
    fig.savefig(path.join(path_to_save_dir, f"embedding_components_{method}.png"), dpi=300)
    plt.close(fig)
    
    if cluster:
        # HDBSCAN parameters for the pooled components
        min_cluster_size = kwargs.get("min_cluster_size", components.shape[0] // 40)
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
        plt.savefig(path.join(path_to_save_dir, f"embedding_components_{method}_clustered.png"), dpi=300)
        plt.close()

        if plot_representations:
            # for each cluster, plot the average topic associated
            components_avg = np.zeros((num_clusters, components.shape[-1]))
            for i, cluster in enumerate(clusters_unique):
                idxs = np.squeeze(np.argwhere(clusterer.labels_ == cluster))
                if len(idxs) > 0:
                    components_avg[i,:] = np.nanmean(np.take(components, indices=idxs, axis=0), axis=0)
            # TODO function to plot the averaged components in triangle to replace JB's older one
            # plot_mean_topics(components_avg, clusters=clusters_unique, colormap="Reds", saving_filename=fig_path)

    return embedding_df


# # TODO common set of functions for embedding which can be reused for optimization
# # TODO finish code for optimization of UMAP and/or t-SNE embeddings
# if method.lower().strip() == "umap":
#     try:
#         from umap import UMAP
#         if cluster:
#             from hdbscan import HDBSCAN
#     except ImportError:
#         raise
    
#     # UMAP parameters list
#     metric = "cosine"
#     n_umap_components = 2
#     # min_dist_all = [0.01, 0.1, 1]
#     min_dist_all = [0.2, 0.4, 0.6, 0.8]
#     # num_neighbors_all = [2, 22, 42, 62, 82]
#     num_neighbors_all = [4, 8, 12, 16, 20]
#     param_sets = list(product(min_dist_all, num_neighbors_all))

#     fig, axes = plt.subplots(nrows=len(min_dist_all), ncols=len(num_neighbors_all), sharex=True, sharey=True)
#     fig.set_figwidth(len(num_neighbors_all) * 4)
#     fig.set_figheight(len(min_dist_all) * 4)

#     for i, (min_dist, num_neighbors) in enumerate(param_sets):

#         reducer = UMAP(
#             n_neighbors=num_neighbors,
#             min_dist=min_dist,
#             metric=metric,
#             n_components=2,
#             random_state=42,
#         )
#         embedding = reducer.fit_transform(self.decomposition)
#         # jl.dump(reducer, path.join(dir_results,
#         #                         "optimization",
#         #                         f"{tag_wt}-{tag_mut}-combined-{num_components_wt}-{num_components_mut}_UMAP-components" \
#         #                         f"_model_md={min_dist}_nn={num_neighbors}.jl"))
#         embedding_df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])

#         seaborn.scatterplot(data=embedding_df,
#                         x="UMAP_1", y="UMAP_2",
#                         s=10, alpha=0.4, color="#216eff", edgecolor="none",
#                         legend=False,
#                         ax=axes.flat[i])
#         axes.flat[i].set_title(f"min_dist={min_dist}, num_neighbors={num_neighbors}")

#     fig.tight_layout()
#     # fig.savefig(path.join(dir_results,
#     #                     "optimization",
#     #                     f"{tag_wt}-{tag_mut}-combined-{num_components_wt}-{num_components_mut}_UMAP-components_grid_1.png"),
#     #                     dpi=300)
#     plt.close(fig)

# elif method.lower() == "pca":
#     reducer = PCA(n_components=2, perplexity=40, random_state=42)
#     embedding = reducer.fit_transform(self.decomposition)
#     embedding_df = pd.DataFrame(embedding, columns=["C1", "C2"])
#     jl.dump(reducer, path_to_save)


# # HDBSCAN parameters for the pooled components
# min_size = 50
# min_sample = 1      # increase it to break apart long clusters
# method = "leaf"     # "eom"  # "leaf"
# alpha = 1.0     # strength of distance scaling — higher = stricter merging

# if cluster:
#     clusterer = HDBSCAN(
#         min_cluster_size=min_size,
#         min_samples=min_sample,
#         cluster_selection_method=method,
#         alpha=alpha,
#         prediction_data=True
#     )
#     clusterer = clusterer.fit(embedding)
#     embedding_df["cluster"] = clusterer.labels_

#     clusters_unique = sorted(list(np.unique(clusterer.labels_).astype(int)))
#     clusters_unique.append(clusters_unique.pop(0))
#     clusters_unique = np.asarray(clusters_unique)
#     num_clusters = len(clusters_unique)     # noise is included as a cluster
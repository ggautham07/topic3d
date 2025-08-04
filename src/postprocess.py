#!/usr/bin/env python
"""
Methods for the post-processing of topic decomposition results (:mod:`topic3d.postprocess`)
=====================================================

.. authors:: jbfiche, gsgautham98, marcnol

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
import visualization


# TODO implementation using jsd + linkage or any other suitable methods
def consensus(components, max_deviation=0.2, min_cluster_size: int = None, return_clusters: bool = True):
    """Compute consensus representations of components merged by clustering of the latter pooled
    from several trainings using Jensen-Shannon divergence or Pearson's correlation
    """

    # assert isinstance(method, str)
    if min_cluster_size is None:
        min_cluster_size = len(components) // 10

    hierarchy = linkage(components, metric="jensenshannon", method="centroid", optimal_ordering=True)
    clusters_init = fcluster(hierarchy, criterion="distance", t=max_deviation)
    cluster_labels, counts = np.unique(clusters_init, return_counts=True)
    cluster_labels_to_remove = cluster_labels[counts < min_cluster_size]
    # replace cluster numbers below min size with NaN
    clusters = np.where(clusters_init == cluster_labels_to_remove, np.nan, clusters_init)
    cluster_labels = np.delete(cluster_labels, cluster_labels_to_remove)
    components_avg = [np.mean(components[clusters == n_cluster], axis=0) \
                      for n_cluster in cluster_labels]
    # TODO plotting grid of traingular components
    # visualization.plot_pyramid_topic(components_avg)
    
    if return_clusters:
        return components_avg, clusters
    return components_avg


def compute_divergence_between_sets(components_by_cluster: dict, path_to_save:str):
    cluster_labels = list(components_by_cluster.keys())
    cluster_pairs = list(combinations(cluster_labels, 2))
    pw_mean_divergence = np.empty(len(cluster_pairs))
    for i, (c0, c1) in enumerate(cluster_pairs):
        pw_mean_divergence[i] = np.mean(cdist(components_by_cluster[c0],
                                              components_by_cluster[c1],
                                              metric="jensenshannon"))
    pw_mean_divergence = squareform(pw_mean_divergence)
    np.fill_diagonal(pw_mean_divergence, [np.mean(pdist(components_by_cluster[cluster], metric="jensenshannon")) \
                           for cluster in cluster_labels])
    assert issymmetric(pw_mean_divergence)

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(8)
    im = ax.imshow(pw_mean_divergence, cmap="Blues")
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

    return pw_mean_divergence


def embed_components(components, path_to_save_dir: str,
                     method: str = "UMAP", cluster: bool = False,
                     plot_representations: bool = False, **kwargs):
    """Embed components (usually pooled) using Principal Component Analysis (PCA), UMAP, or t-SNE and cluster them"""

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
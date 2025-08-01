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
import seaborn
from sklearn.decomposition import PCA
import joblib as jl


# TODO implementation using jsd and correlation (not recommended any more) or any other suitable methods
def consensus(self, method="jsd"):
    """Compute consensus representations of components merged by clustering of the latter pooled
    from several trainings using Jensen-Shannon divergence or Pearson's correlation
    """

    return


def compute_divergence_between_sets(clusters, components_by_cluster, path_to_save:str):
    cluster_pairs = list(combinations(clusters, 2))
    div = np.empty(len(cluster_pairs))
    for i, (c0, c1) in enumerate(cluster_pairs):
        div[i] = np.mean(cdist(components_by_cluster[c0], components_by_cluster[c1], metric="jensenshannon"))
    div = squareform(div)
    np.fill_diagonal(div, [np.mean(pdist(components_by_cluster[cluster], metric="jensenshannon")) for cluster in clusters])
    assert issymmetric(div)

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
    fig.savefig(path_to_save, dpi=300)


def embed_components(components, path_to_save_dir: str, method: str = "UMAP", cluster: bool = False, **kwargs):
    """Embed components (usually pooled) using Principal Component Analysis (PCA), UMAP, or t-SNE and cluster them"""

    assert isinstance(path_to_save_dir, str) and path.isdir(path_to_save_dir)
    assert isinstance(method, str)
    assert isinstance(cluster, bool)

    method = method.strip().lower()

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
            print(f"Parameter values for {method.lower()} not found, supply using a dictionary")
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
        # Prepare df for plotting
        embedding_df = pd.DataFrame(embedding, columns=[f"UMAP_{i}" for i in range(1, num_components_embedding+1)])

    elif method == "pca":
        reducer = PCA(n_components=2, perplexity=40, random_state=42)
        embedding = reducer.fit_transform(components)
        embedding_df = pd.DataFrame(embedding, columns=[f"PC_{i}" for i in range(1, num_components_embedding+1)])

    # Saving the model as joblib file
    jl.dump(reducer, path.join(path_to_save_dir, f"model_{method}.jl"))

    # Plotting the results
    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(8)
    seaborn.scatterplot(data=embedding_df,
            x="UMAP_1", y="UMAP_2",
            s=10, alpha=0.4, color="#216eff", edgecolor="none",
            legend=False,
            ax=ax)
    ax.set_title(method)
    fig.tight_layout()
    fig.savefig(path.join(path_to_save_dir, f"embedding_{method}.png"), dpi=300)
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

        # TODO plot embedding with clusters; can be copied from Bintu data analysis scripts
        # clusters_unique = sorted(list(np.unique(clusterer.labels_).astype(int)))
        # clusters_unique.append(clusters_unique.pop(0))
        # clusters_unique = np.asarray(clusters_unique)
        # num_clusters = len(clusters_unique)     # noise is included as a cluster

    return embedding_df



# # TODO common set of functions for embedding which can be reused for optimization
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
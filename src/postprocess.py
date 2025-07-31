import numpy as np


# TODO implementation using jsd and any other, if any
def consensus(self, method="jsd"):
    """"""

    return

def jsd_cluster(self, clusters, components_by_cluster, path_to_save:str):
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
    fig.savefig(path_to_save, dpi=300)

    assert issymmetric(div)

# TODO
def embed(components_all):
    """Embedding all components"""
    
    return
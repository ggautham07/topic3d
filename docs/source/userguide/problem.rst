Problem
==============

Understanding chromatin organisation at the single-cell level remains a major
challenge in genome biology. Chromatin tracing experiments provide rich
spatial information by measuring pairwise proximities between genomic loci
within individual cells. However, extracting biologically meaningful patterns
from these datasets is far from trivial.

High-dimensional and sparse data
-------------------------------

Each cell is represented by a proximity matrix describing interactions between
many genomic loci. These matrices are:

- high-dimensional (quadratic in the number of loci)
- sparse and noisy
- variable across cells

This makes direct comparison between cells difficult and limits the usefulness
of simple averaging approaches.

Cell-to-cell heterogeneity
--------------------------

Chromatin organisation is inherently heterogeneous. Even within a genetically
identical population, cells can exhibit distinct structural configurations.

Traditional approaches often:

- average across cells, masking variability
- focus on predefined features (e.g. distances, compartments)
- fail to capture combinatorial structural patterns

As a result, important structural states may remain hidden.

Lack of interpretable representations
------------------------------------

Many computational methods applied to chromatin tracing data rely on:

- dimensionality reduction (e.g. PCA, UMAP)
- clustering approaches
- heuristic feature extraction

While useful, these methods often:

- do not provide interpretable components
- are sensitive to preprocessing choices
- do not explicitly model co-occurrence patterns of proximities

This limits their ability to reveal recurrent chromatin conformations.

Need for pattern discovery across cells
--------------------------------------

A central question is:

    *Are there recurring chromatin folding patterns shared across cells,
    and how can we identify them in an unsupervised and interpretable way?*

Answering this requires a method that:

- captures co-occurrence of proximities
- identifies latent structural motifs
- preserves single-cell resolution
- scales to large datasets

This is the core problem addressed by ``topic3d``.
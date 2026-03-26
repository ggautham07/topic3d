Approach
=====================

To address the challenges of analysing chromatin tracing data, ``topic3d``
adopts a probabilistic framework based on **topic modelling**, a class of
methods originally developed for text analysis.

Conceptual mapping
------------------

The key idea is to reinterpret chromatin tracing data through an analogy with
documents and words:

- **Cells** are treated as *documents*
- **Pairwise proximities** between loci are treated as *words*
- **Topics** correspond to *chromatin folding motifs (CFMs)*

Under this framework, each cell is represented as a mixture of latent CFMs,
and each CFM corresponds to a characteristic pattern of proximities.

This formulation allows the model to capture **co-occurrence patterns** of
proximities across many single cells.

Latent Dirichlet Allocation
---------------------------

``topic3d`` uses Latent Dirichlet Allocation (LDA) to infer these latent
structures.

LDA provides:

- a probabilistic model of how CFMs generate observed proximities
- a decomposition of each cell into a mixture of CFMs
- interpretable components representing recurrent structural patterns

Unlike purely geometric or distance-based methods, LDA explicitly models
the joint occurrence of multiple proximities, enabling the discovery of
complex chromatin configurations.

Why topic modelling?
-------------------

Topic modelling is particularly well suited for chromatin tracing data because:

- it captures **combinatorial structure** (co-occurrence of contacts)
- it is **unsupervised**, requiring no prior annotation
- it provides **interpretable components**
- it naturally handles **heterogeneity across cells**

In contrast to standard dimensionality reduction methods, topic models yield
components that can be directly interpreted as structural motifs.

From CFMs to biological insight
-------------------------------

The inferred chromatin folding motifs (CFMs) can be used to:

- identify recurrent structural states across cells
- compare conditions (e.g. wild-type vs perturbation)
- reveal changes in chromatin organisation linked to biological processes
- generate low-dimensional embeddings for visualization

Each cell is described by its mixture of CFMs, providing a compact and
interpretable representation of its chromatin organisation.

Summary
-------

By leveraging topic modelling, ``topic3d`` transforms high-dimensional,
heterogeneous proximity data into a set of interpretable chromatin folding
motifs and per-cell representations.

This enables systematic exploration of chromatin structure at the
single-cell level, while preserving variability and uncovering recurrent
patterns that are not accessible through traditional approaches.
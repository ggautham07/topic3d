Interpretation
========================

Once a topic model has been trained using ``topic3d``, the central task is to
interpret the inferred chromatin folding motifs (CFMs) and their distribution
across single cells.

This step bridges the gap between computational outputs and biological insight.

Model outputs
-------------

The topic modelling framework produces two primary outputs:

**CFMs (topics)**
    Each CFM is a distribution over pairwise proximities between genomic loci.
    It represents a recurring chromatin folding pattern observed across cells.

**Cell-wise decomposition**
    Each cell is described as a mixture of CFMs, indicating how strongly each
    structural motif contributes to that cell’s chromatin organisation.

Together, these outputs provide a compact and interpretable representation of
single-cell chromatin structure.

Understanding CFMs
------------------

Each CFM corresponds to a characteristic pattern of spatial proximities.

These patterns can reflect:

- specific long-range contacts between loci
- domain-like structures or boundaries
- coordinated interactions across multiple genomic regions

Importantly, CFMs are defined by **co-occurring proximities**, not isolated
pairwise interactions. This allows them to capture higher-order structural
features that are difficult to detect with pairwise analyses alone.

CFMs should therefore be interpreted as **structural motifs**, rather than
individual contacts.

Interpreting cell states
-----------------------

The decomposition of each cell into CFMs provides a low-dimensional
representation of its chromatin organisation.

Cells with similar CFM mixtures:

- share similar structural configurations
- may correspond to similar functional or regulatory states

Conversely, differences in CFM composition can reveal:

- structural heterogeneity within a population
- transitions between chromatin states
- condition-specific changes (e.g. perturbations)

This representation enables comparison across cells without losing
single-cell resolution.

Relating CFMs to genomic features
---------------------------------

CFMs can be mapped back to genomic coordinates by examining the loci involved
in their high-probability proximities.

This allows CFMs to be associated with:

- specific genomic regions
- known structural features (e.g. domains or compartments)
- regulatory elements or gene loci

Such mappings can help link structural motifs to functional interpretations.

Comparing conditions
--------------------

A key application of CFMs is the comparison of chromatin organisation across
conditions.

By analysing differences in:

- CFM distributions across cells
- the prevalence of specific CFMs
- the composition of cell populations

it is possible to identify structural changes associated with:

- genetic perturbations
- protein depletion (e.g. cohesin-related effects)
- developmental or environmental changes

This enables a systematic analysis of how chromatin structure responds to
biological perturbations.

Visualisation and downstream analysis
-------------------------------------

The CFM-based representation can be used for downstream analyses such as:

- dimensionality reduction (e.g. UMAP)
- clustering of cells based on structural similarity
- trajectory or state-transition analysis
- identification of subpopulations

These analyses operate on the CFM decomposition rather than raw proximity
matrices, providing a more interpretable and robust representation.

Practical considerations
-----------------------

When interpreting CFMs, it is important to:

- select an appropriate number of CFMs (see the corresponding tutorial)
- consider model stability across runs
- combine quantitative metrics with biological knowledge
- avoid over-interpreting low-weight or noisy motifs

Interpretation should always be guided by both statistical evidence and
biological plausibility.

Summary
-------

``topic3d`` transforms complex, high-dimensional proximity data into a set of
interpretable chromatin folding motifs and per-cell representations.

By analysing:

- the structure of CFMs
- their distribution across cells
- and their variation across conditions

users can uncover recurrent patterns of chromatin organisation and relate them
to underlying biological processes.
# topic3d

**Python package for the decomposition and analysis of chromatin tracing data using topic modeling**

> [!NOTE]
> This project is under active development. The API is not yet stable and will change frequently.

---

## Overview

`topic3d` enables the analysis and interpretation of 3D chromatin tracing data using Latent Dirichlet Allocation (LDA), which captures recurrent Chromatin Folding Motifs (CFMs) based on pairwise proximity maps.

---

## Installation

The package is currently in private development and **must be installed from source**.

```bash
git clone https://github.com/ggautham07/topic3d.git
cd topic3d
pip install .
```

Dependencies include:  
`numpy`, `scipy`, `matplotlib`, `seaborn`, `pandas`, `scikit-learn`, `umap-learn`, `hdbscan`, `joblib`, `h5py`

---

## Usage

To run the full pipeline (train an LDA model and decompose data) on a compatible Fish Omics Format for Chromatin Tracing (FOF-CT) dataset:

```
topic3d runall \
    --data_train_file path/to/data/file.npy \
    --results_dir path/to/results/folder \
    --proximity_threshold 150 \     # in nm
    --num_CFMs 15 \
    --max_iter 100 \
    --evaluate_learning
```

---

## Authors

- Gautham Ganesh ([ggautham07](https://github.com/ggautham07))  
- Jean-Bernard Fiche ([jbfiche](https://github.com/jbfiche))  
- Marcelo Nollmann ([marcnol](https://github.com/marcnol))

---
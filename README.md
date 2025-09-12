# topic3d

**Python package for the decomposition and analysis of chromatin tracing data using topic modeling**

> [!NOTE]
> This project is under active development. The API is not yet stable and may change frequently.

---

## Overview

`topic3d` enables the processing, analysis, and interpretation of 3D chromatin tracing data using **Latent Dirichlet Allocation (LDA)**. It is designed for experimentalists and computational biologists working with *in situ* hybridization, polymer simulations, or other sources of 3D spatial data.

The package allows you to:

- Load and preprocess raw spatial traces from `.npy` or `.csv` (Fish Omics Format, FOF)
- Compute pairwise distance vectors and detection efficiency
- Apply **LDA decomposition** to identify latent structural "topics"
- Analyze components for coherence, representativity, and divergence
- Impute missing coordinates, normalize data, and visualize pairwise/proximity matrices
- Embed and cluster samples using UMAP / PCA / t-SNE + HDBSCAN
- Easily export figures and results for downstream interpretation

---

## Installation

The package is currently in private development and **must be installed from source**.

```bash
git clone https://github.com/ggautham07/topic3d.git
cd topic3d
pip install -e .
```

Dependencies include:  
`numpy`, `scipy`, `matplotlib`, `seaborn`, `pandas`, `scikit-learn`, `umap-learn`, `hdbscan`, `joblib`, `h5py`

---

## Usage

### Load and process raw 3D trace data

```python
import topic3d.handlers as t3h

# Load Fish Omics Format (FOF) data
raw_data = t3h.RawData()
raw_data.load("data/messina2024_fof_core.csv")
processed = raw_data.process()

# Access processed attributes
print(processed.traces.shape)
print(processed.pwds.shape)
```

### Analyze a trained LDA model

```python
from topic3d.handlers import DecompositionResult
import joblib

# Load previously trained model and results
model = joblib.load("results/model_lda.jl")
decomp = np.load("results/decomposition.npy")

res = DecompositionResult(model, decomp)
res.plot_components(normalized=True, path_to_save="figs/components.svg")
res.plot_decomposition(path_to_save="figs/decomposition.svg")
```

### Visualize 3D structure and proximity

```python
# Impute missing coordinates (naive neighbor interpolation)
imputed = processed.traces.impute(method="naive")

# Plot barcode detection efficiency
fig = imputed.plot_barcode_efficiency(path_to_save="figs/efficiency.png")
```

### Component analysis and embedding

```python
# Sort components by representativity
sorted_idx, freqs = res.sort_components(return_freqs=True)

# Embed and cluster with UMAP + HDBSCAN
embedding_df = res.embed_decomposition(
    dir_to_save="results/",
    method="UMAP",
    cluster=True,
    num_neighbors=15,
    min_dist=0.1,
)
```

---

## Project Structure

```
topic3d/
│
├── datatypes/             # CoordsArrayStack and PWDistArrayStack classes
├── handlers/              # RawData, ProcessedData, DecompositionResult
├── visualization/         # Plotting utilities for traces and matrices
├── models/                # (Optional) Topic modeling and decomposition pipeline
└── tests/                 # Unit tests
```

---

## Authors

- Jean-Bernard Fiche ([jbfiche](https://github.com/jbfiche))  
- Gautham Ganesh ([ggautham07](https://github.com/ggautham07))  
- Marcelo Nollmann ([marcnol](https://github.com/marcnol))

---

## License
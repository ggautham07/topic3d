# topic3d

Python package for the decomposition and analysis of chromatin tracing data using topic modeling


## Brief description

topic3d applies Latent Dirichlet Allocation (LDA) to collections of pairwise single-cell proximity vectors gathered from experiments or simulations to capture and quantify the co-occurrence of features across the collection. It also includes useful methods for further analysis of results from trained LDA models, which, along with complementary data from other experiments, significantly improves inference in a biological context.


## Installation

The package is still under active development and cannot be installed at the moment


## Usage

Import full package or submodules

```
import topic3d

```

```
import topic3d.handlers as t3h

```

Load raw data and process it

```
data_raw = t3h.RawData()    # create instance of raw data object
data_raw.load("./data/raw/messina2024Chromatin_fof_core.csv")     # load a Fish Omics Format (FOF) core file
data_processed = data_raw.process()     # create HDF5 file with all processed data
for x in data_processed.keys():
    if data_processed[x] is not None:
        print(x)     # check which data were generated 

```


## Authors

Jean-Bernard Fiche, Gautham Ganesh, Marcelo Nollmann


## License
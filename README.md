# topic3d

#### Analysis of chromatin tracing data using topic modelling


## Directory structure and files

- data
- src
	- handlers.py
	- visualization.py
- exec
	- run.py
- tests
	- test.py
- README.md


## Formats

### Raw data

- FISH Omics Format (FOF) - separate class
- ECSV file - classic, separate class if necessary
- Numpy trace file - no separate class, direct import into raw data object

### Processed data

- One file has all data, one file for results
- Trace coordinates, pairwise distances, barcode efficiencies, metadata, etc.
- Dataset summary (Markdown or HTML file)
	- Barcode efficiency plots and maps - by barcode (proportion of times detected across the dataset) and by trace (number of detected barcodes in each)
	- Physical vs genomic distance plot
	- Mean and median pairwise distance map
	- Distribution of pairwise distances, with KDE, then find datasets where both tracing and Hi-C are available, where is the maximum correlation, any link to threshold on the distribution?
	- Mean proportion of single-cell maps in contact as a function of the contact threshold
	- Contact map examples (maybe 20 or so), for different thresholds?

# Log maintenance

Not sure if log can be maintained by tag
Can be maintained once for the data, and each time for each training?
Plain-text file giving sufficient context

## Results

- Summary of training MD file, same as before or new file?
- Trained model
- Decomposition: components and weights plotted
- Decomposition matrix distribution of values to find a suitable, justifiable cutoff for the representativity
- Jensen-Shannon divergence between components, preferably clustered; extra - its evolution over the course of training
- perplexity evolution over the course of training

## Post-processing

- Coherence, finding the right number of topics for different top contacts
- consensus representations using Jensen-Shannon divergence clustered
- Embedding and clustering of components using UMAP
- Embedding of decomposition matrix and reconstruction using UMAP
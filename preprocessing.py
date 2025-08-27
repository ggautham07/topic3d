import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from os import path, mkdir, makedirs
from glob import glob
# from argparse import ArgumentParser
# from scipy.spatial.distance import pdist, squareform
from sys import stdout
# from matplotlib import colormaps
from classes import CoordsArrayStack, PairwiseArrayStack, RawData, ProcessedData
from copy import copy, deepcopy


def compute_traces(n_bc, raw_data, tr_ids):
    """
    @param mess: (str) message for the progress bar
    @param raw_data: (dataframe) contains all the coordinates extracted from the fof file
    @param tr_ids: (list) uid of each single trace
    @param n_bc: (int) number of barcodes
    @return: tr_eff (np array) barcode efficiency of each trace
             tr_data (numpy array) contains each individual traces
             imputed_tr_eff (np array) barcode efficiency of each imputed trace
             imputed_tr_data (numpy array) contains each individual imputed trace
    """
    tr_eff = np.empty((len(tr_ids), ))
    tr_data = np.empty((len(tr_ids), n_bc, 3))
    tr_data[:] = np.nan

    # from the coordinates, create the trace
    for i in range(len(tr_ids)):
        curr_data = raw_data[raw_data["Trace_ID"] == tr_ids[i]]
        coords = curr_data[["barcode_id", "X", "Y", "Z"]].to_numpy()
        tr_eff[i] = len(coords) / n_bc
        for coord in coords:
            tr_data[i, coord[0], :] = coord[1:]

    return tr_data, tr_eff


def setup_logging(log_path):
    """ Setup a log file where all the general information on the processed dataset will be saved
    @param log_path (str) indicate the full path where to create and save the log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(stdout)
        ]
    )

root_dir = "./"
data_dir = path.join(root_dir, "data", "raw")
args = sys.argv
tag = args[1].strip().lower()

assert path.exists(root_dir)
assert path.exists(data_dir)
# assert len(tag.split("_")) == 

# look for replicate files with the same tag

file_paths = glob(path.join(data_dir, f"{tag}*"))
if len(file_paths) > 0:
    num_replicates = len(file_paths)
    print(f"{num_replicates} files with the tag '{tag}' found")
else:
    print(f"No files with the tag '{tag}' found")
    del file_paths
    raise FileNotFoundError()
ext = file_paths[0].split(".")[-1]
assert ext == "npy" or "csv"
if num_replicates > 1:
    assert np.all([fp.endswith(ext) for fp in file_paths[1:]])

info_dir = path.join(root_dir, "logs")
makedirs(info_dir, exist_ok=True)
setup_logging(path.join(info_dir, f"{tag}.txt"))
logging.info(f"{tag}")
logging.info("=" * 50)
logging.info("Trace information")
logging.info("=" * 10)
logging.info(f"number of replicates: {num_replicates}")


# ------------------------------------------------------------------------------------------------------------------
# Load files and check the dataset (number of barcodes, etc.)
# ------------------------------------------------------------------------------------------------------------------

# # check if the trace files already exist. If not it is necessary to load the fof dataset and check them
# trace_files = glob(path.join(root_path, "data", "traces", f"{tag}_concatenated.npy"))


# open all files and define a list of barcodes (this is necessary since sometimes the number of barcodes is different between replicates)
raw_data_all = []
bc_combinations = []
for fp in file_paths:
    raw_data = RawData()
    raw_data.load(path_to_file=fp)
    raw_data_all.append(raw_data)
    if ext == "csv" and num_replicates > 1:
        start_stop_combinations = raw_data.fofdata[["Chrom", "Chrom_Start", "Chrom_End"]]
        bc_combinations.append(start_stop_combinations.drop_duplicates())
        # Unique barcode numbers
        unique_bc_combinations = pd.concat(bc_combinations).drop_duplicates()
        unique_bc_combinations = unique_bc_combinations.sort_values(by=["Chrom_Start", "Chrom_End"], ascending=True)
        num_bc = len(unique_bc_combinations)
    else:
        num_bc = raw_data.traces.shape[1]
logging.info(f"number of barcodes: {num_bc}")

if ext == "csv":
# for each file, compute the number of similar barcodes between each file
    for n, fp in enumerate(file_paths):
        shared_combinations = pd.merge(unique_bc_combinations, bc_combinations[n],
                                        on=["Chrom", "Chrom_Start", "Chrom_End"],
                                        how="inner")
        num_shared_combinations = len(shared_combinations)
        logging.info(f"for file {path.basename(fp)}, number of shared barcodes: "
                        f"{num_shared_combinations} = "
                        f"{np.around(num_shared_combinations / len(unique_bc_combinations) * 100, decimals=1)}% of the"
                        f" total number of barcodes")

    # format the set of uniques barcodes as a dataframe and save it in a specific folder
    unique_bc_combinations = unique_bc_combinations.to_numpy()
    bc_ids = np.arange(0, unique_bc_combinations.shape[0]).reshape(-1, 1)
    bc_all = pd.DataFrame(np.column_stack((bc_ids, unique_bc_combinations)),
                                columns=["barcode_id", "Chrom", "Chrom_Start", "Chrom_End"])
    inter_locus_dist = [bc_all["Chrom_Start"][n+1] - bc_all["Chrom_Start"][n]
                        for n in range(num_bc - 1)]
    logging.info(f"median distance between consecutive barcodes: "
                    f"{np.around(np.median(inter_locus_dist) / 1000, decimals=1)}kb")

    traces_all = []
    bc_effs_all = []
    for n_exp, raw_data in enumerate(raw_data_all):
        # assign the barcodes id based on the values of 'Chrom_Start' and 'Chrom_End'
        data = pd.merge(raw_data.fofdata, bc_all, on=["Chrom", "Chrom_Start", "Chrom_End"], how="left")
        trace_ids = np.unique(data["Trace_ID"])

        # format the traces and compute the efficiency for all traces
        trace_data, efficiency = compute_traces(num_bc, data, trace_ids)
        # logging.info(f"For file {path.basename(fp)}, {len(trace_data)} traces were found")
        traces_all.append(trace_data)
        bc_effs_all.append(efficiency)

    traces_all = CoordsArrayStack(np.concatenate(traces_all, axis=0))

else:
    traces_all = np.concatenate([rd.traces for rd in raw_data_all], axis=0) if num_replicates > 1 else raw_data_all[0].traces
    traces_all = CoordsArrayStack(traces_all)
    bc_ids = np.arange(num_bc, dtype=int)
    print(traces_all.shape, np.sum([rd.traces.shape[0] for rd in raw_data_all]))
    assert traces_all.shape[0] == np.sum([rd.traces.shape[0] for rd in raw_data_all])


raw_data_pooled = RawData()
raw_data_pooled.traces = traces_all[:]
raw_data_pooled.bc_ids = bc_ids
raw_data_pooled.num_bc = num_bc
proc_data = raw_data_pooled.process()
proc_data.save(path_to_file=path.join(root_dir, "data", "processed", f"{tag}.h5"), replace=True)
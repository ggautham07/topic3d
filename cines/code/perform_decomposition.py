# -*- coding: utf-8 -*-
#

from os import path, makedirs
from glob import glob
from scipy.spatial.distance import squareform
from sklearn.decomposition import LatentDirichletAllocation
from time import perf_counter
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import uuid
import logging
import sys
import json
import joblib
from topic3d.datatypes import CoordsArrayStack, PWDistArrayStack
from topic3d.handlers import ProcessedData
from classes import DecompositionResult

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

"""

Example run:

$ python -m perform_decomposition --root_dir ~/grey/rawData_2024/Experiment_96_Topic_Gautham/topic_decomposition --tag hafner_chr6_CTCF_0 --num_jobs 2

"""


# Function to configure logging
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# Parsing arguments
def arg_parser():
    parser = ArgumentParser(prog="perform_decomposition.py",
                            description="LDA-based decomposition of chromatin structure data")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--num_components", type=int, default=10)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--hardness", type=float, required=False, default=1)
    parser.add_argument("--max_iter", type=int, choices=range(1, 10001), metavar="[1-10000]", default=4000,
                        required=False)
    parser.add_argument("--learning_method", type=str, choices=["batch", "online"], default="batch", required=False)
    parser.add_argument("--num_jobs", type=int, default=1, required=False)
    parser.add_argument("--train_data_proportion", type=float, metavar="[0.1-1]", default=1, required=False)
    parser.add_argument("--normalization_method", type=str, choices=[None, "MMSF", "R(s)"], default=None,
                        help="Method to normalize pairwise distances - dataset needs to exist in the processed data")
    parser.add_argument("--imputed", action="store_true", required=False)
    args = parser.parse_args()

    return args


# To display durations in seconds in a more readable format
def readable_duration(duration):
    remaining = int(duration)
    if remaining >= 3600:
        hours = remaining // 3600
        remaining = remaining % 3600
    else:
        hours = 0
    if remaining >= 60:
        minutes = remaining // 60
        remaining = remaining % 60
    else:
        minutes = 0
    return f"{hours} hours, {minutes} minutes and {remaining} seconds"


# Get number of rows and columns based on number of components to plot
def num_rows_cols(num_items):
    if num_items < 4:
        nrows = 1
        ncols = num_items
    else:
        sqrt = np.sqrt(num_items)
        if sqrt == int(sqrt):
            nrows = int(sqrt)
            ncols = int(sqrt)
        else:
            nrows = int(np.ceil(sqrt))
            while num_items % nrows != 0:
                nrows -= 1
            ncols = num_items // nrows
    return nrows, ncols


def remove_first_diag_from_vecstack(vec: np.ndarray):
    
    n_v = vec.shape[-1]
    n_m = np.ceil(np.sqrt(n_v * 2)).astype(int)

    idxs = [0]
    i = 0
    for n in range(n_m-1, 1, -1):
        idxs.append(i + n)
        i+=n
    
    vec_new = np.delete(vec, idxs, axis=-1)

    return vec_new


def plot_components(matrices, decomposition, training_path, tag):

    # plot components
    num_components = len(matrices)
    num_rows, num_cols = num_rows_cols(num_items=num_components)
    logging.info("Plotting components")

    dratio = 3
    fig = plt.figure(figsize=(num_cols * (dratio + 0.5) + 1, num_rows * (dratio + 0.8) + 1.5), constrained_layout=True)
    fig.suptitle(f"Topic decomposition | {tag} | $n = {decomposition.shape[0]}$", fontsize=12)
    sfig = fig.subfigures(nrows=1, ncols=1)
    # l = np.min(matrices)
    # h = np.max(matrices)
    axes = sfig.subplots(nrows=num_rows, ncols=num_cols)
    for i, ax, matrix in zip(range(num_components), axes.flat, matrices):
        # im = ax.imshow(matrix, vmin=l, vmax=h, cmap="coolwarm")
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.2)
        im = ax.imshow(matrix, cmap="Reds")
        ax.set_title(f"T{i}, $\\rho =$ {np.mean(decomposition[:, i] > 0.1):.2f}", fontsize=10, fontweight=600, y=1.02)
        m, n = matrix.shape[0], matrix.shape[1]
        ax.set_xticks(np.arange(0, m, 10, dtype=int))
        ax.set_xticks(np.arange(0.5, m, 1), minor=True)
        ax.set_yticks(np.arange(0, n, 10, dtype=int))
        ax.set_yticks(np.arange(0.5, n, 1), minor=True)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.grid(which="minor", color="#fff", lw=0.05)
        fig.colorbar(mappable=im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    # sfig.colorbar(im, ax=axes, shrink=0.5, pad=0.04, location='bottom', label="pseudocount")
    # fig.savefig(path.join(training_path, "components.svg"), format="svg", bbox_inches='tight')
    fig.savefig(path.join(training_path, "components.png"), format="png", dpi=200, bbox_inches='tight')
    plt.close(fig=fig)


def plot_decomposition(LDA_decomposition, n_topics, training_path):
    """ Plot a graph illustrating the representativity of each topic for the decompositions

    @param LDA_decomposition:
    @param n_topics:
    @param training_path:
    @return:
    """
    lda_decomposition_ordered = np.empty((0, n_topics), dtype=int)
    idx = np.argmax(LDA_decomposition, axis=1)
    for n_topic in range(n_topics):
        selection = LDA_decomposition[idx == n_topic, :]
        sorted_selection_idx = np.argsort(selection[:, n_topic])[::-1]  # sort them is descending order
        lda_decomposition_ordered = np.concatenate((lda_decomposition_ordered, selection[sorted_selection_idx, :]),
                                                   axis=0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(lda_decomposition_ordered, cmap="coolwarm", interpolation='nearest', aspect='auto')
    ax.get_xaxis().set_visible(True)
    saving_filename = path.join(training_path, f'topic_decomposition.png')
    plt.savefig(saving_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_decomposition(root_dir,
                      tag,
                      file_to_process,
                      num_components: int,
                      normalization_method,
                      imputed,
                      contact_threshold,
                      hardness,
                      max_iter,
                      learning_method,
                      batch_size,
                      num_jobs,
                      train_data_proportion,
                      unique_id,):
    
    ## Define the training folder and the log
    ## ======================================

    # Setup the output file path
    unique_id = str(unique_id)[:6]
    root_dir = root_dir.rstrip('/')
    if learning_method == "online":
        training_path_pattern = root_dir + (
            f"/training/{tag}/{num_components:02d}-components_threshold={contact_threshold}_hardness={hardness}"
            f"_maxiters={max_iter}_{learning_method}_data_prop_{train_data_proportion}_batch_{batch_size}_")
    else:
        training_path_pattern = root_dir + (
            f"/training/{tag}/{num_components:02d}-components_threshold={contact_threshold}_hardness={hardness}"
            f"_maxiters={max_iter}_{learning_method}_data_prop_{train_data_proportion}{'_imputed' if imputed else ''}")


    training_path_pattern = training_path_pattern.rstrip('/')
    training_path = training_path_pattern + f"/{unique_id}"
    #training_path = path.join(training_path_pattern, unique_id)

    #makedirs(training_path, exist_ok=True)

    print(f">>> training path: {training_path_pattern}\n{unique_id}\n{training_path}")
    makedirs(training_path, exist_ok=True)

    # Setup the log file path (in the same folder than the training)
    log_file = path.join(training_path, f"decomposition_{tag}_{unique_id}.log")

    # Setup logging to file and console
    setup_logging(log_file)
    logging.info("Parameters " + "-" * 50)

    summary = {
        "root_dir": root_dir,
        "tag": tag,
        "file_to_process": file_to_process,
        "normalization_method": normalization_method,
        "imputed": imputed,
        "contact_threshold": contact_threshold,
        "hardness": hardness,
        "num_components": num_components,
        "max_iterations": max_iter,
        "learning_method": learning_method,
        "num_jobs": num_jobs,
        "train_data_proportion": train_data_proportion,
        "log_file": log_file,
    }

    logging.info(f"Settings summary for run: {unique_id}")
    for key in summary.keys():
        logging.info(f" {key}: {summary[key]}")

    # setting output folder
    logging.info("Data processing " + "-" * 50)
    logging.info(f"[{unique_id}] Will save trainings to: {training_path}")

## Format the data for LDA training (2D to 1D and pwd to contact)
## ==============================================================

    # Check the data proportion for the training set is valid
    try:
        assert 0.1 <= train_data_proportion <= 1
    except AssertionError:
        train_data_proportion = 1
        logging.warning(
            f"! Proportion of data used for training was invalid - reverting to default value of "
            f"{train_data_proportion}")

    # processing data based on extension - note that the original files from which the data are loaded is also saved

    print(f"Loading data from {file_to_process}")
    data_preproc = ProcessedData()
    data_preproc.load(path_to_file=file_to_process)
    data_trace = CoordsArrayStack(data_preproc.traces_imputed if imputed else data_prepoc.traces)
    logging.info("Taking the imputed trace file to filter by efficiency" if imputed else "Taking non-imputed dataset to filter by efficiency")
    idxs_filt = data_trace.filter(min_efficiency=60, return_idxs_only=True)
    if normalization_method is None:
        logging.info("No normalization, raw dataset taken")
        data_pwds = data_preproc.pwds[:]
    else:
        norm_method_mapping = {"MMSF": "distance distribution median-median scaling factor", "R(s)": "Euclidean vs genomic distance based adjustment"}
        logging.info(f"Dataset normalized by {norm_method_mapping[normalization_method]} taken")
        data_pwds = data_preproc.pwds_imputed_norm-[normalization_method] if imputed else data_prepoc.pwds_norm-[normalization_method]
    data_filt = PWDistArrayStack(np.squeeze(data_pwds[idxs_filt]))
    logging.info(f"Original PWDs shape: {data_pwds.shape}")
    logging.info(f"Filtered PWDs shape: {data_filt.shape}")
    # data_ = PairwiseArrayStack(remove_first_diag_from_vecstack(data_))
    # logging.info(f"Diagonal-removed PWD shape: {data_.shape}")
    # data_filt = PairwiseArrayStack(data_.filter(efficiency=0.1))
    # summary["min_pixel_efficiency"] = 0.1

    summary["min_efficiency"] = 60
    data = data_filt.proximity(threshold=contact_threshold, hardness=hardness, keep_NaNs=False)

    logging.info(f"Plotting examples of proximity matrices")
    data_filt.plot_proximity_random_examples(threshold=contact_threshold, hardness=hardness,
                                             num_items=25,
                                             path_to_save=path.join(training_path, "proximity_examples.png"),
                                             form="triangle", include_cbar=False)

    # computing mean contact matrix
    proximity_avg = np.mean(data, axis=0)
    logging.info(f"Plotting mean contact matrix, type/shape: {type(proximity_avg)}/{squareform(proximity_avg).shape}")
    data_filt.plot_proximity_avg(threshold=contact_threshold, hardness=hardness,
                                 path_to_save=path.join(training_path, "proximity_mean.png"), form="triangle")



## Define the training and testing sets
## ====================================

    # based on the selected data proportion, define the training and testing sets
    if train_data_proportion < 1:
        data_idxs = np.arange(data.shape[0], dtype=int)
        train_idxs = np.random.choice(data_idxs, size=int(train_data_proportion * len(data_idxs)), replace=False)
        test_idxs = np.delete(data_idxs, train_idxs)
        assert np.array_equal(data_idxs, np.sort(np.concatenate((train_idxs, test_idxs))))
        data_train = data[train_idxs]
        data_test = data[test_idxs]
        logging.info(f"[{unique_id}] Training data shape: {data_train.shape}, Testing data shape: {data_test.shape}")
    else:
        data_train = data[:]
        data_test = []
        test_data_filenames = []
        logging.info(f"[{unique_id}] Training data shape: {data_train.shape}")

    # # save a few examples of contact matrix
    # logging.info(f"Plotting examples of contact matrix")
    # for n in range(5):
    #     m = data_train[np.random.randint(data_train.shape[0]), :]
    #     m = squareform(m)
    #     plot_mean_contact_matrix(m, data_train, training_path, f'{tag}_sc_contact_map_{n}')

    # add info regarding the data to the summary dict
    summary["num_barcodes"] = squareform(proximity_avg).shape[0]
    summary["num_data_train"] = len(data_train)
    summary["num_data_test"] = len(data_test)



## Start training and save the results at the end
## ==============================================

    # instantiate the LDA model class
    logging.info(f"[{unique_id}] Getting training ready")
    try:
        if learning_method == "online":
            model = LatentDirichletAllocation(n_components=num_components,
                                              max_iter=max_iter,
                                              n_jobs=num_jobs,
                                              learning_method=learning_method,
                                              batch_size=batch_size,
                                              evaluate_every=5,
                                            #   verbose=1,
                                              perp_tol=1e-3)
        else:
            model = LatentDirichletAllocation(n_components=num_components,
                                              max_iter=max_iter,
                                              n_jobs=num_jobs,
                                              learning_method=learning_method,
                                              evaluate_every=5,
                                            #   verbose=1,
                                              perp_tol=1e-3)
    except Exception as e:
        logging.error(f"Error occurred during LDA instantiation: {str(e)}")

    # starting training
    logging.info(f"[{unique_id}] Start training...")
    start = perf_counter()
    try:
        # if train_data_proportion < 1:
        #     decomposition_train = model.fit_transform(data)
        #     decomposition_test = []
        # else:
        #     model.fit(data_train)
        #     decomposition_test = model.transform(data_test)
        #     decomposition_train = model.transform(data_train)
            
        model.fit(data_train)
        decomposition_train = model.transform(data_train)
        if len(data_test) > 1:
            decomposition_test = model.transform(data_test)
        components = model.components_
        model_score = model.score(data)
        elapsed_time = readable_duration(perf_counter() - start)
    
        # updating summary dict with results
        logging.info(f"[{unique_id}] Training finished successfully")
        results = {
            "model_num_iterations": model.n_iter_,
            "model_score": model_score,
            "model_perplexity": model.bound_,
        }
        summary.update(results)
    
        # save the model, components, weights, and key info
        decomp_results = DecompositionResult(model, decomposition_train)
        decomp_results.save(training_path)
        json.dump(summary, open(path.join(training_path, "summary.json"), mode="w"), indent=4)
        logging.info(f"[{unique_id}] Saved results in {training_path}")
    
        # plot the decomposition and the components
        plot_decomposition(decomposition_train, num_components, training_path)
        matrices = [squareform(components[i]) for i in range(components.shape[0])]
        plot_components(matrices, decomposition_train, training_path, tag)

        # logs elapsed time
        logging.info(f"[{unique_id}] Completed in {elapsed_time}")

    except Exception as e:
        logging.error(f"Error occurred during LDA training: {str(e)}")


# def main():
#     args = arg_parser()

#     # translates arguments to variables
#     root_dir = args.root_dir
#     tag = args.tag
#     num_components = args.num_components
#     contact_threshold = args.threshold
#     max_iter = args.max_iter
#     learning_method = args.learning_method
#     num_jobs = args.num_jobs
#     train_data_proportion = args.train_data_proportion  # proportion of the dataset picked for training

#     # finds and sets the file with data
#     file_to_process = glob(root_dir + f"/data/pwd/{tag}.*")

#     if len(file_to_process) == 1:
#         print(f"Using data file {file_to_process}")
#     elif len(file_to_process) > 1:
#         raise FileNotFoundError("Multiple files exist with the same tag")
#     elif len(file_to_process) < 1:
#         raise FileNotFoundError("No files found with that tag")

#     file_to_process = file_to_process[0]

#     # runs process
#     run_decomposition(root_dir, tag, file_to_process, num_components, contact_threshold, max_iter, learning_method,
#                       num_jobs, train_data_proportion)


# if __name__ == "__main__":
    
#     root_dir = "/mnt/grey/DATA/users/gautham/Workspace/topics/"
#     tag = "chen2023Structural_Hs-CNCC_CTCFdel-Sox9P"
#     num_components = 24
#     contact_threshold = 250
#     max_iter = 1000
#     learning_method = "batch"
#     num_jobs = 1
#     train_data_proportion = 1  # proportion of the dataset picked for training

#     # finds and sets the file with data
#     file_to_process = path.join(root_dir, f"data/processed/{tag}.h5")

#     run_decomposition(root_dir, tag, file_to_process, num_components, contact_threshold, max_iter, learning_method,
#                       None, num_jobs, train_data_proportion, uuid.uuid4())

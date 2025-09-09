from os import path, makedirs
import sys
import logging
import warnings
from time import perf_counter
from numbers import Number
import json
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from topic3d.handlers import DecompositionResult


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

# To display durations in seconds in a more readable format
def readable_duration(duration:Number):
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


def assign_train_test_data(unique_id, data:np.ndarray, train_data_proportion: Number = 1):
    """
    Assign train and test data randomly based on train data proportion
    """

    assert isinstance(data, np.ndarray)
    assert isinstance(train_data_proportion, Number) and 0 < train_data_proportion <= 1
    if 0 < train_data_proportion <= 0.2:
        warnings.warn(f"Train data proportion {train_data_proportion} might be too low")

    if train_data_proportion < 1:
        data_idxs = np.arange(data.shape[0], dtype=int)
        train_idxs = np.random.choice(data_idxs, size=int(train_data_proportion * len(data_idxs)), replace=False)
        test_idxs = np.delete(data_idxs, train_idxs)
        assert np.array_equal(data_idxs, np.sort(np.concatenate((train_idxs, test_idxs))))
        fit_transform_flag = False
    else:
        fit_transform_flag = True

    if fit_transform_flag:
        data_train = data[:][np.random.choice(np.arange(data.shape[0]), size=data.shape[0], replace=False)] # data[:]
        print(np.sum(data_train[0]))
        logging.info(f"[{unique_id}] Training and testing data shape: {data_train.shape}")

    if not fit_transform_flag:
        data_train = data[train_idxs]
        data_test = data[test_idxs]
        logging.info(f"[{unique_id}] Training data shape: {data_train.shape}, Testing data shape: {data_test.shape}")



def run_decomposition(data: np.ndarray, params: dict, dir_to_save: str = "", logger = None):
    """
    Run topic decomposition on data with parameters given.
    Saves results to path_to_save while actively logging.
    """
    
    assert isinstance(data, np.ndarray) and data.ndim == 2
    assert isinstance(params, dict)
    assert isinstance(dir_to_save, str)
    wflag = False
    if len(dir_to_save) > 0:
        assert path.isdir(dir_to_save)     # maybe add makedirs support?
        wflag = True

    num_items, num_pixels = data.shape

    # instantiate the LDA model class
    try:
        model = LatentDirichletAllocation(n_components=params["num_components"],
                                          max_iter=params["max_iter"],
                                          n_jobs=params["num_jobs"],
                                          learning_method=params["learning_method"],
                                          batch_size=params["batch_size"] \
                                            if params["learning_method"] == "online" else None,
                                          random_state=params["random_seed"])
    except Exception as err:
        pass
        # logging.error(f"Error occurred during LDA instantiation: {str(err)}")

    # performing the training and transformation of data to get results
    try:
        decomposition = model.fit_transform(data)
        # elapsed_time = readable_duration(perf_counter() - start)
    except Exception as err:
        pass
        # logging.error(f"Error occurred during LDA training: {str(err)}")

    try:
        results = DecompositionResult(model=model,
                                      decomposition=decomposition,
                                      data_training=np.arange(data.shape[0]),
                                      params=params)
        if wflag:
            results.save(dir_to_save=dir_to_save)
            results.plot_components(ordered=False, path_to_save=path.join(dir_to_save, "components.svg"))
            results.plot_components(ordered=True, path_to_save=path.join(dir_to_save, "components_sorted.svg"))
            results.plot_decomposition(path_to_save=path.join(dir_to_save, "decomposition.svg"), cmap="Blues")
        # results.embed_decomposition
    except Exception as err:
        pass
        # logging.error(f"Error occurred during the compiling and saving of results: {str(err)}")

    return results














def main(path_to_data,
                      tag,
                      file_to_process,
                      num_components,
                      contact_threshold,
                      max_iter,
                      learning_method,
                      num_jobs,
                      train_data_proportion,
                      unique_id,
                      batch_size=None,
                      random_seed=None,):

## Define the training folder and the log
## ======================================

    # Setup the output file path
    unique_id = str(unique_id)[:6]
    if learning_method == "online":
        training_path_pattern = path_to_data + (
            f"/training/{tag}/{num_components:02d}-components_threshold={contact_threshold}"
            f"_maxiters={max_iter}_{learning_method}_data_prop_{train_data_proportion}_batch_{batch_size}_")
    else:
        training_path_pattern = path_to_data + (
            f"/training/{tag}/{num_components:02d}-components_threshold={contact_threshold}"
            f"_maxiters={max_iter}_{learning_method}_data_prop_{train_data_proportion}_")

    training_path = training_path_pattern + f"{unique_id}"
    makedirs(training_path, exist_ok=False)

    # Setup the log file path (in the same folder than the training)
    log_file = path.join(training_path, f"decomposition_{tag}_{unique_id}.log")

    # Setup logging to file and console
    setup_logging(log_file)
    logging.info("-" * 50)

    summary = {
        "data_path": path_to_data,
        "tag": tag,
        "file_to_process": file_to_process,
        # "efficiency_file_to_process": efficiency,
        "contact_threshold": contact_threshold,
        "num_components": num_components,
        "max_iterations": max_iter,
        "learning_method": learning_method,
        "num_jobs": num_jobs,
        "train_data_proportion": train_data_proportion,
        "log_file": log_file,
        "training_path": training_path,
        # "min_barcode_efficiency": min_efficiency,
        "batch_size": batch_size,
        "random_seed": random_seed
    }

    logging.info(f"Settings summary for run: {unique_id}")
    for key in summary.keys():
        logging.info(f" {key}: {summary[key]}")

    # setting output folder
    logging.info(f"[{unique_id}] Will save trainings to: {training_path}")

## Format the data for LDA training (2D to 1D and pwd to contact)
## ==============================================================

    # Check the data proportion for the training set is valid
    try:
        assert 0.01 <= train_data_proportion <= 1
    except AssertionError:
        train_data_proportion = 1
        logging.warning(
            f"! Proportion of data used for training was invalid - reverting to default value of "
            f"{train_data_proportion}")

    # processing data based on extension
    if file_to_process.endswith(".npy"):
        raw_data = np.load(file_to_process, allow_pickle=True)
        # raw_data_efficiency = np.load(efficiency, allow_pickle=True)
        # raw_data = np.moveaxis(raw_data, -1, 0)     # if the npy files have the number of matrices along the last axis instead of the first
    elif file_to_process.endswith(".h5"):
        with File(file_to_process, mode="r") as handler:
            raw_data = handler["pairwise_distances"][:]
    else:
        raise ValueError("Unsupported file extension")

    # getting the upper triangle matrix from the first diagonal
    if raw_data.ndim == 3:  # if the data format is (num_matrices, num_barcodes, num_barcodes)
        ut_idxs = np.triu_indices_from(raw_data[0], k=1)
        assert ut_idxs[0].shape == ut_idxs[1].shape
        data = np.empty((raw_data.shape[0], ut_idxs[0].shape[0]))
        for i in range(raw_data.shape[0]):
            data[i] = raw_data[i][ut_idxs]
    elif raw_data.ndim == 2:  # if the data format is (num_matrices, ((num_barcodes * (num_barcodes - 1)) / 2)
        data = raw_data.copy()
    else:
        raise ValueError("Unsuitable data shape")

    # compute the contact matrix
    if contact_threshold > 0:
        data = np.asarray(data <= contact_threshold, dtype=int)
    elif contact_threshold == -1:
        data2 = data.copy()
        for i in range(data2.shape[0]):
            data[i] = np.max(data2[i]) - data2[i]
        data[np.isnan(data)] = 0
    elif contact_threshold == 0:
        data = 1 / (data ** 0.227)
        data[np.isnan(data)] = 0
    else:
        raise ValueError("Contact threshold not suitable")
    


    # computing mean contact matrix
    contact_matrix = squareform(
        np.mean(data.astype(float) if fit_transform_flag else data_train.astype(float), axis=0))
    np.fill_diagonal(contact_matrix, 1)

    logging.info(f"Plotting mean contact matrix, type/shape: {type(contact_matrix)}/{contact_matrix.shape}")
    plot_mean_contact_matrix(contact_matrix, data_train, training_path,
                             f'{tag}_mean_contact_map')

    # save a few examples of contact matrix
    logging.info(f"Plotting examples of contact matrix")
    for n in range(5):
        m = data_train[np.random.randint(data_train.shape[0]), :]
        m = squareform(m)
        plot_mean_contact_matrix(m, data_train, training_path,
                                 f'{tag}_sc_contact_map_{n}')

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
                                              random_state=random_seed)
        else:
            model = LatentDirichletAllocation(n_components=num_components,
                                              max_iter=max_iter,
                                              n_jobs=num_jobs,
                                              learning_method=learning_method,
                                              random_state=random_seed)
    except Exception as e:
        logging.error(f"Error occurred during LDA instantiation: {str(e)}")

    # starting training
    logging.info(f"[{unique_id}] Start training...")
    start = perf_counter()
    try:
        if fit_transform_flag:
            decomposition_train = model.fit_transform(data)
            decomposition_test = []
        else:
            model.fit(data_train)
            decomposition_test = model.transform(data_test)
            decomposition_train = model.transform(data_train)
        components = model.components_
        components_norm = components / np.sum(components, axis=1)[:, np.newaxis]      # for normalised pseudocount
        elapsed_time = readable_duration(perf_counter() - start)

        # updating summary dict with results
        logging.info(f"[{unique_id}] Training finished successfully")
        results = {
            "model_num_iterations": model.n_iter_,
            "model_score": model.score(data),
            "model_perplexity": model.bound_,
        }
        summary.update(results)

        # save the model, components, weights, and key info
        joblib.dump(model, path.join(training_path, "model.jl"))

        # compute representativities and sort matrices by the values
        rep_threshold = 0.1
        rep = np.mean(decomposition_train >= rep_threshold, axis=0)
        sort_idxs_descending = np.argsort(rep)[::-1]
        rep = np.sort(rep)[::-1]
        matrices = [squareform(components[i]) for i in range(components.shape[0])]
        matrices = [matrices[i] for i in sort_idxs_descending]
        plot_components(matrices, decomposition_train, rep, training_path, tag)

        # save components and weights after classification by representativity - save key info
        np.save(path.join(training_path, "components_sorted.npy"), components[sort_idxs_descending, :])
        np.save(path.join(training_path, "decomposition_test_sorted.npy"), decomposition_test[:, sort_idxs_descending])
        np.save(path.join(training_path, "decomposition_train_sorted.npy"), decomposition_train[:, sort_idxs_descending])
        json.dump(summary, open(path.join(training_path, "summary.json"), mode="w"), indent=4)
        logging.info(f"[{unique_id}] Saved results in {training_path}")

        # plot the decomposition
        plot_decomposition(decomposition_train, num_components, training_path)

        # logs elapsed time
        logging.info(f"[{unique_id}] Completed in {elapsed_time}")

    except Exception as e:
        logging.error(f"Error occurred during LDA training: {str(e)}")

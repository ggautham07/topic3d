from os import path, makedirs
from glob import glob
import sys
import logging
import warnings
from uuid import uuid4
from time import perf_counter
from numbers import Number
import json
import numpy as np
import h5py
from sklearn.decomposition import LatentDirichletAllocation

from topic3d.datatypes import CoordsArrayStack, PWDistArrayStack
from topic3d.handlers import ProcessedData, DecompositionResult


# Function to configure logging
def _setup_logging(fp):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(fp),
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


##################################################################################################################


def load_data(path_, tag: str = "") -> (np.ndarray | ProcessedData | dict):
    """
    Load a dataset from a Numpy or HDF5 file to yield an array, dictionary or ProcessedData object

    If directory is given, a tag must be provided. Else file path is sufficient
    """

    assert isinstance(path_, str)
    if path.isdir(path_):
        assert isinstance(tag, str)
        fp = glob(path.join(path_, tag+".npy")) + glob(path.join(path_, tag+".h5"))
        if len(fp) == 0:
            logging.error(f"No Numpy or HDF5 file(s) found in `{path_}` with tag `{tag}`")
            raise FileNotFoundError
        elif len(fp) > 1:
            warnings.warn("Both file formats found, loading the HDF5 file")
            fp = glob(path.join(path_, tag+".h5"))[0]
        else:
            fp = fp[0]       
    elif path.isfile(path_):
        fp = path_
        assert fp.endswith((".npy", ".h5"))
    else:
        logging.error(f"{path_} is not a file and does not exist")
        raise FileNotFoundError
    
    if fp.endswith(".npy"):
        return np.load(fp, allow_pickle=True)
    
    try:
        dataobj = ProcessedData()
        dataobj.load(fp)
    except AssertionError or KeyError:
        with h5py.File(fp, mode="r") as handler:
            dataobj = {x:y[:] for x, y in handler.items()}
            dataobj.update({x:y[:] for x, y in handler.attrs.items()})

    return dataobj


def prep_data(data: ProcessedData | dict, params: dict, dir_to_save: str = "", **kwargs):
    """
    Process the PWD data supplied to filter by efficiency of barcode detection and generate proximity vectors.

    Then pick data proportion
    """

    def _get_val(X, key, flag):
        return X[key][:] if flag else X.key[:]

    assert isinstance(dir_to_save, str)
    assert isinstance(params, dict)
    assert isinstance(data, (ProcessedData, dict))
    uid = kwargs.get(uid, "no-id")
    
    is_dict = True if isinstance(data, dict) else False
    pwds = _get_val(data, "pwds", is_dict)  # get PWDs whether ProcessedData object or dict is supplied
    assert isinstance(pwds, np.ndarray)
    if not isinstance(pwds, PWDistArrayStack):
        pwds = PWDistArrayStack(pwds)

    min_efficiency = params.get("min_efficiency")
    if min_efficiency is not None:
        try:
            traces = _get_val(data, "traces", is_dict)
        except KeyError or AttributeError:
            logging.error("Traces array not found in supplied data object to filter by minimum efficiency")
            raise
        assert isinstance(traces, np.ndarray)
        if not isinstance(traces, CoordsArrayStack):
            traces = CoordsArrayStack(traces)
        idxs = traces.filter(min_efficiency=min_efficiency, by="item", return_idxs_only=True)

    pwds_filt = PWDistArrayStack(pwds[idxs,:][:])
    proximity = pwds_filt.proximity(threshold=params["proximity_threshold"], hardness=params["proximity_hardness"])
    proximity_avg = np.nanmean(proximity, axis=0)    
    
    if len(dir_to_save) > 0 and path.isdir(dir_to_save):
        logging.info(f"Plotting mean proximity matrix, type/shape: {type(proximity_avg)}/{proximity_avg.shape}")
        pwds_filt.plot_proximity_avg(threshold=params["proximity_threshold"],
                                     hardness=params["proximity_hardness"],
                                     keep_NaNs=False,
                                     path_to_save=path.join(dir_to_save, "proximity_avg.svg"))
        logging.info(f"Plotting randomly-picked examples of proximity matrices")
        pwds_filt.plot_proximity_random_examples(threshold=params["proximity_threshold"],
                                                 hardness=params["proximity_hardness"],
                                                 num_items=16,
                                                 path_to_save=path.join(dir_to_save, "proximity_random_examples.svg"))

    return proximity


def split_data(data:np.ndarray, train_data_proportion: Number = 1, **kwargs):
    """
    Assign train and test data randomly based on train data proportion
    """

    uid = kwargs.get(uid, "no-id")

    assert isinstance(data, np.ndarray), "Data must be an array"
    assert isinstance(train_data_proportion, Number) and 0 < train_data_proportion <= 1
    if 0 < train_data_proportion <= 0.2:
        warnings.warn(f"Train data proportion {train_data_proportion} might be too low")

    if train_data_proportion < 1:
        data_idxs = np.arange(data.shape[0], dtype=int)
        idxs_train = np.random.choice(data_idxs, size=int(train_data_proportion * len(data_idxs)), replace=False)
        idxs_test = np.delete(data_idxs, idxs_train)
        assert np.array_equal(data_idxs, np.sort(np.concatenate((idxs_train, idxs_test))))
        flag_fit_trans = False  # True if fit_transform is to be applied by LDA
    else:
        flag_fit_trans = True

    if flag_fit_trans:
        data_train = data[:][np.random.choice(np.arange(data.shape[0]), size=data.shape[0], replace=False)] # data[:]
        logging.info(f"[{uid}] Training + testing data shape: {data_train.shape}")
        return data_train

    if not flag_fit_trans:
        data_train = data[idxs_train]
        data_test = data[idxs_test]
        logging.info(f"[{uid}] Training data shape: {data_train.shape}, testing data shape: {data_test.shape}")
        return data_train, data_test


def initialize(tag: str, params: dict, dir_base: str, dir_data: str):
    """
    Initialize the directory to train data and save results
    """

    uid = str(uuid4())[:6]

    dir_pattern = f"{params["num_components"]:02d}-components_threshold={params["proximity_threshold"]}"
    dir_training = path.join(dir_base, tag, dir_pattern, f"{uid}")
    makedirs(dir_training, exist_ok=False)

    # Setup the log file path (in the same folder as the training)
    log_file = path.join(dir_training, f"{uid}.log")

    # Setup logging to file and console
    _setup_logging(log_file)
    logging.info("-" * 50)
    logging.info(f"[{uid}] Will save trainings to: {dir_training}")

    summary = params.copy()
    summary.update({"path_to_data": dir_data,
                    "tag": tag,
                    "path_to_training": dir_training,
                    "log": log_file})

    logging.info(f"Settings summary for run: {uid}")
    for key in summary.keys():
        logging.info(f" {key}: {summary[key]}")
    json.dump(summary, open(dir_training, mode="x"), indent=4)

    return summary


def decompose(params: dict,
              data_train: np.ndarray, data_test: np.ndarray = None,
              dir_to_save: str = "",
              **kwargs):
    """
    Run topic decomposition on data with the parameters given.
    
    Saves results to dir_to_save while actively logging if it was initialized.
    """
    
    assert isinstance(params, dict)
    assert isinstance(data_train, np.ndarray) and data_train.ndim == 2
    flag_fit_trans = True
    if data_test is not None:
        flag_fit_trans = False
        assert isinstance(data_test, np.ndarray) and data_test.ndim == 2
    dim = int(np.ceil(np.sqrt(data_train.shape[-1] * 2)))
    assert dim * (dim-1) == data_train.shape[-1] * 2  # ensure that vector can be squareformed
    assert isinstance(dir_to_save, str)
    wflag = False
    if len(dir_to_save) > 0:
        assert path.isdir(dir_to_save)  # directory must be created in initialization of unique ID
        wflag = True

    uid = kwargs.get("uid", "no-id")

    # instantiate the LDA model
    logging.info(f"[{uid}] Getting training ready in {dir_to_save}")
    try:
        model = LatentDirichletAllocation(n_components=params["num_components"],
                                          max_iter=params["max_iter"],
                                          n_jobs=params["num_jobs"],
                                          learning_method=params["learning_method"],
                                          batch_size=params["batch_size"] \
                                            if params["learning_method"] == "online" else None,
                                          random_state=params["random_seed"])
    except Exception as err:
        logging.error(f"Error occurred during LDA instantiation: {str(err)}")

    # performing the training and transformation of data to get results
    try:
        logging.info(f"[{uid}] Starting training...")
        time_start = perf_counter()
        if flag_fit_trans:
            decomposition = model.fit_transform(data_train)
        else:
            model = model.fit(data_train)
            decomposition = model.transform(data_test)
        logging.info(f"[{uid}] Training finished successfully")
    except Exception as err:
        logging.error(f"Error occurred during training: {str(err)}")

    try:
        results = DecompositionResult(model=model,
                                      decomposition=decomposition,
                                      data_training=np.arange(data_train.shape[0]),
                                      params=params)
        summary = kwargs.get(summary, params)
        summary.update({"model_num_iterations": model.n_iter_,
                        "model_score": model.score(data_train if flag_fit_trans else data_test),
                        "model_perplexity": model.bound_,
                        })
        if wflag:
            results.save(dir_to_save=dir_to_save)
            results.plot_components(ordered=False, path_to_save=path.join(dir_to_save, "components.svg"))
            results.plot_components(ordered=True, path_to_save=path.join(dir_to_save, "components_sorted.svg"))
            results.plot_decomposition(path_to_save=path.join(dir_to_save, "decomposition.svg"), cmap="Blues")
            logging.info(f"[{uid}] Saved results in {dir_to_save}")
        else:
            print(*summary, sep="\n")
    except Exception as err:
        logging.error(f"Error occurred during the compiling and saving of results: {str(err)}")

    time_elapsed = readable_duration(perf_counter() - time_start)
    logging.info(f"[{uid}] Completed in {time_elapsed}")

    return results


## Start training and save the results at the end
## ==============================================


# # starting training

# try:
#     if fit_transform_flag:
#         decomposition_train = model.fit_transform(data)
#         decomposition_test = []
#     else:
#         model.fit(data_train)
#         decomposition_test = model.transform(data_test)
#         decomposition_train = model.transform(data_train)
#     components = model.components_
#     components_norm = components / np.sum(components, axis=1)[:, np.newaxis]      # for normalised pseudocount
#     time_elapsed = readable_duration(perf_counter() - time_start)

#     # updating summary dict with results
#     logging.info(f"[{unique_id}] Training finished successfully")
#     results = {
#         "model_num_iterations": model.n_iter_,
#         "model_score": model.score(data),
#         "model_perplexity": model.bound_,
#     }
#     summary.update(results)

# # compute representativities and sort matrices by the values
# rep_threshold = 0.1
# rep = np.mean(decomposition_train >= rep_threshold, axis=0)
# sort_idxs_descending = np.argsort(rep)[::-1]
# rep = np.sort(rep)[::-1]
# matrices = [squareform(components[i]) for i in range(components.shape[0])]
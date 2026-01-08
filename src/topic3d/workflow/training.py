from os import path, makedirs
import logging
from uuid import uuid4
from time import perf_counter
from numbers import Number
import json
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from topic3d.handlers import DecompositionResult

from ._logging import setup_log


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


def initialize(params: dict, dir_results: str, path_to_data: str, tag: str = None):
    """
    Initialize the directory to train data and save results
    """

    assert type(params) == dict
    assert type(tag) == str
    assert type(dir_results) == str and path.isdir(dir_results)
    uid = str(uuid4())[:6]
    tag = "".join(path.basename(path_to_data).split(".")[:-1]) if tag is None else tag.strip()
    
    # TODO set folder pattern to highlight the most important parameters OR have a function to list and pull trainings
    dir_pattern = f"{params["num_components"]:02d}-components_threshold={params["proximity_threshold"]}"
    dir_training = path.join(dir_results, tag, dir_pattern, f"{uid}")
    makedirs(dir_training, exist_ok=False)

    # Setup the log file path (in the same folder as the training)
    log_file = path.join(dir_training, f"{uid}.log")

    # Setup logging to file and console
    setup_log(log_file)
    logging.info("-" * 50)
    logging.info(f"[{uid}] Will save trainings to: {dir_training}")

    summary = params.copy()
    summary.update({"uid": uid,
                    "path_to_data": path_to_data,
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
    flag_ft = True      # Train and test data same if True - performs model.fit_transform()
    if data_test is not None:
        flag_ft = False
        assert isinstance(data_test, np.ndarray) and data_test.ndim == 2
    dim = int(np.ceil(np.sqrt(data_train.shape[-1] * 2)))
    assert dim * (dim-1) == data_train.shape[-1] * 2  # ensure that vector can be squareformed
    assert isinstance(dir_to_save, str)
    wflag = False
    if len(dir_to_save) > 0:
        assert path.isdir(dir_to_save)  # directory must be created in initialization of training with unique ID
        wflag = True

    uid = kwargs.get("uid", "no-uid")
    if uid is None:
        uid = "no-uid"

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
        if flag_ft:
            decomposition = model.fit_transform(data_train)
        else:
            model = model.fit(data_train)
            decomposition = model.transform(data_test)
        logging.info(f"[{uid}] Training finished successfully")
    except Exception as err:
        logging.error(f"[{uid}] Error occurred during training: {str(err)}")

    try:
        results = DecompositionResult(model=model,
                                      decomposition=decomposition,
                                      data_training=np.arange(data_train.shape[0]),
                                      params=params)
        summary = kwargs.get("summary", params)
        summary.update({"model_num_iterations": model.n_iter_,
                        "model_score": model.score(data_train if flag_ft else data_test),
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
        logging.error(f"[{uid}] Error occurred during the compiling and saving of results: {str(err)}")

    time_elapsed = readable_duration(perf_counter() - time_start)
    logging.info(f"[{uid}] Completed in {time_elapsed}")

    return results
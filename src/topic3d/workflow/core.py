"""
Workflow utilities for training and decomposition (:mod:`topic3d.workflow.training`).

This module contains helpers used by the command-line interface to:

- create a results directory and configure logging,
- train a scikit-learn :class:`~sklearn.decomposition.LatentDirichletAllocation` model,
- transform new datasets with an existing model,
- serialize outputs to disk via :class:`topic3d.handlers.DecompositionResult`.

The functions are intentionally lightweight wrappers around the core API.

"""

from os import path, makedirs
import logging
from io import StringIO
from contextlib import redirect_stdout
from uuid import uuid4
from time import perf_counter
from numbers import Number
import json
import numpy as np
import h5py
from sklearn.decomposition import LatentDirichletAllocation

from topic3d.results import Topic3DResult

from ._logging import setup_log, setup_log_simple, setup_log_simple2


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


def initialize(params: dict, dir_results: str, path_to_data: str | list):
    """
    Initialize a run directory and logging for a CLI invocation (simple layout).

    This variant creates a unique subdirectory under ``dir_results`` (if it exists)
    and stores a ``summary.json`` with run metadata.

    Parameters
    ----------
    params : dict
        Decomposition/training parameters. Must be JSON-serializable.
    dir_results : str
        Root directory where run outputs should be stored. If this directory does not
        exist, the run proceeds without saving outputs.
    path_to_data : str | list
        Input dataset path(s) passed by the CLI.

    Returns
    -------
    dict
        A run summary containing ``uid`` and input paths, and (when saving is enabled)
        also ``path_to_training`` and ``path_to_log``.

    See Also
    --------
    initialize_full
        Initialization with a deeper directory hierarchy based on parameters.
    """

    assert type(params) == dict
    assert type(dir_results) == str
    try:
        if dir_results.endswith("###"):
            assert path.isdir(dir_results.strip("###"))
        else:
            assert path.isdir(path.dirname(dir_results))
        save_flag = True
    except AssertionError:
        print("Directory to save results not specified or found. Results and logs will not be saved.")
        save_flag = False

    # Setting a unique ID for the training
    uid = str(uuid4())[:6]
    if save_flag:
        if dir_results.endswith("/###"):
            dir_training = dir_results.replace("/###", "/"+str(uid))
        else:
            dir_training = dir_results
        makedirs(dir_training, exist_ok=False)
        log_file = path.join(dir_training, f"logs.log")    # Setup the log file path (in the same folder as training)
        setup_log(log_file)     # Setup logging to file and console
    else:
        setup_log_simple()
        dir_training = ""
        log_file = ""

    logging.info("-" * 50)
    if save_flag:
        logging.info(f"[{uid}] Will save run outputs to: {dir_training}")
    else:
        logging.info(f"[{uid}] Results/logs will not be saved (no valid dir_results)")

    summary = params.copy()
    summary.update({"uid": uid,
                    "path_to_data": path_to_data,})
    if save_flag:
        summary.update({"path_to_training": dir_training,
                        "path_to_log": log_file,})

    logging.info(f"Settings summary for run: {uid}")
    for key in summary.keys():
        logging.info(f"    {key}: {summary[key]}")
    if save_flag:
        json.dump(summary, open(path.join(dir_training, "summary.json"), mode="x"), indent=4)

    return summary


def _check_proximity_data(data):
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    dim = int(np.ceil(np.sqrt(data.shape[-1] * 2)))
    assert dim * (dim-1) == data.shape[-1] * 2  # ensure that vector can be squareformed
    assert np.all(~np.isnan(data))
    assert np.all(data >= 0)


def _check_LDA_model(model):
    assert isinstance(model, LatentDirichletAllocation)


def train_LDA_model(params: dict,
                    data_train: np.ndarray, data_test: np.ndarray = None,
                    model: LatentDirichletAllocation = None,
                    **kwargs):
    """
    Train an LDA model on pairwise proximity data.

    Inputs must be 2D arrays where rows are items (cells/traces) and columns are
    flattened pairwise features of length ``num_bc * (num_bc - 1) // 2`` (compatible
    with :func:`scipy.spatial.distance.squareform`).

    Parameters
    ----------
    params : dict
        LDA/training parameters. Expected keys include ``num_components``, ``max_iter``,
        ``learning_method``, ``batch_size`` and ``random_state``.
    data_train, data_test : numpy.ndarray, optional
        Arrays of shape ``(n_items, n_pairs)``.
    model : sklearn.decomposition.LatentDirichletAllocation, optional
        Existing fitted model to reuse. If omitted, a new model is instantiated from
        ``params``.
    dir_to_save : str, optional
        If non-empty, save results and standard plots into this directory.
    **kwargs
        Extra metadata used by the CLI (e.g. ``uid`` and a pre-built ``summary`` dict).

    Returns
    -------
    topic3d.results.DecompositionResult
        Object containing the fitted model (possibly reused), the decomposition matrix
        (topic proportions per item), and run parameters.

    Notes
    -----
    When training is performed, the function additionally computes several coherence
    scores for different proportions of top component pixels and stores them under
    ``coherence/`` in the saved HDF5 results file.

    Raises
    ------
    AssertionError
        If inputs are malformed (e.g., incompatible vector lengths).
    Exception
        Propagates errors from model fitting, saving, or coherence computation.
    """
    
    assert isinstance(params, dict)
    existing_log = False
    if kwargs.get("logger") is None:
        setup_log_simple2(name="training-logger")
        logger = logging.getLogger("training-logger")
    else:
        assert isinstance(kwargs.get("logger"), logging.Logger)
        logger = kwargs["logger"]
        existing_log = True
    
    _check_proximity_data(data_train)
    logger.info("Training data passed conformity checks")
    # flag_test = False
    # if not (data_test is None):
    #     _check_proximity_data(data_test)
    #     logging.info("Testing data passed conformity checks")
    #     flag_test = True
    if not (model is None):
        _check_LDA_model(model)
        logger.info(f"[{uid}] Loaded LDA model")
        flag_exising_model = True
    else:
        flag_exising_model = False

    uid = kwargs.get("uid", "no-uid")
    if uid is None:
        uid = "no-uid"

    if flag_exising_model:
        logger.info("Using existing LDA model loaded from the given path")
    else:
        logger.info("No existing model supplied. New LDA model will be created")

    # instantiate the LDA model
    if not flag_exising_model:
        logging.info(f"[{uid}] Initializing LDA model")
        try:
            model = LatentDirichletAllocation(n_components=params["num_CFMs"],
                                              max_iter=params["max_iter"],
                                              evaluate_every=1 if params["evaluate_learning"] else -1,
                                              learning_method=params["learning_method"],
                                              batch_size=params["batch_size"],
                                              random_state=params["random_state"],
                                              verbose=1)
        except Exception as err:
            logging.error(f"[{uid}] Error occurred during LDA instantiation: {str(err)}")
            raise

    # performing the training and transformation of data to get results
    try:
        logging.info(f"[{uid}] Training LDA model...")
        time_start = perf_counter()
        capt_out = StringIO()
        with redirect_stdout(capt_out):
            model = model.fit(data_train)
        output = capt_out.getvalue()
        time_elapsed = readable_duration(perf_counter() - time_start)
        logging.info(f"[{uid}] Training finished successfully in {time_elapsed}")
    except Exception as err:
        logging.error(f"[{uid}] Error occurred during training: {str(err)}")

    # TODO close_log function in _logging.py
    if not existing_log:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    if params["evaluate_learning"]:
        term = "perplexity: "
        perps = [float(line[line.find(term)+len(term):]) for line in output.splitlines()]
        perplexity_over_training = dict(zip(range(1, len(perps)+1), perps))
        return model, perplexity_over_training
    else:
        return model


# TODO
def evaluate_LDA_model(model, data):
    # model_score = model.score(data_train)
    # mode_train_perplexity = model.bound_
    # if flag_test:
    # model_test_perplexity = model.perplexity(data_test)
        # try:
    #     summary = kwargs.get("summary", params)
    #     summary.update({"test_data": flag_test,
    #                     "model_num_iterations": model.n_iter_,
    #                     "model_score": model_score,
    #                     "model_perplexity_train": model.bound_,
    #                     })
    #     if flag_test:
    #         summary["model_perplexity_test"] = model_test_perplexity
    #     if wflag:
    #         results.save(dir_to_save=dir_to_save)
    #         results.plot_components(ordered=False, path_to_save=path.join(dir_to_save, "components.png"))
    #         results.plot_components(ordered=True, path_to_save=path.join(dir_to_save, "components_sorted.png"))
    #         results.plot_decomposition(path_to_save=path.join(dir_to_save, "decomposition.png"), cmap="Blues")
    #         logging.info(f"[{uid}] Saved results in {dir_to_save}")
    #     else:
    #         print(*summary, sep="\n")
    # except Exception as err:
    #     logging.error(f"[{uid}] Error occurred during the compiling and saving of results: {str(err)}")
    #     raise

    # logging.info(f"[{uid}] Starting coherence computations for the trained model using training data")
    # try:
    #     prop_top_pixels_all = [2.5, 5, 7.5, 10]   # NOTE, in percentage
    #     for prop_top_pixels in prop_top_pixels_all:
    #         num_top_pixels = int((prop_top_pixels / 100) * results.components.shape[1])
    #         if num_top_pixels < 3:
    #             prop_top_pixels_all.append(np.max(prop_top_pixels_all)+2.5)
    #             continue
    #         logging.info(f"[{uid}] Taking {num_top_pixels} top pixels...")
    #         c_umass = results._compute_coherence_umass(components=results.components,
    #                                                     data_used=data_train,
    #                                                     num_top_pixels=num_top_pixels,
    #                                                     randomize_data=False)
    #         c_npmi, c_cos = results._compute_coherence_npmi_cos(components=results.components,
    #                                                             data_used=data_train,
    #                                                             num_top_pixels=num_top_pixels,
    #                                                             randomize_data=False)
    #         with h5py.File(path.join(dir_to_save, "results.h5"), "r+") as handler:
    #             handler.create_dataset(f"coherence/umass/{prop_top_pixels}", data=c_umass)
    #             handler.create_dataset(f"coherence/cos/{prop_top_pixels}", data=c_cos)
    #             handler.create_dataset(f"coherence/npmi/{prop_top_pixels}", data=c_npmi)
    # except Exception as err:
    #     logging.error(f"[{uid}] Error occurred during coherence computation: {str(err)}")
    #     raise
    # logging.info(f"[{uid}] Finished coherence computations for the trained model")
    return


def plot_learning_evaluation(perplexity_by_iter: dict, path_to_save: str = ""):
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.plot(perplexity_by_iter.keys(), perplexity_by_iter.values())
    ax.set_xlabel("iter #", labelpad=10)
    ax.set_ylabel("perplexity", labelpad=10)

    if isinstance(path_to_save, str):
        fig.savefig(path_to_save, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig


def decompose_data(data: np.ndarray, model: LatentDirichletAllocation, **kwargs):

    _check_proximity_data(data)
    _check_LDA_model(model)

    # logging.info(f"[{uid}] Decomposing data based on existing LDA model...")
    decomposition = model.transform(data)

    return model.components_, decomposition


def save_results(model: LatentDirichletAllocation,
                 decomposition: np.ndarray,
                 dir_to_save: str,
                 **kwargs):
    

    uid = kwargs.get("uid", "no-uid")
    if uid is None:
        uid = "no-uid"
    
    results = Topic3DResult(model=model,
                            decomposition=decomposition)
    results.save(dir_to_save=dir_to_save)
    results.plot_CFMs(ordered=False, path_to_save=path.join(dir_to_save, "CFMs.png"))
    results.plot_CFMs(ordered=True, path_to_save=path.join(dir_to_save, "CFMs_ordered.png"))
    results.plot_decomposition(ordered=False,
                               path_to_save=path.join(dir_to_save, "decomposition.png"),
                               cmap="Blues")
    results.plot_decomposition(ordered=True,
                               path_to_save=path.join(dir_to_save, "decomposition_ordered.png"),
                               cmap="Blues")
    
    logging.info(f"[{uid}] Saved results in {dir_to_save}")


def write_results(model):
    return
from os import path
from glob import glob
import joblib as jl

import logging

from numbers import Number

import numpy as np
import h5py
from scipy.spatial.distance import squareform

from topic3d.datatypes import PWProxArrayStack, FOFTraceDataCoreTable
from ._logging import setup_log_nosave, kill_logger


def load_data_file(path_to_file,
                   uid: str = None,
                   logger: logging.Logger = None) -> (np.ndarray | dict):
    """
    Load a dataset from a Numpy or HDF5 file to yield an array, dictionary or ProcessedData object

    If directory is given, a tag must be provided. Else file path is sufficient
    """

    assert isinstance(path_to_file, str) and path_to_file.endswith((".npy", ".h5", ".csv"))
    assert path.isfile(path_to_file)

    if uid is None:
        uid = "no-uid"
    else:
        assert isinstance(uid, str)

    log_exists = False
    if logger is None:
        logger = setup_log_nosave(name="topic3d")
    else:
        assert isinstance(logger, logging.Logger)
        log_exists = True
    
    logger.info(f"[{uid}] Attempting to load data from {path_to_file}")

    if path_to_file.endswith(".npy"):
        data = np.load(path_to_file, allow_pickle=True)
    
    elif path_to_file.endswith(".csv"):
        data = FOFTraceDataCoreTable(path_to_file)

    elif path_to_file.endswith("h5"):
        with h5py.File(path_to_file, mode="r") as handler:
            data = {x:y[:] for x, y in handler.items()}
            data.update({x:y[:] for x, y in handler.attrs.items()})

    else:
        logger.info(f"[{uid}] Unrecognized file extension/type")
        raise ValueError
    
    if log_exists:
        kill_logger(logger)

    return data


def load_model_file(path_to_file: str):
    """
    Load a model from a joblib file
    
    Python3 version compatibility is generally important for the model to be loaded correctly.

    See `joblib.load` notes for more information.
    """

    assert path.isfile(path_to_file)

    return jl.load(path_to_file)


def _check_proximity_array(data, uid=None, logger=None):

    if uid is None:
        uid = "no-uid"
    else:
        assert isinstance(uid, str)

    log_exists = False
    if logger is None:
        logger = setup_log_nosave(name="topic3d")
    else:
        assert isinstance(logger, logging.Logger)
        log_exists = True

    assert data.ndim in [2, 3]
    if data.ndim == 2:
        try:
            squareform(data[0])
        except ValueError:
            logger.exception("Loaded data is a not a true pairwise array stack")
        data_ = np.copy(data)
    elif data.ndim == 3:
        assert data.shape[1] == data.shape[2]
        try:
            squareform(data[0])
        except ValueError:
            logger.exception("Loaded data is a not a true pairwise distance map stack")
        data_ = []
        for i in range(data.shape[0]):
            data_.append(squareform(data[i]))
        data_ = np.asarray(data_)

        if log_exists:
            kill_logger(logger)
        
    return data_


def prepare_proximity_data(data,
                           h5_kw: str = None,
                           proximity_params: dict = None,
                           dir_to_save: str = None,
                           uid: str = None,
                           logger: logging.Logger = None):

    assert isinstance(data, (np.ndarray, FOFTraceDataCoreTable, dict))
    save_flag = False
    if not (dir_to_save is None):
        assert(isinstance(dir_to_save, str))
        save_flag = True

    if uid is None:
        uid = "no-uid"
    else:
        assert isinstance(uid, str)

    log_exists = False
    if logger is None:
        logger = setup_log_nosave(name="topic3d")
    else:
        assert isinstance(logger, logging.Logger)
        log_exists = True
    
    if isinstance(data, dict):
        assert not (h5_kw is None)
        data_prep = data[h5_kw]
    
    elif isinstance(data, FOFTraceDataCoreTable):
        logger.info(f"[{uid}] Preparing proximity data from FOF-CT file")
        assert isinstance(proximity_params, dict)
        assert "threshold" in proximity_params.keys()
        data.treat(inplace=True)
        data_prep = data.extract_traces().compute_pwds().compute_proximity(**proximity_params)

    try:
        data_prep = _check_proximity_array(data if isinstance(data, np.ndarray) else data_prep)
    except Exception:
        logger.exception(f"[{uid}] Proximity data failed conformity checks")

    data_prep = PWProxArrayStack(data_prep[np.sum(data_prep, axis=1) > 0])  # filtering empty proximity maps

    if save_flag:
        np.save(path.join(dir_to_save, "proximity_data.npy"), data_prep)
        logger.info(f"[{uid}] Proximity data saved to results directory")

    if not log_exists:
        kill_logger(logger)

    return data_prep


# TODO update logging
def split_proximity_data(data: np.ndarray, train_data_proportion: Number = 1, **kwargs):
    """
    Assign train and test data randomly based on train data proportion
    """

    if uid is None:
        uid = "no-uid"
    else:
        assert isinstance(uid, str)

    log_exists = False
    if logger is None:
        logger = setup_log_nosave(name="topic3d")
    else:
        assert isinstance(logger, logging.Logger)
        log_exists = True

    assert isinstance(data, np.ndarray), "Data must be an array"
    assert isinstance(train_data_proportion, Number) and 0 < train_data_proportion <= 1
    if 0 < train_data_proportion <= 0.2:
        logger.warning(f"[{uid}] Train data proportion {train_data_proportion} might be too low")

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
        data_test = None
        logger.info(f"[{uid}] Training + testing data shape: {data_train.shape}")
    else:
        data_train = data[idxs_train]
        data_test = data[idxs_test]
        logger.info(f"[{uid}] Training data shape: {data_train.shape}, testing data shape: {data_test.shape}")

    if log_exists:
        kill_logger(logger)
    
    return data_train, data_test

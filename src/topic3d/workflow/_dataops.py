from os import path
from glob import glob
import joblib as jl

import logging

from numbers import Number

import numpy as np
import h5py
from scipy.spatial.distance import squareform

from topic3d.datatypes import CoordsArrayStack, PWDistArrayStack, FOFTraceDataCoreTable


def load_data_file(path_to_file) -> (np.ndarray | dict):
    """
    Load a dataset from a Numpy or HDF5 file to yield an array, dictionary or ProcessedData object

    If directory is given, a tag must be provided. Else file path is sufficient
    """

    assert isinstance(path_to_file, str) and path_to_file.endswith((".npy", ".h5", ".csv"))
    assert path.isfile(path_to_file)
    
    logging.info(f"Attempting to load data from {path_to_file}")

    if path_to_file.endswith(".npy"):
        data = np.load(path_to_file, allow_pickle=True)
    
    elif path_to_file.endswith(".csv"):
        data = FOFTraceDataCoreTable(path_to_file)

    elif path_to_file.endswith("h5"):
        with h5py.File(path_to_file, mode="r") as handler:
            data = {x:y[:] for x, y in handler.items()}
            data.update({x:y[:] for x, y in handler.attrs.items()})

    else:
        logging.info("Unrecognized file extension")
        raise ValueError

    return data


def load_model_file(path_to_file: str):
    """
    Load a model from a joblib file
    
    Python3 version compatibility is generally important for the model to be loaded correctly.

    See `joblib.load` notes for more information.
    """

    assert path.isfile(path_to_file)

    return jl.load(path_to_file)


def _check_proximity_array(data):
    assert data.ndim in [2, 3]
    if data.ndim == 2:
        try:
            squareform(data[0])
        except ValueError:
            logging.error("Loaded data is a not a true pairwise array stack")
            raise
        data_ = np.copy(data)
    elif data.ndim == 3:
        assert data.shape[1] == data.shape[2]
        try:
            squareform(data[0])
        except ValueError:
            logging.error("Loaded data is a not a true pairwise distance map stack")
            raise
        data_ = []
        for i in range(data.shape[0]):
            data_.append(squareform(data[i]))
        data_ = np.asarray(data_)
        
    return data_


def prepare_proximity_data(data, h5_kw: str = None, proximity_params: dict = None):

    assert isinstance(data, (np.ndarray, FOFTraceDataCoreTable, dict))

    if isinstance(data, dict):
        assert not (h5_kw is None)
        data_prep = data[h5_kw]
    
    elif isinstance(data, FOFTraceDataCoreTable):
        assert isinstance(proximity_params, dict)
        assert "threshold" in proximity_params.keys()
        data.treat(inplace=True)
        data_prep = data.extract_traces().compute_pwds().compute_proximity(**proximity_params)

    data_prep = _check_proximity_array(data if isinstance(data, np.ndarray) else data_prep)

    return data_prep


def split_proximity_data(data: np.ndarray, train_data_proportion: Number = 1, **kwargs):
    """
    Assign train and test data randomly based on train data proportion
    """

    uid = kwargs.get("uid", "no-uid")
    if uid is None:
        uid = "no-uid"

    assert isinstance(data, np.ndarray), "Data must be an array"
    assert isinstance(train_data_proportion, Number) and 0 < train_data_proportion <= 1
    if 0 < train_data_proportion <= 0.2:
        logging.warning(f"[{uid}] Train data proportion {train_data_proportion} might be too low")

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
        logging.info(f"[{uid}] Training + testing data shape: {data_train.shape}")
    else:
        data_train = data[idxs_train]
        data_test = data[idxs_test]
        logging.info(f"[{uid}] Training data shape: {data_train.shape}, testing data shape: {data_test.shape}")
    
    return data_train, data_test

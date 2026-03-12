from os import path
from glob import glob
import joblib as jl

import logging

from numbers import Number

import numpy as np
import h5py
from scipy.spatial.distance import squareform

from topic3d.datatypes import CoordsArrayStack, PWDistArrayStack
from temp.datamgrs import ProcessedData


# NOTE legacy function to load data
# def load_data(path_to_file, tag: str = "") -> (np.ndarray | ProcessedData | dict):
#     """
#     Load a dataset from a Numpy or HDF5 file to yield an array, dictionary or ProcessedData object

#     If directory is given, a tag must be provided. Else file path is sufficient
#     """

#     assert isinstance(path_to_file, str)
#     if path.isdir(path_to_file):
#         assert isinstance(tag, str)
#         fp = glob(path.join(path_to_file, tag+".npy")) + glob(path.join(path_to_file, tag+".h5"))
#         if len(fp) == 0:
#             logging.error(f"No Numpy or HDF5 file(s) found in `{path_to_file}` with tag `{tag}`")
#             raise FileNotFoundError
#         elif len(fp) > 1:
#             warnings.warn("Both file formats found, loading the HDF5 file")
#             fp = glob(path.join(path_to_file, tag+".h5"))[0]
#         else:
#             fp = fp[0]       
#     elif path.isfile(path_to_file):
#         fp = path_to_file
#         assert fp.endswith((".npy", ".h5"))
#     else:
#         logging.error(f"{path_to_file} is not a file and does not exist")
#         raise FileNotFoundError
    
#     if fp.endswith(".npy"):
#         return np.load(fp, allow_pickle=True)
    
#     try:
#         dataobj = ProcessedData()
#         dataobj.load(fp)
#     except AssertionError or KeyError:
#         with h5py.File(fp, mode="r") as handler:
#             dataobj = {x:y[:] for x, y in handler.items()}
#             dataobj.update({x:y[:] for x, y in handler.attrs.items()})

#     return dataobj


def load_data_file(path_to_file) -> (np.ndarray | ProcessedData | dict):
    """
    Load a dataset from a Numpy or HDF5 file to yield an array, dictionary or ProcessedData object

    If directory is given, a tag must be provided. Else file path is sufficient
    """

    try:
        assert isinstance(path_to_file, str) and path_to_file.endswith((".npy", ".h5"))
        assert path.isfile(path_to_file)
    except AssertionError:
        logging.error(f"No Numpy or compatible HDF5 file found")
        raise
    
    logging.info(f"Loading data from {path_to_file}")

    if path_to_file.endswith(".npy"):
        data = np.load(path_to_file, allow_pickle=True)
        assert data.ndim == 2
        try:
            squareform(data[0])
        except ValueError:
            logging.error("Loaded data is a not a true pairwise array stack")
            raise
        return data
    
    try:
        dataobj = ProcessedData()
        dataobj.load(path_to_file)
    except AssertionError or KeyError:
        with h5py.File(path_to_file, mode="r") as handler:
            dataobj = {x:y[:] for x, y in handler.items()}
            dataobj.update({x:y[:] for x, y in handler.attrs.items()})

    return dataobj


def load_model_file(path_to_file: str):
    """
    Load a model from a joblib file
    
    Python3 version compatibility is generally important for the model to be loaded correctly.

    See `joblib.load` notes for more information.
    """

    assert path.isfile(path_to_file)

    return jl.load(path_to_file)


# TODO update normalisation and imputed parts of code from branch cines - cines/code/perform_decomposition.py
def prepare_proximity_data(data: ProcessedData | dict | np.ndarray,
                           params: dict,
                           dir_to_save: str = "",
                           **kwargs) -> np.ndarray:
    """
    Process the PWD data supplied to filter by efficiency of barcode detection and generate proximity vectors.

    Then pick data proportion
    """

    # get PWDs whether ProcessedData or dict is supplied
    def _get_val(X, key, flag):
        return X[key][:] if flag else getattr(X, key)

    assert isinstance(dir_to_save, str)
    assert isinstance(params, dict)
    assert isinstance(data, (ProcessedData, dict, np.ndarray))
    
    uid = kwargs.get("uid", "no-uid")
    if uid is None:
        uid = "no-uid"

    if isinstance(data, (ProcessedData, dict)):
        imputed = True if params["imputed"] else False
        filtered = True if isinstance(params.get("min_efficiency", None), Number) else False
        normalized = True if isinstance(params.get("pwd_normalization", None), str) else False
        is_dict = True if isinstance(data, dict) else False
        logging.info(f"[{uid}] Taking the imputed trace file to filter by efficiency" \
                     if imputed else "Taking unimputed dataset to filter by efficiency")
        # Normalized data must already exist. TODO normalization using only PWDs
        if normalized:
            norm_method_mapping = {"MMSF": "distance distribution median-median scaling factor",
                                   "R(s)": "Euclidean vs genomic distance based adjustment"}
            pwds = _get_val(data, f"pwds_imputed_norm" if imputed else "pwds_norm", is_dict)
            if isinstance(pwds, dict):
                pwds = pwds[params["pwd_normalization"]]
            logging.info(f"[{uid}] Dataset normalized by {norm_method_mapping[params['pwd_normalization']]} taken")
        else:
            pwds = _get_val(data, "pwds_imputed" if imputed else "pwds", is_dict)
            logging.info(f"[{uid}] No normalization, raw dataset taken")
        assert isinstance(pwds, np.ndarray)
        logging.info(f"[{uid}] Original PWDs shape: {pwds.shape}")
        # Filtering step needs traces
        if filtered:
            try:
                traces = _get_val(data, "traces_imputed" if imputed else "traces", is_dict)
            except KeyError or AttributeError:
                logging.error(f"[{uid}] Traces array not found in supplied data object to filter by minimum efficiency")
                raise
            assert isinstance(traces, np.ndarray)
            if not isinstance(traces, CoordsArrayStack):
                traces = CoordsArrayStack(traces)
            idxs_filt = traces.filter(min_efficiency=params["min_efficiency"], by="item", return_idxs_only=True)
            del traces
        pwds_final = PWDistArrayStack(np.squeeze(pwds[idxs_filt]) if filtered else pwds.copy())
        logging.info(f"[{uid}] Final PWDs shape: {pwds_final.shape}")

        del pwds
    
    else:
        pwds_final = PWDistArrayStack(data)

    proximity = pwds_final.compute_proximity(threshold=params["proximity_threshold"], hardness=params["proximity_hardness"])
    proximity_avg = np.nanmean(proximity, axis=0)
    logging.info(f"[{uid}] Computed proximity stack with threshold {params['proximity_threshold']} and hardness {params['proximity_hardness']}")
   
    if len(dir_to_save) > 0 and path.isdir(dir_to_save):
        logging.info(f"[{uid}] Plotting mean proximity matrix, type/shape: {type(proximity_avg)}/{proximity_avg.shape}")
        pwds_final.plot_proximity_avg(threshold=params["proximity_threshold"],
                                      hardness=params["proximity_hardness"],
                                      keep_NaNs=False,
                                      path_to_save=path.join(dir_to_save, "proximity_avg.svg"))
        logging.info(f"[{uid}] Plotting randomly-picked examples of proximity matrices")
        pwds_final.plot_proximity_random_examples(threshold=params["proximity_threshold"],
                                                  hardness=params["proximity_hardness"],
                                                  num_items=16,
                                                  path_to_save=path.join(dir_to_save, "proximity_random_examples.svg"))
    del pwds_final

    return proximity


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

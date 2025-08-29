from os import path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
import topic3d
import h5py
from time import perf_counter

dir_root = "/mnt/grey/DATA/users/gautham/Workspace/topics"

# Load components captured from single training
with h5py.File(path.join(dir_root,
                         "training", 
                         "bintu2018Superresolution_Hs-HCT116-40-83_wt",
                         "20-components_threshold=150.0_softness=0.0_maxiters=4000_batch_data_prop_1_eeda04",
                         "results.h5")) as fp:
    components = fp["components"][:3]
print("Loaded components with shape", components.shape)

# Load the data used for training
with h5py.File(path.join(dir_root,
                         "data",
                         "processed",
                         "bintu2018Superresolution_Hs-HCT116-40-83_wt.h5")) as fp:
    data = fp["pwds"][:]
    data = topic3d.datatypes.PairwiseArrayStack(data).proximity(threshold=150.0, hardness=1)
print("Loaded used data with shape", data.shape)

# Cleaned up code
st = perf_counter()
coherence_new = topic3d.handlers.DecompositionResult._compute_coherence_npmi_cos(components=components,
                                                                                 data_used=data,
                                                                                 num_top_pixels=10,
                                                                                 randomize_data=False)
et = perf_counter()
print("Time taken: ", et - st)
print(coherence_new)


from os import path
import numpy as np
import h5py
import topic3d.visualization as t3viz
from topic3d.datatypes import PairwiseArrayStack

dir_root = "/mnt/grey/DATA/users/gautham/Workspace/topics"

# Load some data
with h5py.File(path.join(dir_root,
                         "data",
                         "processed",
                         "bintu2018Superresolution_Hs-HCT116-40-83_wt.h5")) as fp:
    data = PairwiseArrayStack(fp["pwds"][:])
print("Loaded used data with shape", data.shape)

thresholds = np.arange(50, 301, 50)

# data.plot_mean_proximity_proportion("0.png", thresholds, hardness=0.5)

prox_by_threshold = []
prox_avg_by_threshold = []
for thr in thresholds:
    prox = PairwiseArrayStack(data.proximity(threshold=thr, hardness=1))
    prox_by_threshold.append(prox[:])
    prox_avg_by_threshold.append(prox.ensemble())

t3viz.plot_2d_square_grid(prox_avg_by_threshold, [f"$k = ${thr} nm" for thr in thresholds],
                          "1.png",
                          common_colorbar=False, cmap="RdBu_r", label="proximity")
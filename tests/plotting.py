from os import path
import numpy as np
import h5py
import topic3d.visualization as t3viz
from topic3d.datatypes import CoordsArrayStack, PairwiseArrayStack

dir_root = "/mnt/grey/DATA/users/gautham/Workspace/topics"

# Load some data
with h5py.File(path.join(dir_root,
                         "data",
                         "processed",
                         "bintu2018Superresolution_Hs-HCT116-40-83_wt.h5")) as fp:
    data_tr = CoordsArrayStack(fp["traces"][:])
    data_pwd = PairwiseArrayStack(fp["pwds"][:])
print("Loaded traces with shape", data_tr.shape)
print("Loaded pairwise distances with shape", data_pwd.shape)

# effs = data_tr.efficiency(by="barcode")
# t3viz.plot_bar({"barcode_id": np.arange(len(effs)), "efficiency": effs}, path_to_save="out/2.png")

thresholds = np.arange(50, 301, 50)

# data.plot_mean_proximity_proportion("0.png", thresholds, hardness=0.5)

prox_by_threshold = []
prox_avg_by_threshold = []
for thr in thresholds:
    prox = PairwiseArrayStack(data_pwd.proximity(threshold=thr, hardness=1))
    prox_by_threshold.append(prox[:])
    prox_avg_by_threshold.append(prox.ensemble())

t3viz.plot_matrix_triangle(matrix=prox_avg_by_threshold[2],
                       path_to_save="out/3.svg",
                       cmap="Blues", cbar_label="proximity")

# t3viz.plot_grid_2d_triangles(prox_avg_by_threshold, "out/3.png",
#                              [f"$k = ${thr} nm" for thr in thresholds],
#                              common_cbar=False, cmap="magma_r", cbar_label="proximity")
from os import path
import numpy as np
import h5py
import topic3d.visualization as t3viz
from topic3d.datatypes import CoordsArrayStack, PWDistArrayStack

dir_root = "/mnt/grey/DATA/users/gautham/Workspace/topics"
# dir_root = "/home/ggautham07/Nextcloud/Datasets"

# Load some data
with h5py.File(path.join(dir_root,
                         "data",
                         "processed",
                         "gg2025_sim0_loopex_1-loop.h5")) as fp:
# with h5py.File(path.join(dir_root,
#                          "bintu2018Superresolution_Hs-HCT116-40-83_wt.h5")) as fp:
    data_tr = CoordsArrayStack(fp["traces"][:])
    data_pwd = PWDistArrayStack(fp["pwds"][:])
print("Loaded traces with shape", data_tr.shape)
print("Loaded pairwise distances with shape", data_pwd.shape)

# effs = data_tr.efficiency(by="barcode")
# t3viz.plot_bar({"barcode_id": np.arange(len(effs)), "efficiency": effs}, path_to_save="out/2.png")
# data_tr.plot_bc_efficiency(path_to_save="out/2.png", alpha=0.95)

hardnesses = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
thresholds = [5, 10, 10, 15, 15, 20] # [50, 100, 150, 150, 150, 150, 200, 250, 300]
hardnesses = [1, 1, 0.5, 1, 0.5, 1] # [1, 1, 0.1, 0.25, 0.5, 1, 1, 1, 1]

# data_pwd.plot_pwds_avg(method="median", path_to_save="out/1_sq.png", form="square", cmap="OrRd_r")
# data_pwd.plot_pixel_efficiency(path_to_save="out/2_sq.png", form="square")

# data_pwd.plot_pwds_avg(path_to_save="out/0.png", form="triangle")

# data_pwd.plot_pwds_random_examples(num_items=9, random_seed=99,
#                                    path_to_save="out/1.png", form="triangle", include_cbar=False)

# data_pwd.plot_proximity_avg(thresholds[2], hardness=1,
#                             path_to_save="out/3.png", form="triangle")

# data_pwd.plot_proximity_random_examples(threshold=thresholds[2], hardness=1,
#                                         num_items=9, random_seed=99,
#                                         path_to_save="out/4.png", form="triangle", include_cbar=False)

data_pwd.plot_proximity_avg_comparisons(thresholds=thresholds,
                                        hardnesses=hardnesses,
                                        path_to_save="out/5.png",
                                        form="triangle")
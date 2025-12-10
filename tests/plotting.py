from os import path
from matplotlib import pyplot as plt
import numpy as np
import h5py
import topic3d.visualization as t3viz
from topic3d.datatypes import CoordsArrayStack, PWDistArrayStack
from scipy.spatial.distance import squareform
from glob import glob
import joblib as jl

# dir_root = "/mnt/grey/DATA/users/gautham/Workspace/topics"
dir_root = "/mnt/PALM_dataserv/DATA/Gautham/Topics"
# dir_root = "/home/ggautham07/Nextcloud/Datasets"

# Load some data
with h5py.File(path.join(dir_root,
                         "processed",
                         "data",
                         "chen2023Structural_Hs-CNCC_wt_0_bc-0-52.h5")) as fp:     # messina2024Chromatin_Mm-Pancreas-Pdx1_wt
# with h5py.File(path.join(dir_root,
#                          "bintu2018Superresolution_Hs-HCT116-40-83_wt.h5")) as fp:
    data_tr = CoordsArrayStack(fp["traces"][:])
    data_pwd = PWDistArrayStack(fp["pwds"][:])
print("Loaded traces with shape", data_tr.shape)
# with h5py.File("/mnt/grey/DATA/users/gautham/Workspace/doc/polymer_simulations/chrom_pdx1_mouse/trajectories/locus/02_loop_extrusion_manual-optim/pwd_matrices.h5") as fp:
#     arr = fp["pairwise_distances"][:]
#     arr = np.moveaxis(arr, -1, 0)[:]
#     s = arr.shape[1]
#     data_pwd = np.empty((len(arr), (s * (s-1)) // 2))
#     for i in range(len(arr)):
#         data_pwd[i] = squareform(arr[i])
#     data_pwd = PWDistArrayStack(data_pwd)
# data_pwd = PWDistArrayStack(np.load("/mnt/grey/DATA/users/JB/Topic_analysis/2025_FOF_format/data/pwd/Hung_Pitx1_mm1_chr13_750kb_WT_high_bc_eff.npy"))
print("Loaded pairwise distances with shape", data_pwd.shape)

# effs = data_tr.efficiency(by="barcode")
# fig = t3viz.plot_bar(data={"barcode_id": np.arange(len(effs)), "efficiency": effs},
#                      x="barcode_id",
#                      y="efficiency",)
# fig.savefig("out/chen_bc_effs_by_bc.png")
# plt.close(fig)
data_tr.plot_bc_efficiency(path_to_save="out/chen_bc_effs_by_bc.png", alpha=0.95)
data_pwd.plot_pixel_efficiency(path_to_save="out/chen_px_effs.png")

# hardnesses = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
# thresholds = [5, 10, 10, 15, 15, 20] # [50, 100, 150, 150, 150, 150, 200, 250, 300]
# hardnesses = [1, 1, 0.5, 1, 0.5, 1] # [1, 1, 0.1, 0.25, 0.5, 1, 1, 1, 1]

# data_pwd.plot_pwds_avg(method="median", path_to_save="out/1_sq.png", form="square", cmap="OrRd_r")
# data_pwd.plot_pixel_efficiency(path_to_save="out/2_sq.png", form="square")

# _, data_pwd = data_pwd.filter(min_efficiency=0.5, return_idxs_only=False)
# print(data_pwd.num_items)

data_pwd.plot_pwds_avg(path_to_save="out/chen_pwds.png", form="triangle")

data_pwd.plot_pwds_random_examples(num_items=36, random_seed=42,
                                   path_to_save="out/chen_pwds_examples.png", form="triangle", include_cbar=False)

data_pwd.plot_proximity_avg(threshold=150, hardness=1,
                            path_to_save="out/chen_prox.png", form="triangle")

data_pwd.plot_proximity_random_examples(threshold=150, hardness=1,
                                        num_items=36, random_seed=42,
                                        path_to_save="out/chen_prox_examples.png", form="triangle", include_cbar=False)


# fp = "/mnt/grey/DATA/users/JB/Topic_analysis/2025_FOF_format/data/barcode_efficiency/Pdx1LR_ND_mouse_pancreas_exo_endo.npy"
# bc_effs = np.load(fp)
# print(len(bc_effs))

# fig = t3viz.plot_bar({"barcode_id": np.arange(len(bc_effs)), "efficiency": bc_effs},
#                      x="barcode_id", y="efficiency",
#                      xlabel="Barcode #", ylabel="Efficiency",
#                      alpha=0.95)
# fig.savefig("out/m_0.png")

# data_pwd.plot_proximity_avg_comparisons(thresholds=thresholds,
#                                         hardnesses=hardnesses,
#                                         path_to_save="out/5.png",
#                                         form="triangle")

# from sklearn.decomposition import LatentDirichletAllocation
# from time import perf_counter

# tag = "heteropolymer_02_b70"
# dir_root = "/mnt/grey/DATA/users/gautham/Workspace/doc/data_analysis/topic_decomposition"
# trains = glob(path.join(dir_root, f"training/{tag}/*"))
# data = PWDistArrayStack(np.load(path.join(dir_root, "data", "pwd", f"{tag}.npy")))
# # data_prox = data.proximity(threshold=14.0, hardness=1)
# for tp in trains:
#     # st = perf_counter()
#     print(tag, tp.split("_")[-1])
#     # c = np.load(path.join(tp, "components_sorted.npy"))
#     # flag = False
#     # try:
#     #     d = np.load(path.join(tp, "decomposition_train_sorted.npy"))
#     # except FileNotFoundError or OSError:
#     #     print("Failed to get decomposition")
#     #     flag = True
#     #     continue
#     # # c = c / c.sum(axis=1)[:, np.newaxis]
#     # if not flag:
#     #     r = np.mean(d >= 0.33, axis=0)
#     #     fig = t3viz.plot_matrices_triangle(items=c,
#     #                                        titles=[f"Topic {i}  |  $\\rho =$ {r[i] * 100:.2f}%" for i in range(len(c))],
#     #                                        cbar_label="pseudocount")
#     # else:
#     #     fig = t3viz.plot_matrices_triangle(items=c,
#     #                                        titles=[f"Topic {i}" for i in range(len(c))],
#     #                                        cbar_label="pseudocount")

#     data.plot_proximity_avg(threshold=12, hardness=1,
#                             path_to_save=path.join(tp, "proximity_mean.svg"),
#                             form="triangle")

#     data.plot_proximity_random_examples(threshold=12, hardness=1,
#                                         num_items=16, random_seed=42,
#                                         path_to_save=path.join(tp, "proximity_examples.svg"),
#                                         form="triangle", include_cbar=False)

    # model = jl.load(path.join(tp, "model.jl"))
    # c = model.components_[:]
    # d = model.transform(data_prox)
    # np.save(path.join(tp, "components.npy"), c)
    # np.save(path.join(tp, "decomposition.npy"), d)
    # r = np.mean(d >= 0.33, axis=0)
    # idxs = np.argsort(r)[::-1]
    # c = c[idxs]
    # r = r[idxs]
    # np.save(path.join(tp, "components_sorted.npy"), c)
    # fig = t3viz.plot_matrices_triangle(items=c,
    #                                    titles=[f"Topic {i}  |  $\\rho =$ {r[i] * 100:.2f}%" for i in range(len(c))],
    #                                    cbar_label="pseudocount")
    # fig.savefig(path.join(tp, "components.svg"))
    # fig.savefig(path.join(tp, "components.png"), dpi=300)
    # plt.close(fig)
    # print(f"Time taken: {perf_counter() - st:.2f}s")
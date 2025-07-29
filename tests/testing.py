from os import path
import numpy as np
import pandas as pd
from classes import RawData, ProcessedData, PairwiseArrayStack, CoordsArrayStack
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

# path_to_file = "../../topics/data/raw/messina20233DChromatin_Dm-Embryo_wt_0_4DNFI1T9J25Q.csv"
# raw_data = RawData()
# raw_data.load(path_to_file=path_to_file)
# print(raw_data.fofdata.head())

# processed_data = raw_data.process()

processed_data = ProcessedData()
processed_data.load(path_to_file="../../topics/data/processed/messina2024Chromatin_Mm-Pancreas-Pdx1_wt.h5")
print(type(processed_data.pwds))

processed_data.pwds.plot_pwds_avg("./res/mean-pwd-avg-matrix.png", method="median", form="triangle")
processed_data.pwds.plot_proximity_avg("./res/mean-proximity-avg-matrix.png",
                                       threshold=100, hardness=0.3,
                                       form="square")

# processed_data.save("../../topics/data/processed/chen2023Structural_wt-CNCC_0.h5", replace=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jan 12th 2023
@author: jb

------------------------------------------------------------------------------------------------------------------------
This code is used to compile multiple topic trainings and test the reproducibility of the results. In a first time, the
script will output the topics found with the highest probability, and then all the ones that have been discarded.
------------------------------------------------------------------------------------------------------------------------
"""
from glob import glob
from matplotlib import use
from os import path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import squareform
use('TkAgg')






    # def plot_triangular_matrix(topic, ax=None, cmap="coolwarm"):
    #     """
    #     Plot a triangular matrix using imshow, and clip to a lower triangle.

    #     @param topic: (np.array) square matrix
    #     @param ax: matplotlib axis
    #     @param cmap: colormap string
    #     @return: imshow object
    #     """
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     import matplotlib.patches as patches
    #     from matplotlib import transforms

    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(10, 10))

    #     # create a mask, selecting only the lower part of the topic array (using np.triu) and then apply it to the topic
    #     # array, selecting only the lower part of the array (np.ma.masked_array)
    #     n_bins = topic.shape[0]
    #     mask = np.triu(np.ones_like(topic, dtype=bool))
    #     topic_masked = np.ma.masked_array(topic, mask=~mask)

    #     # plot the masked array
    #     im = ax.imshow(topic_masked, origin='lower', cmap=cmap, interpolation='none')

    #     # rotate the plot by -45° to display the masked topic as a triangle with the diagonal as basis
    #     transform = transforms.Affine2D().rotate_deg(-45).scale(1 / np.sqrt(2), -1 / np.sqrt(2)) + ax.transData
    #     im.set_transform(transform)

    #     # Clip to triangle shape - in the new set of axis, define the coordinates of the three extremities of a triangle.
    #     # Note that we need to start at [-0.5, 0] instead of [0, 0] to avoid having a small portion of the triangle being
    #     # clipped
    #     coord = np.array([[-0.5, 0], [n_bins, 0], [n_bins / 2, n_bins / np.sqrt(2)]])
    #     triangle = patches.Polygon(coord, closed=True, transform=ax.transData)
    #     im.set_clip_path(triangle)

    #     # Adjust axes
    #     ax.set_xlim(-1, n_bins)
    #     ax.set_ylim(0, n_bins )
    #     ax.set_aspect('equal')
    #     ax.axis('off')

    #     return im

    

# parameters
# ----------
root_data_folder = "/mnt/grey/DATA/users/JB/Topic_analysis/2025_FOF_format/training/ins2HR_ND_mouse_all_tissues_except_pancreas/barcode_efficiency_55"
# root_data_folder = "/mnt/grey/DATA/users/JB/Topic_analysis/2025_FOF_format/training/Pdx1LR_Fetal_Pancreas_Human_Filtered/2025_04_01/raw_data"
# root_data_folder = "/mnt/grey/DATA/users/JB/Topic_analysis/2025_FOF_format/training/Pdx1LR_ND_mouse_all_tissues_except_pancreas"
files = glob(path.join(root_data_folder, '**', "components*.npy"))

n_trainings = len(files)
min_correlation = 0.55
min_frequency = 0.4

# load all the topics and concatenate them in a single file
# ---------------------------------------------------------
for n_file, file in enumerate(files):
    topics = np.load(file)
    if n_file == 0:
        all_topics = topics
    else:
        all_topics = np.concatenate((all_topics, topics))

# analyze each topic and select the ones that are similar, based on Pearson correlation score. If the number of selected
# topics is high enough,
# -------------------------------------------------------------------------------------------
mean_topics = []
mean_topics_frequency = []
mean_correlation = []
residual_topics = []
residual_topics_frequency = []
initial_n_topics = all_topics.shape[0]

while all_topics.shape[0] > 0:

    n_total_topics = all_topics.shape[0]
    selected_topics = []
    pearson = []

    for n in tqdm(range(n_total_topics)):
        pearson_correlation = np.corrcoef(all_topics[0, :], all_topics[n, :])
        if pearson_correlation[0, 1] > min_correlation:
            selected_topics.append(n)
            pearson.append(pearson_correlation[0, 1])

    f = len(selected_topics) / n_trainings
    if f > min_frequency:
        mean_topics.append(np.mean(all_topics[selected_topics], axis=0))
        mean_topics_frequency.append(len(selected_topics))
        mean_correlation.append(np.around(np.mean(pearson), decimals=2))
        all_topics = np.delete(all_topics, selected_topics, 0)
    else:
        residual_topics.append(all_topics[0, :])
        residual_topics_frequency.append(len(selected_topics))
        all_topics = np.delete(all_topics, 0, 0)

p_topics_used = np.around(100 * (initial_n_topics - len(residual_topics)) / initial_n_topics, decimals=1)
print(f"{len(mean_topics)} consensus topics were computed, using {p_topics_used}% of the total number of topics")

# plot the best topics in a single figure
# ---------------------------------------
idx_corr = np.argsort(np.array(mean_correlation))[::-1]
mean_topics_frequency = np.round(np.array(mean_topics_frequency) / n_trainings * 100)

fontdic = {'fontsize': 7,
           'fontweight': 'normal',
           'verticalalignment': 'baseline',
           'horizontalalignment': 'center'}
# max_idx = np.argsort(np.max(model.components_, axis=1))

n_component = len(mean_topics)

plot_array = int(np.ceil(np.sqrt(n_component)))
fig, axs = plt.subplots(plot_array, plot_array, figsize=(10, 10))
axs = axs.flatten()
for n_plot, idx in enumerate(idx_corr):
    topic = squareform(mean_topics[idx])
    sns.heatmap(topic, cmap='coolwarm', ax=axs[n_plot], cbar=False)
    axs[n_plot].get_xaxis().set_visible(False)
    axs[n_plot].get_yaxis().set_visible(True)
    axs[n_plot].set_aspect('equal')
    axs[n_plot].set_title(f'corr={mean_correlation[idx]:.2f} - f={mean_topics_frequency[idx]:.2f}',
                          fontdict={'fontsize': 8})
    axs[n_plot].tick_params(axis='both', labelsize=8)

# Turn off any unused subplots.
for ax in axs[len(idx_corr):]:
    ax.axis('off')

figure_name = path.join(root_data_folder,
                        f"consensus_min_correlation_{min_correlation}_min_frequency_{min_frequency}.png")
plt.savefig(figure_name, dpi=250, bbox_inches='tight')

# save the consensus topics as npy file
file_name = path.join(root_data_folder,
                        f"consensus_min_correlation_{min_correlation}_min_frequency_{min_frequency}.npy")
mean_topics = np.array(mean_topics)
mean_topics = mean_topics[idx_corr, :]
np.save(file_name, mean_topics)

# plot_topics(len(mean_topics), number_of_rt, np.array(mean_topics), colormap="coolwarm")

# plot the residual topics in a single figure
# ------------------------------------------
# plot_topics(len(residual_topics), number_of_rt, np.array(residual_topics), colormap="coolwarm")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

"""
This script is used to compute the coherence for a series of training and plot the coherence vs. number of components
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from os import path
from pathlib import Path
from tqdm import tqdm
from matplotlib import use
from scipy.spatial.distance import squareform
use('Agg')


def sort_topics_by_representativity(n_topics, LDA_decomposition, min_weight=0.2):
    """ Using the topic decomposition, sort the topic by their representativity.

    @param min_weight: (float) minimum weight threshold above which the topic is considered relevant
    @param n_topics: (int) number of topics output by the LDA analysis
    @param LDA_decomposition: (np.array) topic decomposition for each cell - indicates the weight associated to each
    topic for the decomposition
    @return: topic_frequency (np.array) : frequency of finding each topic within the data (after sorting)
             topic_idx (np.array) : indices used to sort the topics by representativity
    """
    topic_representation = [len(LDA_decomposition[LDA_decomposition[:, n] > min_weight, n])
                            for n in range(n_topics)]
    topic_representation = np.array(topic_representation)
    topic_idx = np.argsort(topic_representation)[::-1]
    topic_frequency = topic_representation / LDA_decomposition.shape[0]

    return topic_frequency, topic_idx


def plot_topics(topics, n_topics, representativity, colormap="coolwarm", saving_filename=None):
    """ Based on the topics computed by LDA, reformat them as 2D maps and display all of them as an array of subplots

    @param saving_filename: (str) indicate the complete path where to save the plot. If None, the plot is simply
    displayed.
    @param colormap: (str) indicate the colormap to be used for display
    @param topics: (numpy array) 2D array containing all the topics output by the LDA
    @param topic_sublist: (list) if not empty, indicate the list of topics to display
    """
    plot_array = np.ceil(np.sqrt(n_topics)).astype('int')
    fig, axs = plt.subplots(plot_array, plot_array, figsize=(15, 10))

    if n_topics == 1:
        axs = [axs]

    for n_plot in tqdm(range(n_topics), desc="plotting topics..."):
        if n_topics > 1:
            idx = np.unravel_index(n_plot, (plot_array, plot_array))
        else:
            idx = 0

        if topics.ndim == 2:
            topic = squareform(topics[n_plot, :])
        else:
            topic = topics[n_plot, :]
        topic = np.nan_to_num(topic, nan=0)

        sns.heatmap(topic, cmap=colormap, ax=axs[idx])
        axs[idx].get_xaxis().set_visible(False)
        axs[idx].get_yaxis().set_visible(True)
        axs[idx].set_aspect('equal')
        axs[idx].set_title(f'{np.around(representativity[n_plot] * 100, decimals=1)}%')

    while n_plot < plot_array * plot_array - 1:
        n_plot += 1
        idx = np.unravel_index(n_plot, (plot_array, plot_array))
        fig.delaxes(axs[idx[0]][idx[1]])

    if saving_filename is not None:
        plt.savefig(saving_filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def umass_coherence(n_topics, topics, raw_data, n_top_words=10, random=False):
    """ Compute the umass coherence for a selected set of topics and raw data. The calculation is performed by selecting
     the top words of each topic. A default value of 10 words (or bins) is indicated.

    @param random:
    @param n_top_words: (int) number of top-words (bins) to select for computing the coherence
    @param n_topics: (int) number of topics used for LDA decomposition
    @param topics: (numpy array) list all the topics processed during LDA decomposition
    @param raw_data: (numpy array) contain the contact vectors for each single cells (previously selected for LDA)
    @param random: (boolean) indicate whether the data should be randomized (in order to compute a baseline)
    @return coherence_umass: (numpy array) contain the coherence values computed for each topic
    """
    coherence_umass = np.zeros((n_topics,))
    data_frequency = np.sum(raw_data, axis=0)
    n_data = raw_data.shape[0]

    for n_topic in range(n_topics):
        topic = topics[n_topic, :]
        if random is False:
            idx = np.argsort(topic, )[::-1]  # return the bin index in descending order
        else:
            idx = np.random.permutation(topic.shape[0])

        c_umass = 0
        for m in range(n_top_words):
            for n in range(0, m - 1):
                d_mn = raw_data[(raw_data[:, idx[m]] == 1) & (raw_data[:, idx[n]] == 1)].shape[0]
                d_nn = data_frequency[idx[n]]
                c_umass += np.log2((d_mn + 10 ** -6) / d_nn)

        coherence_umass[n_topic] = 2 / (n_top_words * (n_top_words - 1)) * c_umass

    return coherence_umass


def npmi_coherence(n_topics, topics, raw_data, n_top_words=10, random=False):
    """ Compute coherence for a selected set of topics and raw data using two methods. The calculation is based on the
    normalized pointwise mutual information and is performed by selecting the top words of each topic. A default value
    of 10 words (or bins) is indicated.

    @param n_top_words: (int) number of top-words (bins) to select for computing the coherence
    @param n_topics: (int) number of topics used for LDA decomposition
    @param topics: (numpy array) list all the topics processed during LDA decomposition
    @param raw_data: (numpy array) contain the contact vectors for each single cells (previously selected for LDA)
    @param random: (boolean) indicate whether the data should be randomized (in order to compute a baseline)
    @return coherence_npmi: (numpy array) contain the npmi coherence values computed for each topic
            coherence_cos: (numpy array) contain the coherence values computed for each topic based on cosine similarity
    """
    coherence_npmi = np.zeros((n_topics,))
    coherence_cos = np.zeros((n_topics,))
    n_data = raw_data.shape[0]
    data_frequency = np.sum(raw_data, axis=0)

    for n_topic in range(n_topics):
        topic = topics[n_topic, :]
        if random is False:
            idx = np.argsort(topic, )[::-1]  # return the bin index in descending order
        else:
            idx = np.random.permutation(topic.shape[0])
        c_npmi = 0
        c_cos = 0

        # compute the NPMI for all pairs of selected bins
        npmi = np.zeros((n_top_words, n_top_words))
        for m in range(n_top_words):
            for n in range(n_top_words):
                d_mn = raw_data[(raw_data[:, idx[m]] == 1) & (raw_data[:, idx[n]] == 1)].shape[0]
                p_mn = d_mn / n_data
                p_nn = data_frequency[idx[n]] / n_data
                p_mm = data_frequency[idx[m]] / n_data
                pmi = np.log2((p_mn + 10 ** -6) / (p_nn * p_mm))
                npmi[m, n] = - pmi / np.log2(p_mn + 10**-6)

        # for each selected bins (top words), compute the norm of the NPMI computed for all the bins (one top bin versus
        # all). This is required for the calculation of the cosine similarity.
        norm = np.zeros((n_top_words,))
        for n in range(n_top_words):
            norm[n] = np.sqrt(np.sum(np.square(npmi[n, :])))

        # based on the previous calculation, compute the NPMI and COS coherence
        for m in range(0, n_top_words - 1, 1):
            for n in range(m + 1, n_top_words):
                c_npmi += npmi[m, n]
                c_cos += np.sum(np.multiply(npmi[m, :], npmi[n, :])) / (norm[m] * norm[n])

        coherence_cos[n_topic] = 2 / (n_top_words * (n_top_words - 1)) * c_cos
        coherence_npmi[n_topic] = 2 / (n_top_words * (n_top_words - 1)) * c_npmi

    return coherence_cos, coherence_npmi


if __name__ == "__main__":

    root = "/mnt/grey/DATA/users/JB/Topic_analysis/2025_FOF_format"
    tag = "Bintu_HCT116_chr21_WT_40_83"
    training_folder = "2025_05_09"
    contact_cutoff = 150  # in nm, the expected contact threshold we are aiming for

    # look for all the folder containing a components file
    starting_folder = Path(path.join(root, "training", tag, training_folder))
    keyword_folder = starting_folder.rglob("*-components_*")
    selected_folders = []
    for folder in keyword_folder:
        if any(folder.glob('components.npy')):
            selected_folders.append(folder)

    # load the raw data (pwd & efficiency) and compute the contacts
    pwd = np.load(path.join(root, "data", "pwd", f"{tag}.npy"))
    efficiency = np.load(path.join(root, "data", "barcode_efficiency", f"{tag}.npy"))
    contacts = np.asarray(pwd <= contact_cutoff, dtype=int)

    # iterate over all the folder and compute the coherence
    coherence = np.zeros((len(selected_folders), 4))
    for n_folder, folder in tqdm(enumerate(selected_folders)):

        # load the data and parameters
        components = np.load(path.join(folder, "components.npy"))
        decomposition = np.load(path.join(folder, "decomposition_train.npy"))
        n_components = components.shape[0]
        with open(path.join(folder, "summary.json")) as f:
            parameters = json.load(f)
        contact_threshold = parameters["contact_threshold"]
        min_barcode_efficiency = parameters["min_barcode_efficiency"]

        # select the contact data with the same minimum efficiency used for the training
        if contact_threshold != contact_cutoff:
            continue

        idx = np.argwhere(efficiency >= min_barcode_efficiency)
        contacts = contacts[idx.flatten(), :]

        # compute the representativity for each topic
        representativity, sorted_components_idx = sort_topics_by_representativity(n_components, decomposition,
                                                                                  min_weight=0.1)
        components = components[sorted_components_idx, :]
        representativity = representativity[sorted_components_idx]
        plot_name = path.join(folder, "sorted_CFMs.png")
        plot_topics(components, len(components), representativity, saving_filename=plot_name)

        # compute the coherence and save the average values
        c_umass = umass_coherence(n_components, components, contacts, n_top_words=10, random=False)
        c_cos, c_npmi = npmi_coherence(n_components, components, contacts, n_top_words=10, random=False)
        coherence[n_folder, :] = [n_components, np.average(c_umass, weights=representativity),
                                  np.average(c_cos, weights=representativity),
                                  np.average(c_npmi, weights=representativity)]

    # convert results into a dataframe
    df = pd.DataFrame(coherence, columns=["# of CFMs", "umass", "npmi", "cos"])
    coherence_df = df.melt(id_vars='# of CFMs', var_name='method', value_name='coherence')

    # plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    g = sns.catplot(
        data=coherence_df, x="# of CFMs", y="coherence", col="method",
        capsize=.2, palette="YlGnBu_d", errorbar="se",
        kind="point", height=6, aspect=.75,
    )
    g.despine(left=True)

    plt.savefig("/mnt/grey/DATA/users/JB/Topic_analysis/2025_FOF_format/training/Bintu_HCT116_chr21_WT_40_83/TEST.png")



    g = sns.catplot(
        data=results_df, x="time", y="pulse", hue="kind", col="diet",
        capsize=.2, palette="YlGnBu_d", errorbar="se",
        kind="point", height=6, aspect=.75,
    )
    g.despine(left=True)
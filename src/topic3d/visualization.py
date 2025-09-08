#!/usr/bin/env python
"""
Plotting methods for tracing data and topic decomposition (:mod:`topic3d.visualization`)
=====================================================

.. authors:: jbfiche, marcnol, ggautham07

Function reference
------------------
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.stats import fisher_exact
import math

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Patch, Polygon
import matplotlib.patheffects as patheffects
from matplotlib import rcParams
import seaborn as sns
from PIL import Image

from os import path
from typing import Dict, List, Tuple, Iterable
from collections import Counter
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import helper as hp


# Setting global matplotlib parameters
# TODO change all properties that have a global same value (or most of the time) to avoid rewriting properties
rcParams["savefig.dpi"] = 300
rcParams["figure.dpi"] = 300


def _get_grid_num_rows_cols(num_items):
    """Get the best number of rows and columns for grid plots"""

    sqrt = np.sqrt(num_items)
    if sqrt == int(sqrt):
        nrows = int(sqrt)
        ncols = int(sqrt)
    else:
        nrows = int(np.ceil(sqrt))
        ncols = int(np.floor(sqrt))
        if nrows * ncols < num_items:
            ncols += 1
    return nrows, ncols


# TODO add possibility of passing x, y pairs as parameters so that multiple lines can be plotted on the same axis
def plot_line(x, y, xlabel="", ylabel="", **kwargs) -> Figure:
    """
    Create a figure and axis object, and plot the data in a manuscript-ready graphic and format

    Parameters
    --------
    x : Iterable
        points along the x-axis
    y : Iterable
        points along the y-axis
    """

    properties = kwargs.copy()

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(9)
    ax.tick_params(axis="x", labelsize=30, which="major", pad=16, direction="in", length=8, width=1.1)
    ax.tick_params(axis="x", labelsize=22, which="minor", pad=16, direction="in", length=4, width=1)
    ax.tick_params(axis="y", labelsize=30, which="major", pad=16, direction="in", length=8, width=1.1)
    ax.tick_params(axis="y", labelsize=22, which="minor", pad=16, direction="in", length=4, width=1)
    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.set_xlabel(xlabel, fontsize=36, fontweight=400, labelpad=28)
    ax.set_ylabel(ylabel, fontsize=36, fontweight=400, labelpad=30)
    sns.lineplot(x=x, y=y, ax=ax, **properties)
    plt.tight_layout()
    
    return fig


# TODO change setting from dict to x, y pairs as parameters so that multiple bar plots can be plotted on the same axis
def plot_bar(data:dict, x:str, y:str, xlabel="", ylabel="", **kwargs):
    """
    Create a figure and axis object, and plot the data in a bar plot in a manuscript-ready graphic and format
    """

    assert isinstance(data, dict)
    assert isinstance(x, str)
    assert isinstance(y, str)

    fig, ax = plt.subplots()
    fig.set_figwidth(len(data[x]) // 2.5)
    fig.set_figheight(6)
    ax.tick_params(axis="x", labelsize=30, which="major", pad=16, direction="in", length=8, width=1.1)
    ax.tick_params(axis="x", labelsize=22, which="minor", pad=16, direction="in", length=4, width=1)
    ax.tick_params(axis="y", labelsize=30, which="major", pad=16, direction="in", length=8, width=1.1)
    ax.tick_params(axis="y", labelsize=22, which="minor", pad=16, direction="in", length=4, width=1)
    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    sns.barplot(data=data,
                x=x,
                y=y,
                estimator="sum",
                errorbar=None,
                ax=ax,
                width=0.65,
                legend=False,
                **kwargs)
    ax.margins(0.03, 0.12)
    tks = list(range(0, len(data[x]), 5))
    ax.set_xticks(tks, data[x][::5])
    ax.bar_label(ax.containers[0], fmt="{:.0f}", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=36, fontweight=400, labelpad=28)
    ax.set_ylabel(ylabel, fontsize=36, fontweight=400, labelpad=30)
    
    return fig


# # TODO finish setting ticks, labels and colours in manuscript-ready format
# def plot_bar(data: dict | list[dict],
#              x: str | list[str], y: str | list[str],
#              xlabel="", ylabel="", **kwargs):
#     """
#     Create a figure and axis object, and plot the data in a bar plot in a manuscript-ready graphic and format
#     """

#     assert isinstance(data, (dict, list[dict]))
#     assert isinstance(data, (list, list[str]))
#     assert isinstance(data, (list, list[str]))

#     if isinstance(data, dict):
#         data = [data]
#     if isinstance(x, str):
#         x = [x]
#     if isinstance(y, str):
#         y = [y]
#     assert len(x) == len(y) == len(data)

#     properties = kwargs.copy()

#     fig, ax = plt.subplots()
#     for i in range(len(data)):
#         sns.barplot(data=data[i],
#                     x=[i],
#                     y=[i],
#                     estimator="sum",
#                     errorbar=None,
#                     ax=ax,
#                     legend=True if i > 0 else False,
#                     **properties)
#     ax.bar_label(ax.containers[0], fmt=":d", fontsize=8)
#     fig.set_figwidth(ax.get_xlim() // 2.5)
#     fig.set_figheight(5)
#     ax.tick_params(axis="x", labelsize=30, which="major", pad=16, direction="in", length=8, width=1.1)
#     ax.tick_params(axis="x", labelsize=22, which="minor", pad=16, direction="in", length=4, width=1)
#     ax.tick_params(axis="y", labelsize=30, which="major", pad=16, direction="in", length=8, width=1.1)
#     ax.tick_params(axis="y", labelsize=22, which="minor", pad=16, direction="in", length=4, width=1)
#     ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
#     ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
#     ax.set_xlabel(xlabel, fontsize=36, fontweight=400, labelpad=28)
#     ax.set_ylabel(ylabel, fontsize=36, fontweight=400, labelpad=30)
#     tks = list(range(0, len(data["barcode_id"]), 5))
#     ax.set_xticks(tks, data["barcode_id"][::5])

#     return fig


# ---------------------------------------------------------------------------
#  Configuration constants
# ---------------------------------------------------------------------------
BRIGHT_Q   = 0.97                   # top‑quantile threshold for hot pixels
EDGE_TYPES = [
    "CTCF-CTCF",
    "CTCF-CRE",
    "CRE-CRE",
    "other",
]
OR_DIGITS  = 2                      # decimals in overlay text


# ---------------------------------------------------------------------------
#  Helper functions for enrichment analysis
# ---------------------------------------------------------------------------

def _extract_bin_sets(locus_dict: Dict) -> Tuple[set[int], set[int]]:
    """Return two *sets* – (ctcf_bins, cre_bins) – parsed from *locus_dict*."""
    ctcf, cre = set(), set()
    # --- CTCF --------------------------------------------------------------
    for orient, bins in locus_dict.get("CTCF", {}).items():
        ctcf.update(bins if isinstance(bins, list) else [bins])
    # --- CREs --------------------------------------------------------------
    cre_raw = locus_dict.get("CREs", {})
    if isinstance(cre_raw, dict):
        for bins in cre_raw.values():
            cre.update(bins if isinstance(bins, list) else [bins])
    else:                               # user passed a single list
        cre.update(cre_raw)
    return ctcf, cre


def _bright_mask(mat: np.ndarray, q: float = BRIGHT_Q) -> np.ndarray:
    """Boolean mask of the brightest *q*‑quantile pixels."""
    return mat >= np.quantile(mat, q)


def _bright_pairs(mat: np.ndarray, q: float = BRIGHT_Q) -> List[Tuple[int, int]]:
    t = np.quantile(mat, q)
    rr, cc = np.where(mat >= t)
    return [(int(i), int(j)) for i, j in zip(rr, cc) if i < j]


def _edge_type(i: int, j: int, ctcf: set[int], cre: set[int]) -> str:
    a, b = i in ctcf, j in ctcf
    c, d = i in cre,  j in cre
    if a and b:
        return "CTCF-CTCF"
    if c and d:
        return "CRE-CRE"
    if (a and d) or (b and c):
        return "CTCF-CRE"
    return "other"


def _odds_ratio_matrix(mat: np.ndarray, locus_dict: Dict, bright_q: float = BRIGHT_Q
                      ) -> Dict[str, Tuple[float, float]]:
    """
    Return **odds ratio and p-value** for every *EDGE_TYPES* category.

    The 2×2 table is built *within the matrix*:

    ============  =================  ========================
    cell class    edge == category   edge ≠ category
    ============  =================  ========================
    bright        *a*                *b*
    non-bright    *c*                *d*
    ============  =================  ========================

    ``OR = (a/b) / (c/d)`` - computed by *scipy*'s ``fisher_exact`` which also
    returns the two-sided p-value.
    """
    ctcf_bins, cre_bins = _extract_bin_sets(locus_dict)
    # NOTE changed for Bintu data since bins are barcode-based but OR calculation is matrix index-based
    ctcf_bins = [c-39 for c in ctcf_bins]
    cre_bins = [c-39 for c in cre_bins] # TODO change this because it was kept for Bintu's data

    bright = _bright_pairs(mat, bright_q)
    bright_total = len(bright)

    # total edges per type in the *whole* matrix (upper triangle)
    n = mat.shape[0]
    type_total = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            et = _edge_type(i, j, ctcf_bins, cre_bins)
            type_total[et] += 1

    # counts restricted to bright mask
    bright_type = Counter(_edge_type(i, j, ctcf_bins, cre_bins) for i, j in bright)

    results: Dict[str, Tuple[float, float]] = {}
    for et in EDGE_TYPES:
        a = bright_type[et]
        b = bright_total - a
        c = type_total[et] - a
        d = (n * (n - 1) // 2) - bright_total - c
        OR, p = fisher_exact([[a, b], [c, d]])
        results[et] = (OR, p)
    return results


def _annotate_labels(ax, tick_dict, loc_dict, lib, fontsize=12, line_sep=0.04):
    """
    Annotate the triangular plot axis with parsed labels, including:
    - Orientation markers for CTCF
    - Gene barcodes with color coding
    - Generic loci annotations
    @param ax: current axis associated to the triangular plot
    @param tick_dict: (dict) parsed dictionnary containing all the ticks
    @param loc_dict: (dict) initial dicttionary containing all the loci to highlight
    @param fontsize: (int) indicate the fontsize to use for the labels
    @param line_sep: (float) indicate the separation between label rows
    @return:
    """
    # 1. Set tick positions and clear labels - the labels will be assigned iteratively in step 2
    bc_list = lib["barcode_id"].tolist()
    x_positions = [i for i in range(len(bc_list))]
    ax.set_xticks(x_positions)
    ax.set_xticklabels([])

    # 2. Assign tick labels and colors - this is used to indicate the positions of the genes
    for i, bc in enumerate(bc_list):
        tick = i  # barcode index
        label_color = 'black'

        # Check if barcode is associated to a gene
        tick_pos = tick  # match tick_dict convention
        if tick_pos in tick_dict:
            for entry in tick_dict[tick_pos]:
                if len(entry) == 4 and entry[2] == 'gene':
                    label_color = entry[3]  # assign gene color

        # Draw the label manually
        if label_color == 'black':
            ax.text(
                x=tick, y=-0.01, s=bc,
                transform=ax.get_xaxis_transform(),
                rotation=0,
                ha='center', va='top',
                fontsize=fontsize,
                fontweight='bold',
                color=label_color  # ✅ this sets the correct color
                )
        else:
            ax.text(
                x=tick, y=-0.01, s=bc,
                transform=ax.get_xaxis_transform(),
                rotation=0,
                ha='center', va='top',
                fontsize=fontsize,
                fontweight='bold',
                color=label_color,  # ✅ this sets the correct color
                path_effects=[patheffects.Stroke(linewidth=1.5, foreground='black'), patheffects.Normal()]
            )

        # Draw a small tick
        ax.plot([i, i], [0, -0.2], color='black', lw=1, transform=ax.transData, clip_on=False)

    # 3. Add text labels for CTCF, CREs etc.
    for tick, entries in tick_dict.items():
        for entry in sorted(entries, key=lambda x: x[0]):
            group_index = entry[0]
            offset = -0.01 - line_sep * group_index
            if len(entry) == 3 and entry[2] == "CTCF":
                _, label_text, _ = entry
                ax.text(tick, offset, label_text,
                        transform=ax.get_xaxis_transform(), ha='center', va='top',
                        fontsize=fontsize - 3, fontstyle='italic', linespacing=0.4)

            elif len(entry) == 4 and entry[2] == "CREs":
                label_text = entry[1]
                idx_min, idx_max = entry[3]
                # tick_start = idx_min - 0.5
                # tick_end = idx_max + 0.5
                # box_width = tick_end - tick_start

                # # Coordinates in unrotated data space
                # x = idx_min
                # width = idx_max - idx_min + 1
                # y = -0.8  # adjust vertically to place below axis
                # height = 0.5  # vertical height of the box

                # Draw the CRE label with a wide bbox centered on the barcode span
                ax.text(
                    x=tick, y=offset, s=label_text,
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top',
                    fontsize=fontsize,
                    bbox=dict(boxstyle='round,pad=0.2', edgecolor='black', facecolor='white', linewidth=1,
                              mutation_aspect=1)
                )

            # elif len(entry) < 3 or entry[2] != "gene":
            #     _, label_text = entry[:2]
            #     # ax.text(tick + 0.5, offset, label_text,
            #     #         transform=ax.get_xaxis_transform(), ha='center', va='top',
            #     #         fontsize=12, fontstyle='normal')
            #     ax.text(tick, offset, label_text,
            #             transform=ax.get_xaxis_transform(), ha='center', va='top',
            #             fontsize=12, fontweight='normal',
            #             bbox=dict(boxstyle='round,pad=0.2', edgecolor='black', facecolor='white', linewidth=1))

    # Add group labels on the left
    left_x = min(x_positions) - 1 if x_positions else 0
    for group_index, locus_type in enumerate(loc_dict.keys()):
        if locus_type == "genes":
            continue
        offset = -0.01 - line_sep * group_index
        ax.text(left_x, offset, locus_type,
                transform=ax.get_xaxis_transform(), ha='right', va='top',
                fontsize=fontsize, fontweight='bold')

    return ax


def _add_gene_legend(ax, loci_dict, location='outside', anchor=(1.05, 1), fontsize=9):
    """
    Adds a legend to the plot for genes with their associated colors.

    @param ax: matplotlib axis
    @param loci_dict: dictionary of loci, must include 'genes'
    @param location: 'outside' (default), 'inside', or any valid matplotlib location string
    @param anchor: tuple for bbox_to_anchor if location='outside'
    @param fontsize: label font size
    """
    from matplotlib.patches import Patch

    if "genes" not in loci_dict:
        return ax

    gene_entries = loci_dict["genes"]
    # gene_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    legend_elements = []

    for idx, (gene, (_, color)) in enumerate(gene_entries.items()):
        # color = gene_palette[idx % len(gene_palette)]
        patch = Patch(facecolor=color, edgecolor='black', label=gene)
        legend_elements.append(patch)

    if location == 'outside':
        ax.legend(handles=legend_elements, title="Genes",
                  bbox_to_anchor=anchor, loc='upper left', fontsize=fontsize, title_fontsize=fontsize+1)
    else:
        ax.legend(handles=legend_elements, title="Genes",
                  loc=location, fontsize=fontsize, title_fontsize=fontsize+1)

    return ax


# TODO add gridlines possibility. With automaticallu picked color or supplied using kwargs?
# BUG saving as png is a problem, plots become blurred due to rotation
def _draw_2d_tri_map(matrix: np.ndarray, ax: Axes, cmap: str = "Reds", gridlines: bool = False):
    """Helper that draws a matrix as a rotated lower triangle heatmap on an existing matplotlib Axes object."""

    num_bins = matrix.shape[0]
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    matrix_masked = np.ma.masked_array(matrix, mask=~mask)

    im = ax.imshow(matrix_masked, origin="lower", cmap=cmap, interpolation="auto")
    # rotate ‑45° so that the matrix diagonal becomes the base of a triangle
    trans = transforms.Affine2D().rotate_deg(-45).scale(1/ math.sqrt(2), -1/ math.sqrt(2)) + ax.transData
    im.set_transform(trans)
    # clip to triangle
    coords = np.array([[-0.5, 0], [num_bins, 0], [num_bins / 2, num_bins / math.sqrt(2)]])
    triangle = Polygon(coords, closed=True, transform=ax.transData)
    im.set_clip_path(triangle)
    # setting axis limits
    xf = 0.05 * num_bins
    yf = -0.05 * num_bins
    ax.set_xlim(-xf, num_bins+xf)
    ax.set_ylim(yf, num_bins*0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    return im


# NOTE do not change values of to axis sizes, limits and font sizes, they are optimized to fit nicely
def plot_matrices_triangle(items:Iterable, titles:Iterable[str] = None, cmap="Reds", cbar_label="") -> Figure:
    """
    Plot a grid of matrices in triangular form with a colorbar
    
    Parameters
    --------
    items : iterable of np.ndarray
        matrices to plot
    titles : iterable of str
        titles/text to print beside the maps
    cmap : 
        colormap for the plot (default: Reds)
    cbar_label : str
        label for the colorbar
    
    Returns
    --------
    Figure object
    """

    if titles is None:
        titles = [""] * len(items)
    assert isinstance(titles, Iterable) and len(titles) == len(items)

    num_rows, num_cols = _get_grid_num_rows_cols(len(items))
    flag_no_grid = (num_cols * num_rows == 1)
    
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols,
                            figsize=(num_cols*5, num_rows*3),
                            layout="none" if flag_no_grid else "tight",
                            dpi=300)
    
    for ax, itm, ttl in zip(axs.flat if num_rows * num_cols > 1 else [axs], items, titles):
        
        # Take only upper triangle by masking the lower one
        if itm.ndim == 1:
            mat = squareform(itm)
            np.fill_diagonal(mat, np.nan)
        else:
            mat = itm[:]
        num_bc = mat.shape[0]
        lower = np.tril(mat, k=-1)
        mask = np.tril(np.ones_like(mat, dtype=bool), k=-1)
        data_masked = np.where(mask, lower, np.nan)
        
        # Values computed to be used in colorbar
        min_val = np.nanmin(data_masked)
        max_val = np.nanmax(data_masked)
        mid_val = (max_val + min_val) / 2
        
        # Drawing the map, creating space for the colorbar and drawing that as well
        im = _draw_2d_tri_map(mat, ax=ax, cmap=cmap)    # using internal routine
        ticks_bc = np.arange(0, num_bc, 5)
        ax.set_xticks(ticks_bc)
        for tk in ticks_bc:
            ax.text(x=tk, y=0.095, s="|",
                    transform=ax.get_xaxis_transform(), rotation=0,
                    ha="center", va="top",
                    fontsize=4, fontweight="bold", color="black")
            ax.text(x=tk, y=0.04 if flag_no_grid else 0.05, s=str(tk),
                    transform=ax.get_xaxis_transform(), rotation=0,
                    ha="center", va="top",
                    fontsize=6 if flag_no_grid else 8, fontweight="bold", color="black")
        # axpos = ax.get_position()
        # cax = fig.add_axes([axpos.x0+0.01, axpos.y0+0.05, axpos.width, axpos.height * 0.075])
        cbar = fig.colorbar(mappable=im, ax=ax, pad=0.1 if flag_no_grid else 0.075, shrink=0.5, aspect=15)
        cbar.set_ticks(ticks=[min_val, mid_val, max_val],
                       labels=[f"{min_val:.1f}", f"{mid_val:.1f}", f"{max_val:.1f}"],
                       fontsize=8 if flag_no_grid else 9)
        cbar.ax.set_ylabel(cbar_label, fontsize=10, labelpad=5)    # "Pseudocount", fontsize=font_info["tick_label"]
        cbar.outline.set_visible(False)
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.yaxis.set_ticks_position("left")
        
        # Setting the title/text for the map
        ax.text(x=0.04, y=0.2, s=ttl, transform=ax.transAxes, fontsize=6 if flag_no_grid else 9, rotation=45)

        if (flag_no_grid == False) and (len(axs.flat) > len(items)):
            for ax in axs.flat[len(items):]:
                # fig.delaxes(ax)
                ax.set_axis_off()

    return fig


# TODO add gridlines possibility, same as for triangle map
def _draw_2d_sq_map(matrix: np.ndarray, ax: Axes, cmap: str = "Reds", gridlines: bool = False):
    """Helper that draws a square matrix on an existing matplotlib Axes object"""

    im = ax.imshow(matrix, origin="lower", cmap=cmap, interpolation="auto")
    ax.yaxis.set_inverted(True)
    for _, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_aspect("equal")

    return im


# TODO include plots for a common colorbar case
def plot_matrices_square(items: Iterable, titles: Iterable[str] = None,
                         cmap="Reds", cbar_label=""):
    """
    Plot a tightly-spaced grid of matrices with or without a common colorbar in a square form.

    Used for plotting examples of pairwise proximity of contact matrices, components from topic decomposition and their
    comparisons. The number of rows and columns are automatically detected to have a square layout.

    Parameters
    --------
    matrices : list
        list of matrices to plot
    
    Returns
    --------
    fig : matplotlib.Figure
        Figure object after saving the plot
    """

    if titles is None:
        titles = [""] * len(items)
    assert isinstance(titles, Iterable) and len(titles) == len(items)

    num_rows, num_cols = _get_grid_num_rows_cols(len(items))
    flag_no_grid = (num_cols * num_rows == 1)

    num_rows, num_cols = _get_grid_num_rows_cols(len(items))
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols,
                            figsize=(num_cols*5, num_rows*4),
                            layout="none",
                            dpi=300)
    # if common_cbar:
        # fig.subplots_adjust(bottom=0.8)
        # cbar_ax = fig.add_axes([0.1, 0.01, 0.8, 0.03])

    for ax, itm, ttl in zip(axs.flat if num_rows * num_cols > 1 else [axs], items, titles):
        
        if itm.ndim == 1:
            mat = squareform(itm)
            np.fill_diagonal(mat, np.nan)
        else:
            mat = itm[:]
        num_bc = mat.shape[0]
        
        # Values computed to be used in colorbar
        min_val = np.nanmin(mat)
        max_val = np.nanmax(mat)
        mid_val = (max_val + min_val) / 2
        
        # Drawing the map, creating space for the colorbar and drawing that as well
        im = _draw_2d_sq_map(mat, ax=ax, cmap=cmap)    # using internal routine
        ticks_bc = np.arange(0, num_bc, 5)
        ax.tick_params(bottom=True, top=True, right=True, left=True)
        ax.set_xticks(ticks_bc)
        ax.set_yticks(ticks_bc)
        ax.set_xticklabels(ticks_bc, fontsize=8 if flag_no_grid else 9, fontweight="bold", color="black")
        ax.set_yticklabels(ticks_bc, fontsize=8 if flag_no_grid else 9, fontweight="bold", color="black")

        cbar = fig.colorbar(mappable=im, ax=ax, pad=0.175 if flag_no_grid else 0.225, shrink=0.5, aspect=15)
        cbar.set_ticks(ticks=[min_val, mid_val, max_val],
                       labels=[f"{min_val:.1f}", f"{mid_val:.1f}", f"{max_val:.1f}"],
                       fontsize=9 if flag_no_grid else 11)
        cbar.ax.set_ylabel(cbar_label, fontsize=11 if flag_no_grid else 12, labelpad=5)    # "Pseudocount", fontsize=font_info["tick_label"]
        cbar.outline.set_visible(False)
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.yaxis.set_ticks_position("left")
        
        # Setting the title/text for the map
        ax.set_title(ttl, fontsize=10 if flag_no_grid else 13, loc="center", y=1.025)

        if (flag_no_grid == False) and (len(axs.flat) > len(items)):
            for ax in axs.flat[len(items):]:
                # fig.delaxes(ax)
                ax.set_axis_off()
    
    return fig


# ---------------------------------------------------------------------------
#  Main pyramid plot with OR overlay
# ---------------------------------------------------------------------------

def plot_pyramid_topic(topic_n: int, topic: np.ndarray, topic_repr: float,
                       lib, locus_dict: Dict,
                       figname: str,
                       font_info: Dict[str, int],
                       line_sep: float,
                       cmap: str = "Reds",
                       bright_q: float = BRIGHT_Q) -> None:
    """Plot a single topic with annotation **and** edge‑type enrichment.

    Besides the behaviour of the original implementation this version:
      • computes per‑matrix odds ratios using ``odds_ratio_matrix``;
      • prints every edge type with OR > 1 (sorted descending) onto the
        triangle, one per line, small font.
    """
    # ------------------------------------------------------------------
    #  Prepare data
    # ------------------------------------------------------------------
    topic_sq = squareform(topic) if topic.ndim == 1 else topic

    or_dict = _odds_ratio_matrix(topic_sq, locus_dict, bright_q)
    print(or_dict)
    enrich  = [(et, OR) for et, (OR, _) in or_dict.items() if OR > 1]
    enrich.sort(key=lambda x: x[1], reverse=True)
    print(enrich)

    # ------------------------------------------------------------------
    #  Base triangular heat‑map (unchanged from original)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    im = _draw_2d_tri_map(topic_sq, ax=ax, cmap=cmap)

    tick_dict = hp.parse_locus_labels(locus_dict, lib)
    ax = _annotate_labels(ax, tick_dict, locus_dict, lib,
                         fontsize=font_info["tick_label"],
                         line_sep=line_sep)
    _add_gene_legend(ax, locus_dict,
                    location="outside", anchor=(0, 0.5),
                    fontsize=font_info["legend"])

    # ------------------------------------------------------------------
    #  Overlay ORs  -----------------------------------------------------
    # ------------------------------------------------------------------
    if enrich:
        txt_lines = [f"{et}: {OR:.{OR_DIGITS}f}" for et, OR in enrich]
        ax.text(0.15, 0.45, "\n".join(txt_lines),
                transform=ax.transAxes,
                ha="left", va="top", fontsize=font_info["tick_label"],   # - 3 (before)
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=0.7))

    # ------------------------------------------------------------------
    #  Colour‑bar and title (original style) ----------------------------
    # ------------------------------------------------------------------
    axpos = ax.get_position()
    cax = fig.add_axes([axpos.x1 + 0.01, axpos.y0, 0.03, axpos.height / 2])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("Pseudocount", fontsize=font_info["tick_label"])
    cbar.ax.tick_params(labelsize=font_info["tick_label"] - 2)

    ax.text(0.5, 0.55,
            f"CFM {topic_n} – representativity: {round(topic_repr*100, 1)}%",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=font_info["title"], fontweight="bold")

    # save & close
    plt.savefig(figname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def combine_images(columns, space, images, output_path='image.png'):
    rows = len(images) // columns + (len(images) % columns > 0)
    width_max = max(Image.open(image).width for image in images)
    height_max = max(Image.open(image).height for image in images)

    background_width = width_max * columns + space * (columns - 1)
    background_height = height_max * rows + space * (rows - 1)
    background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))

    x = 0
    y = 0
    for i, image in enumerate(images):
        img = Image.open(image)
        x_offset = (width_max - img.width) // 2
        y_offset = (height_max - img.height) // 2
        background.paste(img, (x + x_offset, y + y_offset))
        x += width_max + space
        if (i + 1) % columns == 0:
            y += height_max + space
            x = 0

    background.save(output_path)
    print(f"Saved combined image to: {output_path}")


def plot_traces_3d_simple(traces: np.ndarray, use_one_ax=False):
    """
    Plot up to fifty traces as simple graphs in 3D
    """

    assert isinstance(traces, np.ndarray)
    num_traces, num_bc, _ = traces.shape

    if use_one_ax:
        nr = 1
        nc = 1
    else:
        nr, nc = _get_grid_num_rows_cols(num_traces)

    f = 8   # factor for scaling of plot based on number of rows and columns

    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    fd = {"fontfamily": "sans-serif", "fontsize": 20}

    # # To set local rcParams
    # with plt.rc_context({"axes.grid": True, "grid.linewidth": 0.75}):
    #     f, ax = plt.subplots()  # Will make a figure with a grid
    #     ax.plot(x, y)

    # f, ax = plt.subplots()  # Will make a figure without a grid
    # ax.plot(x, y)

    fig, axs = plt.subplots(nrows=nr, ncols=nc, subplot_kw=dict(projection="3d"))
    fig.set_figwidth(f * nr)
    fig.set_figheight(f * nc)
    for ax, n in zip(axs.flat if nr > 1 else [axs], range(num_traces)):
        x, y, z = zip(*traces[n])
        ax.scatter(x, y, z, s=70, marker="o")
        ax.plot(x, y, z, linestyle="-", linewidth=1.25)
        for i in range(0, num_bc, 3):
            ax.text(x[i], y[i], z[i], str(i), fontdict=fd)
        ax.set_xlabel("X-axis", labelpad=15, fontdict=fd)
        ax.set_ylabel("Y-axis", labelpad=15, fontdict=fd)
        ax.set_zlabel("Z-axis", labelpad=15, fontdict=fd)
    fig.set_layout_engine("tight")
    
    return fig
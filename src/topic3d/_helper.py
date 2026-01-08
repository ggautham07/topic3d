#!/usr/bin/env python
"""
Helper functions to work with chromatin tracing data (:mod:`topic3d._helper`)
=====================================================

.. authors:: ggautham07

Function reference
------------------
"""

import numpy as np
import pandas as pd
from pathlib import Path
from numbers import Number


def normalize_minmax(arr:np.ndarray) -> np.ndarray:
    """
    Normalize array values to [0, 1] using min-max normalization.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Normalized array.
    """

    arr_min = np.min(arr)
    arr_rescaled = (arr - arr_min) / (np.max(arr) - arr_min)
    
    return arr_rescaled


def rescale_minmax(arr:np.ndarray, ref:np.ndarray) -> np.ndarray:
    """
    Rescale arr to the range of ref using min-max normalization.

    Parameters
    ----------
    arr : np.ndarray
        Array to rescale.
    ref : np.ndarray
        Reference array to match range.

    Returns
    -------
    np.ndarray
        Rescaled array.
    """
    
    assert np.logical_not(np.isnan(arr).all())

    ref = ref[np.logical_not(np.isnan(ref))]
    new_min = np.min(ref)
    new_max = np.max(ref)

    arr_rescaled = (normalize_minmax(arr) * (new_max - new_min)) + new_min

    return arr_rescaled


def _bins_equilen(start, end, num_bins):
    """Generate bins of lengths that are as close to equal as possible"""

    return np.linspace(start, end, num=num_bins+1, dtype=int)


def _bins_randlen(start, end, num_bins):
    """Generate bins of random lengths"""

    bins_raw = np.random.random(num_bins).cumsum()
    bins_raw = (bins_raw - bins_raw.min()) / bins_raw.ptp()
    bins = np.asarray((end - start) * bins_raw + start, dtype=int)
    
    return bins


def generate_bc_bins(start:int, end:int, num_bins:int, bin_lengths:str = "equal"):
    """
    Generate barcode targets of equal (as much as possible) or random lengths.

    Used mainly to perform coarse graining of polymer conformations from simulations to match 
    traces from experiments.

    Parameters
    ----------
    start : int
        start position
    end : int
        end position
    num_bins : int
        number of bins required
    bin_lengths : str
        whether bins should be of "equal" (default) and "random" lengths

    Returns
    ----------
    barcode target bins
    """
    
    method = {"equal": _bins_equilen, "random": _bins_randlen}
    bins = method[bin_lengths](start, end, num_bins)
    tgts = np.concatenate((bins[:-1][...,np.newaxis], bins[1:][...,np.newaxis]), axis=1)

    return tgts


def monomer_idx_from_genomic_pos(positions:np.ndarray | list, start:int, monomer_size:int):
    """
    Convert genomic positions to monomer indices given a start position and monomer size

    Parameters
    ----------
    positions : array of positions
    start : start position
    monomer_size : size of each monomer in bases

    Returns
    ----------
    Monomer indices
    """

    if isinstance(positions, Number):
        assert positions >= start
    assert np.min(positions) >= start

    return (positions - start) // monomer_size


def bc_idx_from_monomer_idx(positions:np.ndarray | list, barcode_targets: np.ndarray) -> np.ndarray:
    """
    Find which barcode index a monomer position falls into given a set of barcode targets.
    
    Both scalars and a vector of scalars are accepted.

    Parameters
    ----------
    positions : array of positions
    barcode_targets : barcode target bins with each row having a barcode's start and end position

    Returns
    ----------
    Barcode index into which the position falls
    """

    positions = np.asarray(positions)
    assert np.all(np.logical_and(positions <= np.max(barcode_targets), positions >= np.min(barcode_targets)))

    return np.array([np.argmax(pos < barcode_targets[:, 1]) for pos in positions])


# TODO properly integrate marcnol's function with the new plotting routine
# TODO generalize output format for other analyes
def parse_locus_labels(loci_dict, lib):
    """
    Parsing function to handle the different labels from the json file:
    - Handles CTCF as before, indicating the associated barcode and its orientation if it is known
    - Handles genes by assigning a unique color to each gene's barcode - a legend for the plot will be added later
    - Handles CREs
    @param: loci_dict (dict): contains all the loci that should be displayed in the label axis
    @param: lib (dataframe): library associated to the locus
    @return: tick_dict (dict): dictionary containing for each barcode a list of tuples with the following information:
        - group_index - will determine on which line the label will be represented (0 is by default associated to genes,
        1 to CTCF and 2 to CREs and it corresponds to the order in the input json file)
        - label_text - for CREs it will be the number of the CRE, for a CTCF site, its orientation
        - the nature of group ("gene", "CTCF" or "CREs")
        - the last entry will defer according to the nature of the label. If it is a gene, the last entry will be the
        color code associated to the gene. Else, it will indicate the positions of the ticks.

    """
    tick_dict = {}
    bc_list = lib["barcode_id"].tolist()
    # gene_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    # color_idx = 0

    for group_index, (locus_type, group_data) in enumerate(loci_dict.items()):
        if locus_type == "CTCF":
            orientation_map = {}
            for orientation, barcode_list in group_data.items():
                for p in barcode_list if isinstance(barcode_list, list) else [barcode_list]:
                    idx = bc_list.index(p)
                    tick = idx
                    orientation_map.setdefault(tick, []).append(orientation)
            for tick, orientations in orientation_map.items():
                label_text = '\n'.join(orientations)
                tick_dict.setdefault(tick, []).append((group_index, label_text, "CTCF"))
            continue

        if locus_type == "genes":
            for gene, (barcodes, color) in group_data.items():
                # color = gene_palette[color_idx % len(gene_palette)]
                # color_idx += 1
                for p in barcodes if isinstance(barcodes, list) else [barcodes]:
                    idx = bc_list.index(p)
                    tick = idx
                    tick_dict.setdefault(tick, []).append((group_index, "", "gene", color))
            continue

        if locus_type == "CREs":
            for cre, barcodes in group_data.items():
                if isinstance(barcodes, list):
                    idx_min = bc_list.index(min(barcodes))
                    idx_max = bc_list.index(max(barcodes))
                    tick = (idx_min + idx_max) / 2
                    label_text = str(cre)
                    tick_dict.setdefault(tick, []).append((group_index, label_text, "CREs", [idx_min, idx_max]))
                else:
                    tick = bc_list.index(barcodes)
                    label_text = str(cre)
                    tick_dict.setdefault(tick, []).append((group_index, label_text, "CREs", [tick, tick]))

        for locus, bc in group_data.items():
            if isinstance(bc, list):
                idx_min = bc_list.index(min(bc))
                idx_max = bc_list.index(max(bc))
                tick = (idx_min + idx_max) / 2
            else:
                tick = bc_list.index(bc)
            label_text = str(locus)
            tick_dict.setdefault(tick, []).append((group_index, label_text))

    return tick_dict
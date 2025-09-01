#!/usr/bin/env python
"""
Helper functions to work with chromatin tracing data (:mod:`topic3d.helper`)
=====================================================

.. authors:: ggautham07

Function reference
------------------
"""


import numpy as np
import pandas as pd
from pathlib import Path
from numbers import Number


########################

def _equilen_bins(start, end, num_bins):
    "Generate bins of lengths that are as close to equal as possible"
    return np.linspace(start, end, num=num_bins+1, dtype=int)


def _randlen_bins(start, end, num_bins):
    "Generate bins of random lengths"
    bins_raw = np.random.random(num_bins).cumsum()
    bins_raw = (bins_raw - bins_raw.min()) / bins_raw.ptp()
    bins = np.asarray((end - start) * bins_raw + start, dtype=int)
    return bins

########################


def generate_barcode_targets(start:int, end:int, num_bins:int, bin_lengths="equal"):

    """
    Generate barcode targets of equal (as much as possible) or random lengths to perform coarse graining of polymer conformations

    Parameters
    ----------
    start : start position
    end : end position
    num_bins : number of bins required
    bin_lengths : bin lengths property, accepted values are "equal" and "random"

    Returns
    ----------
    barcode_targets : barcode target bins
    """
    
    method = {"equal": _equilen_bins, "random": _randlen_bins}
    bins = method[bin_lengths](start, end, num_bins)
    barcode_targets = np.concatenate((bins[:-1][...,np.newaxis], bins[1:][...,np.newaxis]), axis=1)

    return barcode_targets

########################


def genomic_position_to_monomer(positions:np.ndarray | list, start:int, monomer_size:int):

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

########################


def position_to_barcode_index(positions:np.ndarray | list, barcode_targets: np.ndarray) -> np.ndarray:

    """
    Find which barcode index a monomer position falls into given a set of barcode targets. Both scalars and a vector of scalars are accepted

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

########################
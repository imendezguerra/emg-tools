"""Rearrange EMG channels on an electrode grid."""

from typing import Optional, Tuple, Union

import numpy as np


def arrange_data_spatially(
    data: np.ndarray,
    ch_map: np.ndarray
    ) -> np.ndarray:

    """Arrange (samples, chs) data on the electrode grid given by the channel map.

    Grid positions with a negative or out-of-range channel index are filled with zeros.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        ch_map (np.ndarray): Channel map with shape (rows, cols) holding the 0-based
            channel index of each electrode, and -1 for empty positions.

    Returns:
        np.ndarray: Spatially arranged data with shape (rows, cols, samples).
    """

    rows, cols = ch_map.shape
    samples, chs = data.shape
    data_spatial = np.zeros((rows, cols, samples))

    for row, col in np.ndindex(ch_map.shape):
        ch = ch_map[row, col]
        if (ch < 0) or (ch >= chs):
            continue
        data_spatial[row, col] = data[:, ch]

    return data_spatial


def flatten_data_spatially(
    data: np.ndarray,
    ch_map: np.ndarray
    ) -> np.ndarray:

    """Flatten (rows, cols, samples) grid data back to (samples, chs).

    Inverse of `arrange_data_spatially`. Channels missing from the map are filled with zeros.

    Args:
        data (np.ndarray): Input data with shape (rows, cols, samples).
        ch_map (np.ndarray): Channel map with shape (rows, cols) holding the 0-based
            channel index of each electrode, and -1 for empty positions.

    Returns:
        np.ndarray: Flattened data with shape (samples, chs), where chs is
            `ch_map.max() + 1`.
    """

    samples = data.shape[-1]
    chs = np.amax(ch_map) + 1

    data_flat = np.zeros((samples, chs))

    for ch in range(chs):
        idx = np.where(ch_map == ch)
        if len(idx[0]):
            data_flat[:, ch] = data[idx[0], idx[1]]

    return data_flat


def ch_map_to_coord(
    ch_map: np.ndarray,
    ied: Optional[Union[float, Tuple[float, float]]] = 1.0,
    ) -> np.ndarray:

    """Convert a channel map into the physical (x, y) position of each channel.

    The column of an electrode gives x and its row gives y, both scaled by the
    inter-electrode distance. Row i of the output is channel i, matching the
    columns of (samples, chs) data.

    Args:
        ch_map (np.ndarray): Channel map with shape (rows, cols) holding the 0-based
            channel index of each electrode, and -1 for empty positions.
        ied (Union[float, Tuple[float, float]], optional): Inter-electrode
            distance, or (x, y) distances for non-square grids. Default is 1.0.

    Returns:
        np.ndarray: Channel coordinates with shape (chs, 2), where
            chs = max(ch_map) + 1. Channels not on the grid are NaN.
    """

    # Initialise variables
    chs = np.amax(ch_map) + 1
    ch_coord = np.full((chs, 2), np.nan)

    # Place each mapped channel at its scaled (col, row) position
    rows, cols = np.nonzero(ch_map >= 0)
    ch_coord[ch_map[rows, cols]] = np.c_[cols, rows] * np.asarray(ied, dtype=float)

    return ch_coord

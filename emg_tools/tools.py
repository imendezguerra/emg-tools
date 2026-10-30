"""Rearrange EMG channels on an electrode grid and replace bad channels."""

from copy import copy
from typing import Union

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


def replace_bad_ch(
    data: np.ndarray,
    bad_ch: Union[list, np.ndarray],
    ch_map: np.ndarray,
    ) -> np.ndarray:

    """Replace bad channels with the mean of their good neighbouring channels.

    Neighbours are the up to 8 surrounding electrodes on the grid. Bad channels are
    replaced in the given order, so a replaced channel can serve as a neighbour for the
    next ones. Bad channels without good neighbours are set to zero.

    Args:
        data (np.ndarray): Input data with shape (samples, chs) or (rows, cols, samples).
        bad_ch (Union[list, np.ndarray]): 0-based indices of the bad channels.
        ch_map (np.ndarray): Channel map with shape (rows, cols) holding the 0-based
            channel index of each electrode, and -1 for empty positions.

    Returns:
        np.ndarray: Copy of the data, same shape as the input, with the bad channels replaced.
    """

    # Initialise variables
    rows, cols = ch_map.shape
    data_out = copy(data)
    if isinstance(bad_ch, list):
        bad_ch_out = copy(bad_ch)
    else:
        bad_ch_out = copy(bad_ch).tolist()

    # Substitute bad channels until there are no more bad channels left
    while len(bad_ch_out) > 0:

        # Set updated bad channels to -1
        ch_map_out = copy(ch_map)
        for ch in bad_ch_out:
            ch_map_out[ ch_map == ch ] = -1

        # Get first bad channel
        curr_bad_ch = bad_ch_out.pop(0)

        # Find coordinates of the current bad channel
        [x, y] = np.where(ch_map == curr_bad_ch)
        x = x[0]
        y = y[0]

        # # Define mask
        neigh_mask = np.array([
            [x, y + 1],
            [x, y - 1],
            [x + 1, y],
            [x - 1, y],
            [x + 1, y + 1],
            [x + 1, y - 1],
            [x - 1, y + 1],
            [x - 1, y - 1]
            ])

        # Remove neighbours out of bounds
        in_range = np.logical_and(
            np.logical_and( neigh_mask[:,0] > -1, neigh_mask[:,0] < rows),
            np.logical_and( neigh_mask[:,1] > -1, neigh_mask[:,1] < cols)
        )
        neigh_mask = neigh_mask[in_range]
        in_range = in_range[in_range]

        # Check for good neighbours
        for i in range(neigh_mask.shape[0]):
            if (ch_map_out[ neigh_mask[i,0], neigh_mask[i,1] ] < 0):
                in_range[i] = False

        # Get the actual neighbouring channels
        neigh_mask = neigh_mask[in_range, :]
        neigh_chs = ch_map[neigh_mask[:,0], neigh_mask[:,1]]

        # Substitue channels (set to zero if there are no good neighbours)
        if len(data_out.shape) == 2: # (samples, chs) format
            if len(neigh_chs) == 0:
                data_out[:, curr_bad_ch] = 0
            else:
                data_out[:, curr_bad_ch] = np.mean( data_out[:, neigh_chs], axis=-1)
        elif len(data_out.shape) == 3: # (ch_rows, ch_cols, samples) format
            if len(neigh_chs) == 0:
                data_out[x,y] = 0
            else:
                data_out[x,y] = np.mean( data_out[neigh_mask[:,0], neigh_mask[:,1]], axis=0)

    return data_out

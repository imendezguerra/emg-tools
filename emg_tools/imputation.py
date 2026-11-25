"""Detect outliers and impute bad channels of EMG signals."""

from typing import Literal, Optional, Tuple, Union

import numpy as np
from scipy import interpolate, spatial


def find_outliers(data: np.ndarray, k: Optional[float] = 1.5) -> np.ndarray:
    """Find outliers in a distribution with Tukey's fences.

    Args:
        data (np.ndarray): Input data with shape (dim,).
        k (float, optional): Fence multiplier of the IQR. Default is 1.5.

    Returns:
        np.ndarray: Sorted indices of the outlying values.
    """

    # Compute the fences
    q25, q75 = np.quantile(data, [0.25, 0.75])
    iqr = q75 - q25

    # Find values outside the fences
    outliers = np.where((data < q25 - k * iqr) | (data > q75 + k * iqr))[0]

    return outliers


def replace_bad_ch(
    data: np.ndarray,
    bad_ch: Union[list, np.ndarray],
    ch_coord: np.ndarray,
    method: Optional[Literal['max_dist', 'delaunay']] = 'max_dist',
    max_dist: Optional[float] = None,
    verbose: Optional[bool] = False,
    ) -> np.ndarray:

    """Replace bad channels with the mean of their good neighbouring channels.

    Neighbours are found with one of two methods:

    - 'max_dist': the good channels within max_dist of the bad channel. By
      default max_dist = ied * sqrt(2), with the inter-electrode distance (ied)
      estimated as the median nearest-neighbour distance. On a regular grid this
      gives the up to 8 surrounding electrodes. For non-square grids, pass
      max_dist.
    - 'delaunay': the good channels sharing an edge with the bad channel in the
      Delaunay triangulation of the good channels plus that bad channel, or the 2
      nearest good channels if the points can't be triangulated (e.g. a single
      line of electrodes). Meant for irregular layouts: on a regular grid it
      links only one arbitrary diagonal per square.

    Bad channels are replaced in the given order, so a replaced channel can serve
    as a neighbour for the next ones. Bad channels without good neighbours are
    set to zero.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        bad_ch (Union[list, np.ndarray]): 0-based indices of the bad channels.
        ch_coord (np.ndarray): Physical (x, y) position of each channel with
            shape (chs, 2), NaN for channels without a position. See
            `tools.ch_map_to_coord` for grids.
        method (str, optional): Neighbour method, 'max_dist' or 'delaunay'.
            Default is 'max_dist'.
        max_dist (float, optional): Neighbour radius for 'max_dist', in the
            units of ch_coord. Default is None (ied * sqrt(2)).
        verbose (bool, optional): Print each replaced channel and its
            neighbours. Default is False.

    Returns:
        np.ndarray: Copy of the data with the bad channels replaced.
    """

    # Initialise variables
    coord, bad = _check_inputs(data, bad_ch, ch_coord)
    if method not in ('max_dist', 'delaunay'):
        raise ValueError(f"method must be 'max_dist' or 'delaunay', got {method!r}")
    data_out = np.array(data, dtype=float)
    is_good = np.isfinite(coord).all(axis=1)

    # Estimate the neighbour radius from the median nearest-neighbour distance
    if method == 'max_dist' and max_dist is None:
        nn_dist = spatial.cKDTree(coord[is_good]).query(coord[is_good], k=2)[0][:, 1]
        max_dist = np.median(nn_dist) * np.sqrt(2)

    is_good[bad] = False
    for ch in bad:
        good = np.flatnonzero(is_good)
        dist = np.linalg.norm(coord[good] - coord[ch], axis=1)

        # Find the neighbours within max_dist
        if method == 'max_dist':
            neigh = good[(dist <= max_dist) | np.isclose(dist, max_dist)]

        # Find the Delaunay neighbours, or the 2 nearest channels if collinear
        else:
            idx = np.r_[good, ch]
            try:
                indptr, indices = spatial.Delaunay(coord[idx]).vertex_neighbor_vertices
                neigh = idx[indices[indptr[-2]:indptr[-1]]]
            except (RuntimeError, ValueError):  # QhullError is a RuntimeError
                neigh = good[np.argsort(dist)[:2]]

        # Substitute channel (set to zero if there are no good neighbours)
        data_out[:, ch] = np.mean(data_out[:, neigh], axis=1) if neigh.size else 0
        is_good[ch] = True

        if verbose:
            print(f"Channel {ch}: replaced by mean of channels {neigh.tolist()}")

    return data_out


def interp_bad_ch(
    data: np.ndarray,
    bad_ch: Union[list, np.ndarray],
    ch_coord: np.ndarray,
    kernel: Optional[str] = "thin_plate_spline",
    verbose: Optional[bool] = False,
    **rbf_kwargs,
    ) -> np.ndarray:

    """Replace bad channels by radial basis function (RBF) interpolation.

    A smooth 2D field is fitted through the good channels and evaluated at the
    bad channels' positions. The interpolation is linear in the data, so it is
    computed once as a weight matrix W with shape (bad, good) and applied to all
    samples: data[:, bad] = data[:, good] @ W.T.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        bad_ch (Union[list, np.ndarray]): 0-based indices of the bad channels.
        ch_coord (np.ndarray): Physical (x, y) position of each channel with
            shape (chs, 2), NaN for channels without a position.
        kernel (str, optional): RBF kernel, see
            `scipy.interpolate.RBFInterpolator`. Default is
            "thin_plate_spline", which needs no tuning. Kernels with a shape
            parameter (`epsilon`) depend on the units of ch_coord.
        verbose (bool, optional): Print each bad channel with the 5 good
            channels weighing most in its interpolation. Default is False.
        **rbf_kwargs: Passed to `scipy.interpolate.RBFInterpolator`, e.g.
            `smoothing` or `epsilon`.

    Returns:
        np.ndarray: Copy of the data with the bad channels replaced.

    Raises:
        ValueError: If the good channels lie on a single line, where a 2D
            field can't be fitted.
    """

    # Initialise variables
    coord, bad = _check_inputs(data, bad_ch, ch_coord)
    bad = np.unique(bad)
    good = np.setdiff1d(np.flatnonzero(np.isfinite(coord).all(axis=1)), bad)

    # Interpolate the identity to get the weights
    try:
        rbf = interpolate.RBFInterpolator(
            coord[good], np.eye(len(good)), kernel=kernel, **rbf_kwargs
            )
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "Can't fit a 2D field through the good channels, they may lie on a single "
            "line. Use replace_bad_ch instead."
            ) from err
    weights = rbf(coord[bad])

    # Apply the weights to all samples
    data_out = np.array(data, dtype=float)
    data_out[:, bad] = data_out[:, good] @ weights.T

    if verbose:
        for ch, w in zip(bad, weights, strict=False):
            top = np.argsort(-np.abs(w))[:5]
            print(f"Channel {ch}: " + ", ".join(f"{good[i]} ({w[i]:.2f})" for i in top))

    return data_out


def get_loo_error(
    data: np.ndarray,
    ch_coord: np.ndarray,
    bad_ch: Optional[Union[list, np.ndarray]] = (),
    **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:

    """Compute the leave-one-out error of `interp_bad_ch`.

    Each good channel is predicted from the other good channels and compared with
    its actual signal. Use it to compare kernels or settings (lower is better),
    and to check how predictable the channels are from their surroundings. A
    short segment of data is enough.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        ch_coord (np.ndarray): Physical (x, y) position of each channel with
            shape (chs, 2), NaN for channels without a position.
        bad_ch (Union[list, np.ndarray], optional): 0-based indices of the bad
            channels, excluded from the check. Default is none.
        **kwargs: Passed to `interp_bad_ch`, e.g. `kernel` or `smoothing`.

    Returns:
        error (np.ndarray): Relative error ||predicted - actual|| / ||actual||
            per good channel, with shape (good chs,). 0 is perfect and 1 is as
            bad as predicting zeros.
        good (np.ndarray): 0-based indices of the good channels, with shape
            (good chs,).
    """

    # Initialise variables
    coord, bad = _check_inputs(data, bad_ch, ch_coord)
    good = np.setdiff1d(np.flatnonzero(np.isfinite(coord).all(axis=1)), bad)
    error = np.zeros(len(good))

    # Predict each good channel from the others
    for i, ch in enumerate(good):
        pred = interp_bad_ch(data, np.r_[bad, ch], coord, **kwargs)[:, ch]
        error[i] = np.linalg.norm(pred - data[:, ch]) / np.linalg.norm(data[:, ch])

    return error, good


def _check_inputs(
    data: np.ndarray,
    bad_ch: Union[list, np.ndarray],
    ch_coord: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

    """Validate the channel coordinates and bad channels against the data.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        bad_ch (Union[list, np.ndarray]): 0-based indices of the bad channels.
        ch_coord (np.ndarray): Physical (x, y) position of each channel with
            shape (chs, 2).

    Returns:
        coord (np.ndarray): Channel coordinates as floats, with shape (chs, 2).
        bad (np.ndarray): Bad channels as integers, in the given order.

    Raises:
        ValueError: If the shapes don't match or a bad channel has no position.
    """

    coord = np.asarray(ch_coord, dtype=float)
    bad = np.asarray(bad_ch, dtype=int).ravel()

    if data.ndim != 2 or coord.shape != (data.shape[1], 2):
        raise ValueError(
            f"Expected data (samples, chs) and ch_coord (chs, 2), got {data.shape} "
            f"and {coord.shape}."
            )
    if not np.isfinite(coord[bad]).all():
        raise ValueError("Every bad channel needs a finite position in ch_coord.")

    return coord, bad

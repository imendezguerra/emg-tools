"""Detect outliers, find and impute bad channels of EMG signals."""

import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy import interpolate, spatial

from emg_tools import feats, freq


def find_outliers(
    data: np.ndarray,
    k: Optional[float] = 1.5,
    side: Optional[Literal['both', 'low', 'high']] = 'both',
    ) -> np.ndarray:
    """Find outliers in a distribution with Tukey's fences.

    NaNs are ignored and never returned as outliers.

    Args:
        data (np.ndarray): Input data with shape (dim,).
        k (float, optional): Fence multiplier of the IQR. Default is 1.5.
        side (str, optional): Fences to test, 'both', 'low' or 'high'.
            Default is 'both'.

    Returns:
        np.ndarray: Sorted indices of the outlying values.
    """

    # Compute the fences
    q25, q75 = np.nanquantile(data, [0.25, 0.75])
    iqr = q75 - q25

    # Find values outside the fences
    low = (data < q25 - k * iqr) & (side != 'high')
    high = (data > q75 + k * iqr) & (side != 'low')
    outliers = np.flatnonzero(low | high)

    return outliers


def get_neighbours(
    ch: int,
    ch_coord: np.ndarray,
    method: Optional[Literal['max_dist', 'delaunay']] = 'max_dist',
    max_dist: Optional[float] = None,
    exclude: Optional[Union[list, np.ndarray]] = (),
    ) -> np.ndarray:

    """Find the neighbouring channels of a channel from their physical positions.

    Neighbours are found with one of two methods:

    - 'max_dist': the channels within max_dist of the channel. By default
      max_dist = ied * sqrt(2), with the inter-electrode distance (ied)
      estimated as the median nearest-neighbour distance of all positioned
      channels. On a regular grid this gives the up to 8 surrounding
      electrodes. For non-square grids, pass max_dist.
    - 'delaunay': the channels sharing an edge with the channel in the Delaunay
      triangulation of the candidate channels plus that channel, or the 2
      nearest candidates if the points can't be triangulated (e.g. a single
      line of electrodes). Meant for irregular layouts: on a regular grid it
      links only one arbitrary diagonal per square.

    Args:
        ch (int): 0-based index of the channel.
        ch_coord (np.ndarray): Physical (x, y) position of each channel with
            shape (chs, 2), NaN for channels without a position. See
            `tools.ch_map_to_coord` for grids.
        method (str, optional): Neighbour method, 'max_dist' or 'delaunay'.
            Default is 'max_dist'.
        max_dist (float, optional): Neighbour radius for 'max_dist', in the
            units of ch_coord. Default is None (ied * sqrt(2)).
        exclude (Union[list, np.ndarray], optional): 0-based indices of channels
            that can't be neighbours, e.g. bad channels. They still count for
            the default max_dist. Default is none.

    Returns:
        np.ndarray: Sorted 0-based indices of the neighbours.

    Raises:
        ValueError: If the method is unknown or the channel has no position.
    """

    # Initialise variables
    coord = np.asarray(ch_coord, dtype=float)
    if method not in ('max_dist', 'delaunay'):
        raise ValueError(f"method must be 'max_dist' or 'delaunay', got {method!r}")
    if not np.isfinite(coord[ch]).all():
        raise ValueError(f"Channel {ch} needs a finite position in ch_coord.")
    is_pos = np.isfinite(coord).all(axis=1)

    # Estimate the neighbour radius from the median nearest-neighbour distance
    if method == 'max_dist' and max_dist is None:
        nn_dist = spatial.cKDTree(coord[is_pos]).query(coord[is_pos], k=2)[0][:, 1]
        max_dist = np.median(nn_dist) * np.sqrt(2)

    # Candidates: positioned channels, other than ch and the excluded ones
    is_cand = is_pos.copy()
    is_cand[np.asarray(exclude, dtype=int)] = False
    is_cand[ch] = False
    cand = np.flatnonzero(is_cand)
    dist = np.linalg.norm(coord[cand] - coord[ch], axis=1)

    # Find the neighbours within max_dist
    if method == 'max_dist':
        neigh = cand[(dist <= max_dist) | np.isclose(dist, max_dist)]

    # Find the Delaunay neighbours, or the 2 nearest channels if collinear
    else:
        idx = np.r_[cand, ch]
        try:
            indptr, indices = spatial.Delaunay(coord[idx]).vertex_neighbor_vertices
            neigh = idx[indices[indptr[-2]:indptr[-1]]]
        except (RuntimeError, ValueError):  # QhullError is a RuntimeError
            neigh = cand[np.argsort(dist)[:2]]

    return np.sort(neigh)


def replace_bad_ch(
    data: np.ndarray,
    bad_ch: Union[list, np.ndarray],
    ch_coord: np.ndarray,
    neighbour_method: Optional[Literal['max_dist', 'delaunay']] = 'max_dist',
    max_dist: Optional[float] = None,
    verbose: Optional[bool] = False,
    ) -> np.ndarray:

    """Replace bad channels with the mean of their good neighbouring channels.

    Neighbours are found with `get_neighbours`. Bad channels are replaced in the
    given order, so a replaced channel can serve as a neighbour for the next
    ones. Bad channels without good neighbours are set to zero.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        bad_ch (Union[list, np.ndarray]): 0-based indices of the bad channels.
        ch_coord (np.ndarray): Physical (x, y) position of each channel with
            shape (chs, 2), NaN for channels without a position. See
            `tools.ch_map_to_coord` for grids.
        neighbour_method (str, optional): Neighbour method, passed to
            `get_neighbours`: 'max_dist' or 'delaunay'. Default is 'max_dist'.
        max_dist (float, optional): Neighbour radius for 'max_dist', in the
            units of ch_coord. Default is None (ied * sqrt(2)).
        verbose (bool, optional): Print each replaced channel and its
            neighbours. Default is False.

    Returns:
        np.ndarray: Copy of the data with the bad channels replaced.
    """

    # Initialise variables
    coord, bad = _check_inputs(data, bad_ch, ch_coord)
    data_out = np.array(data, dtype=float)

    for i, ch in enumerate(bad):
        # Channels still pending replacement are not neighbours
        neigh = get_neighbours(ch, coord, neighbour_method, max_dist, exclude=bad[i:])

        # Substitute channel (set to zero if there are no good neighbours)
        data_out[:, ch] = np.mean(data_out[:, neigh], axis=1) if neigh.size else 0

        if verbose:
            print(f"Channel {ch}: replaced by mean of channels {neigh.tolist()}")

    return data_out


def interp_bad_ch(
    data: np.ndarray,
    bad_ch: Union[list, np.ndarray],
    ch_coord: np.ndarray,
    kernel: Optional[str] = "thin_plate_spline",
    verbose: Optional[bool] = False,
    **rbf_kwargs: Any,
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
    ref: Optional[np.ndarray] = None,
    **kwargs: Any,
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
        ref (np.ndarray, optional): Data the predictions are made from, with the
            shape of `data`, e.g. `data` with its bad channels imputed. Each
            prediction is still compared with `data`. Default is None (`data`).
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
    ref = data if ref is None else ref

    # Predict each good channel from the others
    for i, ch in enumerate(good):
        pred = interp_bad_ch(ref, np.r_[bad, ch], coord, **kwargs)[:, ch]
        error[i] = np.linalg.norm(pred - data[:, ch]) / np.linalg.norm(data[:, ch])

    return error, good


# Tail of each score that marks a bad channel
_SIDES = {
    'amp': 'both',             # dead or flat (low), poor contact (high)
    'corr': 'low',             # unlike its neighbours in time
    'psd_corr': 'low',         # unlike its neighbours in spectral shape
    }

# Default Tukey fence multiplier per score, from simulated HD-EMG (see docs)
K_DEFAULT = {'amp': 3.0, 'corr': 3.0, 'psd_corr': 9.0}


def find_bad_ch(
    data: np.ndarray,
    ch_coord: np.ndarray,
    fs: Optional[int] = 2048,
    psd_band: Optional[Tuple[float, float]] = (20, 500),
    k: Optional[Dict[str, float]] = None,
    max_frac: Optional[float] = 0.25,
    max_iter: Optional[int] = 10,
    seg_s: Optional[float] = 1.0,
    neighbour_method: Optional[Literal['max_dist', 'delaunay']] = 'max_dist',
    max_dist: Optional[float] = None,
    **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[dict]]:

    """Find bad channels by iterative imputation until the bad set stops changing.

    Each iteration imputes the current bad channels with `interp_bad_ch` to get a
    reference, then tests the original signal of every positioned channel
    (including the current bad ones) against it:

    - 'amp': RMS minus the median RMS of its neighbours in the reference.
    - 'corr': median correlation with its neighbours in the reference, ignoring
      flat neighbours. The median keeps one bad neighbour from lowering it.
    - 'psd_corr': median correlation of its log PSD (Welch) in `psd_band` with
      those of its neighbours in the reference, ignoring flat neighbours.

    A channel is flagged if any score is outside its Tukey fences
    (`find_outliers` with that score's k) on its bad side: low for the
    correlations and either for 'amp'. A NaN score (e.g. the correlation of a
    flat channel) also flags it. The flagged set is the bad set of the next
    iteration.

    The leave-one-out error of each original channel predicted from the
    reference is a diagnostic only, not a score: channels at the grid edges are
    extrapolated, so their error is higher even on clean data.

    The loop stops with one of these statuses:

    - 'converged': the flagged set equals the current bad set.
    - 'cycle': the flagged set was already tested. Of the sets in the cycle, the
      one with the lowest median leave-one-out error on the channels outside all
      of them is returned.
    - 'max_frac': more than `max_frac` of the channels are flagged. The last
      accepted set is returned, with a warning.
    - 'max_iter': no convergence after `max_iter` iterations. The last flagged
      set is returned, with a warning.

    Args:
        data (np.ndarray): Filtered input data with shape (samples, chs).
        ch_coord (np.ndarray): Physical (x, y) position of each channel with
            shape (chs, 2), NaN for channels without a position (not tested).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        psd_band (Tuple[float, float], optional): Lower and upper edges in Hz
            of the PSD used for 'psd_corr', e.g. the passband of the filter.
            Default is (20, 500).
        k (Dict[str, float], optional): Fence multiplier of the IQR for
            `find_outliers` per score. Given scores override `K_DEFAULT`
            ({'amp': 3.0, 'corr': 3.0, 'psd_corr': 9.0}), e.g. {'psd_corr': 7}.
            Default is None (`K_DEFAULT`).
        max_frac (float, optional): Maximum fraction of positioned channels
            that can be bad. Default is 0.25.
        max_iter (int, optional): Maximum number of iterations. Default is 10.
        seg_s (float, optional): Length in seconds of the initial segment used
            for the leave-one-out error. Default is 1.0.
        neighbour_method (str, optional): Neighbour method used for the scores,
            passed to `get_neighbours`: 'max_dist' or 'delaunay'. Default is
            'max_dist'.
        max_dist (float, optional): Neighbour radius for 'max_dist', in the
            units of ch_coord. Default is None (ied * sqrt(2)).
        **kwargs: Passed to `interp_bad_ch`, e.g. `kernel` or `smoothing`.

    Returns:
        bad (np.ndarray): Sorted 0-based indices of the bad channels.
        scores (Dict[str, np.ndarray]): Scores of the last iteration, each with
            shape (chs,), NaN for channels without a position.
        history (List[dict]): One dict per iteration with 'bad' (tested set),
            'flagged', 'flagged_by' (flagged channels per score), 'scores', 'loo_error'
            (leave-one-out error per channel), 'dispersion' (mean across-channel
            std of the reference over the RMS of its grid mean) and
            'dev_from_mean' (per channel ||ref - grid mean|| / ||grid mean||).
            The last dict is the final result, with 'bad', 'status',
            'eval_ch' (channels never flagged), and 'loo_eval' and 'dev_eval'
            (median 'loo_error' and 'dev_from_mean' over 'eval_ch' per iteration).

    Raises:
        ValueError: If `k` has a key that isn't a score name.

    Example:
        Find the bad channels, then impute them::

            bad, scores, history = find_bad_ch(data, ch_coord, fs)
            data_clean = interp_bad_ch(data, bad, ch_coord)
    """

    # Initialise variables
    coord, _ = _check_inputs(data, (), ch_coord)
    pos = np.flatnonzero(np.isfinite(coord).all(axis=1))
    neigh = [get_neighbours(ch, coord, neighbour_method, max_dist) for ch in pos]
    k = {**K_DEFAULT, **(k or {})}
    if set(k) != set(K_DEFAULT):
        raise ValueError(f"k keys must be in {sorted(K_DEFAULT)}, got {sorted(k)}")

    bad, tested, history = frozenset(), [], []
    status = 'max_iter'
    for _ in range(max_iter):
        # Reference: current bad channels imputed
        ref = interp_bad_ch(data, sorted(bad), coord, **kwargs) if bad else data
        scores = _ref_scores(data, ref, pos, neigh, fs, psd_band)
        # Flag outliers on the bad side of each score, and NaN scores
        flagged_by = {}
        for name, score in scores.items():
            hit = np.isnan(score[pos])
            hit[find_outliers(score[pos], k[name], _SIDES[name])] = True
            flagged_by[name] = pos[hit]
        flagged = frozenset(np.concatenate(list(flagged_by.values())).tolist())
        loo_error = _loo_error(data, ref, coord, int(seg_s * fs), **kwargs)
        history.append(_diagnostics(ref, pos, bad, flagged, flagged_by, scores, loo_error))
        tested.append(bad)

        if flagged == bad:
            status = 'converged'
            break
        if flagged in tested:
            status = 'cycle'
            bad = _pick_from_cycle(tested[tested.index(flagged):],
                                   history[tested.index(flagged):], pos)
            break
        if len(flagged) > max_frac * len(pos):
            status = 'max_frac'
            break
        bad = flagged

    if status == 'max_iter':
        bad = flagged
    if status in ('max_frac', 'max_iter'):
        warnings.warn(f"find_bad_ch stopped with status {status!r}.", stacklevel=2)

    # Final result, with diagnostics on the channels never flagged
    ever = set().union(*(set(h['flagged']) | set(h['bad']) for h in history))
    eval_ch = np.setdiff1d(pos, list(ever))
    history.append({
        'bad': sorted(bad),
        'status': status,
        'eval_ch': eval_ch,
        'loo_eval': [np.median(h['loo_error'][eval_ch]) for h in history],
        'dev_eval': [np.median(h['dev_from_mean'][eval_ch]) for h in history],
        })

    return np.array(sorted(bad), dtype=int), scores, history


def _ref_scores(
    data: np.ndarray,
    ref: np.ndarray,
    pos: np.ndarray,
    neigh: List[np.ndarray],
    fs: int,
    psd_band: Tuple[float, float],
    ) -> Dict[str, np.ndarray]:
    """Scores of each original channel against its neighbours in the reference."""

    timestamps = np.arange(len(data)) / fs

    def rms(x):
        rms_win, _ = feats.get_rms(x, timestamps, fs=fs, interp=False)
        return np.median(rms_win, axis=0)

    def log_psd(x):
        psd, xf = freq.get_psd(x, fs, welch=True)
        with np.errstate(divide='ignore'):
            return np.log(psd[(xf >= psd_band[0]) & (xf <= psd_band[1])])

    rms_data, rms_ref = rms(data), rms(ref)
    amp = np.full(data.shape[1], np.nan)
    for ch, n in zip(pos, neigh, strict=True):
        amp[ch] = rms_data[ch] - np.median(rms_ref[n])

    return {
        'amp': amp,
        'corr': _neigh_corr(data, ref, pos, neigh),
        'psd_corr': _neigh_corr(log_psd(data), log_psd(ref), pos, neigh),
        }


def _neigh_corr(
    x: np.ndarray,
    x_ref: np.ndarray,
    pos: np.ndarray,
    neigh: List[np.ndarray],
    ) -> np.ndarray:
    """Median correlation of each column of x with its neighbours' columns in x_ref.

    Flat columns have NaN correlations: they are ignored as neighbours, and a
    flat channel itself gets NaN.
    """

    def standardise(a):
        with np.errstate(divide='ignore', invalid='ignore'):
            return (a - a.mean(axis=0)) / a.std(axis=0)

    z, z_ref = standardise(x), standardise(x_ref)
    corr = np.full(x.shape[1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # flat channel: all-NaN median
        for ch, n in zip(pos, neigh, strict=True):
            corr[ch] = np.nanmedian(z[:, ch] @ z_ref[:, n]) / len(x)

    return corr


def _loo_error(
    data: np.ndarray,
    ref: np.ndarray,
    coord: np.ndarray,
    seg_len: int,
    **kwargs: Any,
    ) -> np.ndarray:
    """Leave-one-out error per channel of the original signal predicted from the reference."""

    loo_error = np.full(data.shape[1], np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        error, good = get_loo_error(data[:seg_len], coord, ref=ref[:seg_len], **kwargs)
    loo_error[good] = error

    return loo_error


def _diagnostics(
    ref: np.ndarray,
    pos: np.ndarray,
    bad: frozenset,
    flagged: frozenset,
    flagged_by: Dict[str, np.ndarray],
    scores: Dict[str, np.ndarray],
    loo_error: np.ndarray,
    ) -> dict:
    """Debugging variables of one iteration of `find_bad_ch`."""

    grid_mean = ref[:, pos].mean(axis=1)
    norm_mean = np.linalg.norm(grid_mean)
    dev = np.full(ref.shape[1], np.nan)
    dev[pos] = np.linalg.norm(ref[:, pos] - grid_mean[:, None], axis=0) / norm_mean

    return {
        'bad': sorted(bad),
        'flagged': sorted(flagged),
        'flagged_by': {name: chs.tolist() for name, chs in flagged_by.items()},
        'scores': scores,
        'loo_error': loo_error,
        'dispersion': np.mean(np.std(ref[:, pos], axis=1)) / np.sqrt(np.mean(grid_mean**2)),
        'dev_from_mean': dev,
        }


def _pick_from_cycle(sets: List[frozenset], rounds: List[dict], pos: np.ndarray) -> frozenset:
    """Pick the set of a cycle with the lowest median leave-one-out error outside all sets."""

    eval_ch = np.setdiff1d(pos, list(frozenset().union(*sets)))
    loo_error = [np.median(r['loo_error'][eval_ch]) for r in rounds]

    return sets[int(np.argmin(loo_error))]


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

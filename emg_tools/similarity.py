"""Similarity between the features of EMG signals."""

import itertools
from typing import Literal, Optional, Tuple

import numpy as np
from scipy import signal


def get_similarity(
    data: np.ndarray,
    metric: Optional[Literal['corr', 'cos']] = 'corr',
    max_lag: Optional[int] = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:

    """Compute the pairwise similarity between features, optionally lag-aligned.

    For each pair of features, the lag with the largest absolute cross-correlation
    within +-max_lag is found, the signals are cropped to their overlap at that
    lag, and the similarity is computed on the aligned signals. For 'corr' the
    features are mean-removed before the lag search.

    Args:
        data (np.ndarray): Input data with shape (samples, feats).
        metric (str, optional): Similarity metric, 'corr' (Pearson correlation)
            or 'cos' (cosine similarity). Default is 'corr'.
        max_lag (int, optional): Maximum lag in samples searched for the
            alignment. Default is 0 (no alignment).

    Returns:
        sim (np.ndarray): Similarity matrix with shape (feats, feats), with ones
            on the diagonal.
        lags (np.ndarray): Lag matrix in samples with shape (feats, feats).
            lags[i, j] = k means feature i at sample n + k best matches
            feature j at sample n, i.e. feature j leads by k samples.
            lags[j, i] = -lags[i, j].
    """

    # Initialise variables
    samples, feats = data.shape
    if metric == 'corr':
        data = data - np.mean(data, axis=0)
    sim = np.eye(feats)
    lags = np.zeros((feats, feats), dtype=int)

    # Lags searched for the alignment
    all_lags = signal.correlation_lags(samples, samples)
    lag_mask = np.abs(all_lags) <= max_lag

    for i, j in itertools.combinations(range(feats), 2):
        x, y = data[:, i], data[:, j]

        # Find the lag with the largest absolute cross-correlation
        xcorr = signal.correlate(x, y)[lag_mask]
        lag = all_lags[lag_mask][np.argmax(np.abs(xcorr))]

        # Crop both signals to their overlap at that lag
        if lag >= 0:
            x, y = x[lag:], y[:samples - lag]
        else:
            x, y = x[:samples + lag], y[-lag:]

        # Compute the similarity of the aligned signals
        if metric == 'corr':
            curr_sim = np.corrcoef(x, y)[0, 1]
        elif metric == 'cos':
            curr_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        else:
            raise ValueError(f"metric must be 'corr' or 'cos', got {metric!r}")

        # Fill matrices
        sim[i, j] = sim[j, i] = curr_sim
        lags[i, j], lags[j, i] = lag, -lag

    return sim, lags


def get_coherence(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    nperseg: Optional[int] = None,
    band: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

    """Compute the pairwise magnitude-squared coherence between features.

    The coherence is estimated with Welch's method (`scipy.signal.coherence`).
    It depends on the phase relationship only through its consistency over
    segments, so no lag alignment is applied.

    Args:
        data (np.ndarray): Input data with shape (samples, feats).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        nperseg (int, optional): Length of each segment in samples. Default is
            None, which uses scipy's default (256).
        band (Tuple[float, float], optional): Lower and upper band edges in Hz,
            both included. If given, the coherence is averaged over the band.
            Default is None (full spectrum).

    Returns:
        coherence (np.ndarray): Coherence in [0, 1] with shape
            (freqs, feats, feats), or (feats, feats) if band is given.
        xf (np.ndarray): Frequency axis in Hz with shape (freqs,), restricted
            to the band if band is given.
    """

    # Compute the coherence of every pair of features at once
    xf, coherence = signal.coherence(
        data[:, :, None], data[:, None, :], fs=fs, nperseg=nperseg, axis=0
        )

    # Average over the band
    if band is not None:
        mask = (xf >= band[0]) & (xf <= band[1])
        coherence, xf = np.mean(coherence[mask], axis=0), xf[mask]

    return coherence, xf

"""Amplitude and spectral features of EMG signals."""

from typing import Optional, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import Akima1DInterpolator
from scipy.special import entr


def compute_rms(
    data: np.ndarray,
    timestamps: np.ndarray,
    win_len_s: Optional[float] = 0.1,
    win_step_s: Optional[float] = 0.1,
    fs: Optional[int] = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:

    """Compute the moving-window Root Mean Square (RMS) of the data.

    The RMS is computed per window and interpolated (Akima) back to the input
    timestamps. Before the first and after the last window centre the edge values
    are held.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        timestamps (np.ndarray): Time of each sample in seconds, with shape (samples,).
        win_len_s (float, optional): Window length in seconds. Default is 0.1.
        win_step_s (float, optional): Window step in seconds. Default is 0.1.
        fs (int, optional): Sampling frequency in Hz. Default is 2048.

    Returns:
        rms (np.ndarray): RMS interpolated to the input timestamps, with shape
            (samples, chs).
        timestamps_win (np.ndarray): Time of each window centre in seconds, with
            shape (windows,).
    """

    # Initialise variables
    samples, chs = data.shape
    win_len = min(np.round( win_len_s * fs ).astype(int), samples)
    win_step = np.round( win_step_s * fs ).astype(int)

    win_num = np.round( (samples - win_len)/win_step ).astype(int) + 1

    timestamps_aux = np.zeros(win_num)
    rms_aux = np.zeros((win_num, chs))

    # Compute RMS
    for win in range(win_num):
        mask = np.arange(
            0 + win_step * win,
            np.amin([win_len + win_step * win, samples])
            ).astype(int)
        rms_aux[win] = np.sqrt(np.mean(data[mask]**2, axis=0))
        timestamps_aux[win] = np.mean(timestamps[mask])

    # Single window, no interpolation possible
    if win_num < 2:
        rms = np.tile(rms_aux, (samples, 1))
        return rms, timestamps_aux

    # Interpolate RMS to match data signals, holding the edge values before
    # the first and after the last window centre
    timestamps_interp = np.clip(timestamps, timestamps_aux[0], timestamps_aux[-1])
    rms = np.zeros_like(data, dtype=float)
    for ch in range(chs):
        rms[:, ch] = Akima1DInterpolator(timestamps_aux, rms_aux[:, ch])(timestamps_interp)

    return rms, timestamps_aux


def get_mean_freq(psd: np.ndarray, xf: np.ndarray) -> np.ndarray:
    """Compute the mean frequency of the Power Spectral Density (PSD).

    Args:
        psd (np.ndarray): PSD with shape (freqs, chs), e.g. from `freq.get_psd`.
        xf (np.ndarray): Frequency axis in Hz, with shape (freqs,).

    Returns:
        mean_freq (np.ndarray): Mean frequency in Hz, with shape (chs,).
    """

    # Weight each frequency by its power
    mean_freq = np.sum(xf[:, None] * psd, axis=0) / np.sum(psd, axis=0)

    return mean_freq


def get_median_freq(psd: np.ndarray, xf: np.ndarray) -> np.ndarray:
    """Compute the median frequency of the Power Spectral Density (PSD).

    Args:
        psd (np.ndarray): PSD with shape (freqs, chs), e.g. from `freq.get_psd`.
        xf (np.ndarray): Frequency axis in Hz, with shape (freqs,).

    Returns:
        median_freq (np.ndarray): Median frequency in Hz, with shape (chs,).
    """

    # Find where the cumulative power crosses half of the total
    cum_power = np.cumsum(psd, axis=0)
    idx = np.argmax(cum_power >= cum_power[-1] / 2, axis=0)
    median_freq = xf[idx]

    return median_freq


def get_band_power(
    psd: np.ndarray,
    xf: np.ndarray,
    band: Tuple[float, float],
    ) -> np.ndarray:
    """Compute the power of the Power Spectral Density (PSD) within a frequency band.

    The power is the integral of the PSD over the band (trapezoidal rule), with
    both band edges included.

    Args:
        psd (np.ndarray): PSD with shape (freqs, chs), e.g. from `freq.get_psd`.
        xf (np.ndarray): Frequency axis in Hz, with shape (freqs,).
        band (Tuple[float, float]): Lower and upper band edges in Hz.

    Returns:
        band_power (np.ndarray): Power in the band with shape (chs,), in units
            of data^2.
    """

    # Select the frequencies inside the band
    mask = (xf >= band[0]) & (xf <= band[1])

    # Integrate the PSD over the band
    band_power = trapezoid(psd[mask], xf[mask], axis=0)

    return band_power


def get_spectral_entropy(
    psd: np.ndarray,
    normalise: Optional[bool] = True,
    ) -> np.ndarray:
    """Compute the spectral entropy of the Power Spectral Density (PSD).

    The PSD of each channel is normalised to a probability distribution and its
    Shannon entropy is computed. A flat spectrum has maximal entropy and a single
    tone has zero entropy.

    Args:
        psd (np.ndarray): PSD with shape (freqs, chs), e.g. from `freq.get_psd`.
        normalise (bool, optional): If True, the entropy is divided by its
            maximum, log(freqs), so it lies in [0, 1]. Default is True.

    Returns:
        entropy (np.ndarray): Spectral entropy with shape (chs,), in nats if
            not normalised.
    """

    # Normalise the PSD to a probability distribution
    prob = psd / np.sum(psd, axis=0)

    # Compute Shannon entropy (0 * log(0) = 0)
    entropy = np.sum(entr(prob), axis=0)
    if normalise:
        entropy /= np.log(psd.shape[0])

    return entropy

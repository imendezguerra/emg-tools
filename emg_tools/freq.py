"""Frequency-domain analysis of EMG signals."""

from typing import Optional, Tuple

import numpy as np
from scipy import fft, signal


def get_spectrum(
    data: np.ndarray,
    fs: Optional[int] = 2048,
) -> Tuple[np.ndarray, np.ndarray]:

    """Compute the single-sided amplitude spectrum of the input data.

    Args:
        data (np.ndarray): Input data with shape (samples, channels).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.

    Returns:
        spectrum (np.ndarray): Amplitude spectrum of the input data with shape
            (samples//2, channels), in the same units as the data, where the
            first dimension corresponds to the positive frequencies.
        xf (np.ndarray): Frequency axis in Hz, with shape (samples//2,).
    """

    # Initialise variables
    samples, chs = data.shape

    # Compute amplitude spectrum
    yf = fft.fft(data, axis=0)
    xf = fft.fftfreq(samples, 1/fs)[:samples//2]
    spectrum = 2/samples * np.abs(yf[0:samples//2])
    spectrum[0] /= 2 # DC component is not mirrored

    return spectrum, xf


def get_psd(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    *,
    welch: Optional[bool] = False,
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    """Compute the one-sided Power Spectral Density (PSD) of the input data.

    By default the periodogram is used. Welch's method can be requested instead.

    Args:
        data (np.ndarray): Input data with shape (samples, channels).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        welch (bool, optional): If True, the PSD is estimated with Welch's
            method. Default is False (periodogram).
        nperseg (int, optional): Length of each segment in samples, only used
            if welch is True. Default is None, which uses scipy's default (256).

    Returns:
        psd (np.ndarray): One-sided PSD of the input data with shape
            (freqs, channels), in units of data^2/Hz.
        xf (np.ndarray): Frequency axis in Hz, with shape (freqs,).
    """

    # Compute PSD
    if welch:
        xf, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=0)
    else:
        xf, psd = signal.periodogram(data, fs=fs, axis=0)

    return psd, xf

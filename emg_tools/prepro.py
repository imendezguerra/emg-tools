"""Filters for EMG preprocessing.

All filters work along `axis=0`, so data has shape (samples, chs) or (samples,).
"""

from typing import Optional

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


def bandpass_filter(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[list] = None,
    order: Optional[int] = 4,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """Apply a Butterworth bandpass filter.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        cutoff (list, optional): Low and high cutoff frequencies in Hz.
            Default is None, which uses [20, 500].
        order (int, optional): Filter order. Default is 4.
        filtfilt (bool, optional): If True, filter forward and backward (zero phase,
            doubled effective order). If False, filter forward only. Default is True.

    Returns:
        np.ndarray: Filtered data, same shape as the input.
    """
    # Define cutoff frequencies
    if cutoff is None:
        cutoff = [20, 500]

    # Define filter
    sos = signal.butter(order, cutoff, btype='band', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data, axis=0)
    else:
        out = signal.sosfilt(sos, data, axis=0)

    return out


def highpass_filter(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[float] = 20,
    order: Optional[int] = 2,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """Apply a Butterworth highpass filter.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        cutoff (float, optional): Cutoff frequency in Hz. Default is 20.
        order (int, optional): Filter order. Default is 2.
        filtfilt (bool, optional): If True, filter forward and backward (zero phase,
            doubled effective order). If False, filter forward only. Default is True.

    Returns:
        np.ndarray: Filtered data, same shape as the input.
    """
    # Define filter
    sos = signal.butter(order, cutoff, btype='high', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data, axis=0)
    else:
        out = signal.sosfilt(sos, data, axis=0)

    return out


def lowpass_filter(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[float] = 500,
    order: Optional[int] = 2,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """Apply a Butterworth lowpass filter.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        cutoff (float, optional): Cutoff frequency in Hz. Default is 500.
        order (int, optional): Filter order. Default is 2.
        filtfilt (bool, optional): If True, filter forward and backward (zero phase,
            doubled effective order). If False, filter forward only. Default is True.

    Returns:
        np.ndarray: Filtered data, same shape as the input.
    """
    # Define filter
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data, axis=0)
    else:
        out = signal.sosfilt(sos, data, axis=0)

    return out


def remove_powerline(
    data: np.ndarray,
    fs: Optional[int] = 2048,
    cutoff: Optional[float] = 50,
    width: Optional[float] = 1,
    order: Optional[int] = 2,
    n_harmonics: Optional[int] = 1,
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """Remove powerline noise and its harmonics with Butterworth bandstop filters.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        cutoff (float, optional): Powerline frequency in Hz. Default is 50.
        width (float, optional): Width of each stop band in Hz, centred on each
            harmonic. Default is 1.
        order (int, optional): Filter order of each stop band. Default is 2.
        n_harmonics (int, optional): Number of harmonics to remove, including the
            fundamental (e.g. 3 removes 50, 100 and 150 Hz). Harmonics whose stop band
            reaches the Nyquist frequency are skipped. Default is 1.
        filtfilt (bool, optional): If True, filter forward and backward (zero phase,
            doubled effective order). If False, filter forward only. Default is True.

    Returns:
        np.ndarray: Filtered data, same shape as the input.
    """

    # Harmonics below Nyquist
    harmonics = cutoff * np.arange(1, n_harmonics + 1)
    harmonics = harmonics[harmonics + width/2 < fs/2]

    # Define filter: one bandstop per harmonic, cascaded into a single sos
    sos = np.vstack([
        signal.butter(order, [h - width/2, h + width/2], btype='bandstop', fs=fs, output='sos')
        for h in harmonics
    ])

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data, axis=0)
    else:
        out = signal.sosfilt(sos, data, axis=0)

    return out


def remove_drift(
    data: np.ndarray,
    window: int,
    ) -> np.ndarray:

    """Remove slow drift by subtracting a centred moving average.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        window (int): Moving average length in samples. Drift slower than about
            fs / window Hz is removed.

    Returns:
        np.ndarray: Data without drift, same shape as the input.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    data = np.asarray(data, dtype=float)
    return data - uniform_filter1d(data, size=window, axis=0, mode='nearest')

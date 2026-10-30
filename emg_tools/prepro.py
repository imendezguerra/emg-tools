"""Butterworth filters for EMG preprocessing.

All filters work along `axis=0`, so data has shape (samples, chs) or (samples,).
"""

from typing import Optional

import numpy as np
from scipy import signal


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
    filtfilt: Optional[bool] = True,
    ) -> np.ndarray:

    """Remove powerline noise with a Butterworth bandstop filter.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        cutoff (float, optional): Powerline frequency in Hz. Default is 50.
        width (float, optional): Width of the stop band in Hz, centred on cutoff.
            Default is 1.
        order (int, optional): Filter order. Default is 2.
        filtfilt (bool, optional): If True, filter forward and backward (zero phase,
            doubled effective order). If False, filter forward only. Default is True.

    Returns:
        np.ndarray: Filtered data, same shape as the input.
    """

    # Build cutoff
    cutoff = [cutoff - width/2, cutoff + width/2]

    # Define filter
    sos = signal.butter(order, cutoff, btype='bandstop', fs=fs, output='sos')

    # Apply filter
    if filtfilt:
        out = signal.sosfiltfilt(sos, data, axis=0)
    else:
        out = signal.sosfilt(sos, data, axis=0)

    return out

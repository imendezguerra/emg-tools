"""Amplitude, spatial and spectral features of EMG signals."""

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import Akima1DInterpolator
from scipy.special import entr


def _windowed(
    data: np.ndarray,
    timestamps: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
    win_len_s: float,
    win_step_s: float,
    fs: int,
    interp: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a feature function in moving windows.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        timestamps (np.ndarray): Time of each sample in seconds, with shape (samples,).
        func (Callable): Maps a (win_samples, chs) window to a (chs,) feature.
        win_len_s (float): Window length in seconds.
        win_step_s (float): Window step in seconds.
        fs (int): Sampling frequency in Hz.
        interp (bool): If True, interpolate (Akima) the window values back to
            the input timestamps, holding the edge values before the first and
            after the last window centre.

    Returns:
        feat (np.ndarray): Feature with shape (samples, chs) if `interp`, else
            (windows, chs).
        timestamps_win (np.ndarray): Time of each window centre in seconds, with
            shape (windows,).
    """

    # Initialise variables
    samples, chs = data.shape
    win_len = min(np.round( win_len_s * fs ).astype(int), samples)
    win_step = np.round( win_step_s * fs ).astype(int)

    # Full-length windows only, so sums (WL, ZC, SSC) are comparable
    win_num = (samples - win_len) // win_step + 1

    timestamps_win = np.zeros(win_num)
    feat_win = np.zeros((win_num, chs))

    # Compute the feature per window
    for win in range(win_num):
        start = win_step * win
        stop = start + win_len
        feat_win[win] = func(data[start:stop])
        timestamps_win[win] = np.mean(timestamps[start:stop])

    if not interp:
        return feat_win, timestamps_win

    # Single window, no interpolation possible
    if win_num < 2:
        return np.tile(feat_win, (samples, 1)), timestamps_win

    timestamps_interp = np.clip(timestamps, timestamps_win[0], timestamps_win[-1])
    feat = np.zeros_like(data, dtype=float)
    for ch in range(chs):
        feat[:, ch] = Akima1DInterpolator(timestamps_win, feat_win[:, ch])(timestamps_interp)

    return feat, timestamps_win


def compute_rms(
    data: np.ndarray,
    timestamps: np.ndarray,
    win_len_s: Optional[float] = 0.1,
    win_step_s: Optional[float] = 0.1,
    fs: Optional[int] = 2048,
    interp: Optional[bool] = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the moving-window Root Mean Square (RMS) of the data.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        timestamps (np.ndarray): Time of each sample in seconds, with shape (samples,).
        win_len_s (float, optional): Window length in seconds. Default is 0.1.
        win_step_s (float, optional): Window step in seconds. Default is 0.1.
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        interp (bool, optional): If True, interpolate the RMS back to the input
            timestamps (samples, chs), else return one value per window
            (windows, chs). Default is True.

    Returns:
        rms (np.ndarray): RMS with shape (samples, chs) or (windows, chs).
        timestamps_win (np.ndarray): Time of each window centre in seconds, with
            shape (windows,).
    """

    def rms(x):
        return np.sqrt(np.mean(x**2, axis=0))

    return _windowed(data, timestamps, rms, win_len_s, win_step_s, fs, interp)


def compute_mav(
    data: np.ndarray,
    timestamps: np.ndarray,
    win_len_s: Optional[float] = 0.1,
    win_step_s: Optional[float] = 0.1,
    fs: Optional[int] = 2048,
    interp: Optional[bool] = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the moving-window Mean Absolute Value (MAV) of the data.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        timestamps (np.ndarray): Time of each sample in seconds, with shape (samples,).
        win_len_s (float, optional): Window length in seconds. Default is 0.1.
        win_step_s (float, optional): Window step in seconds. Default is 0.1.
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        interp (bool, optional): If True, interpolate back to the input
            timestamps (samples, chs), else return (windows, chs). Default is True.

    Returns:
        mav (np.ndarray): MAV with shape (samples, chs) or (windows, chs).
        timestamps_win (np.ndarray): Time of each window centre in seconds, with
            shape (windows,).
    """

    def mav(x):
        return np.mean(np.abs(x), axis=0)

    return _windowed(data, timestamps, mav, win_len_s, win_step_s, fs, interp)


def compute_wl(
    data: np.ndarray,
    timestamps: np.ndarray,
    win_len_s: Optional[float] = 0.1,
    win_step_s: Optional[float] = 0.1,
    fs: Optional[int] = 2048,
    interp: Optional[bool] = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the moving-window Waveform Length (WL) of the data.

    WL is the sum of absolute sample-to-sample differences in each window, so it
    scales with the window length.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        timestamps (np.ndarray): Time of each sample in seconds, with shape (samples,).
        win_len_s (float, optional): Window length in seconds. Default is 0.1.
        win_step_s (float, optional): Window step in seconds. Default is 0.1.
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        interp (bool, optional): If True, interpolate back to the input
            timestamps (samples, chs), else return (windows, chs). Default is True.

    Returns:
        wl (np.ndarray): WL with shape (samples, chs) or (windows, chs).
        timestamps_win (np.ndarray): Time of each window centre in seconds, with
            shape (windows,).
    """

    def wl(x):
        return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

    return _windowed(data, timestamps, wl, win_len_s, win_step_s, fs, interp)


def compute_zc(
    data: np.ndarray,
    timestamps: np.ndarray,
    threshold: Optional[float] = 0.0,
    win_len_s: Optional[float] = 0.1,
    win_step_s: Optional[float] = 0.1,
    fs: Optional[int] = 2048,
    interp: Optional[bool] = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the moving-window number of Zero Crossings (ZC) of the data.

    A crossing counts when consecutive samples change sign and their absolute
    difference is at least `threshold`, which rejects noise around zero.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        timestamps (np.ndarray): Time of each sample in seconds, with shape (samples,).
        threshold (float, optional): Minimum amplitude step, in data units.
            Default is 0.
        win_len_s (float, optional): Window length in seconds. Default is 0.1.
        win_step_s (float, optional): Window step in seconds. Default is 0.1.
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        interp (bool, optional): If True, interpolate back to the input
            timestamps (samples, chs), else return (windows, chs). Default is True.

    Returns:
        zc (np.ndarray): ZC count with shape (samples, chs) or (windows, chs).
        timestamps_win (np.ndarray): Time of each window centre in seconds, with
            shape (windows,).
    """

    def zc(x):
        cross = (x[:-1] * x[1:] < 0) & (np.abs(np.diff(x, axis=0)) >= threshold)
        return np.sum(cross, axis=0)

    return _windowed(data, timestamps, zc, win_len_s, win_step_s, fs, interp)


def compute_ssc(
    data: np.ndarray,
    timestamps: np.ndarray,
    threshold: Optional[float] = 0.0,
    win_len_s: Optional[float] = 0.1,
    win_step_s: Optional[float] = 0.1,
    fs: Optional[int] = 2048,
    interp: Optional[bool] = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the moving-window number of Slope Sign Changes (SSC) of the data.

    A change counts at a local peak or trough when at least one of its two
    slopes is at least `threshold` in absolute value.

    Args:
        data (np.ndarray): Input data with shape (samples, chs).
        timestamps (np.ndarray): Time of each sample in seconds, with shape (samples,).
        threshold (float, optional): Minimum slope, in data units per sample.
            Default is 0.
        win_len_s (float, optional): Window length in seconds. Default is 0.1.
        win_step_s (float, optional): Window step in seconds. Default is 0.1.
        fs (int, optional): Sampling frequency in Hz. Default is 2048.
        interp (bool, optional): If True, interpolate back to the input
            timestamps (samples, chs), else return (windows, chs). Default is True.

    Returns:
        ssc (np.ndarray): SSC count with shape (samples, chs) or (windows, chs).
        timestamps_win (np.ndarray): Time of each window centre in seconds, with
            shape (windows,).
    """

    def ssc(x):
        d = np.diff(x, axis=0)
        turn = d[:-1] * d[1:] < 0
        steep = np.maximum(np.abs(d[:-1]), np.abs(d[1:])) >= threshold
        return np.sum(turn & steep, axis=0)

    return _windowed(data, timestamps, ssc, win_len_s, win_step_s, fs, interp)


def _mapped(data: np.ndarray, ch_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the channels with a grid position (non-NaN coordinates)."""
    on_grid = ~np.isnan(ch_coord).any(axis=1)
    return data[:, on_grid], ch_coord[on_grid]


def get_center_of_gravity(data: np.ndarray, ch_coord: np.ndarray) -> np.ndarray:
    """Compute the amplitude-weighted centre of gravity of the channels.

    Args:
        data (np.ndarray): Non-negative amplitude (e.g. RMS) with shape
            (samples, chs) or (windows, chs).
        ch_coord (np.ndarray): Channel (x, y) coordinates with shape (chs, 2),
            e.g. from `tools.ch_map_to_coord`. NaN channels are ignored.

    Returns:
        cog (np.ndarray): Centre of gravity (x, y) with shape (samples, 2), in
            the units of `ch_coord`.
    """

    amp, coord = _mapped(data, ch_coord)
    cog = amp @ coord / np.sum(amp, axis=1, keepdims=True)

    return cog


def get_spatial_entropy(
    data: np.ndarray,
    ch_coord: np.ndarray,
    normalise: Optional[bool] = True,
    ) -> np.ndarray:
    """Compute the spatial entropy of the amplitude distribution across channels.

    The squared amplitude of the channels is normalised to a probability
    distribution and its Shannon entropy is computed. A uniform map has maximal
    entropy and activity on a single channel has zero entropy.

    Args:
        data (np.ndarray): Amplitude (e.g. RMS) with shape (samples, chs) or
            (windows, chs).
        ch_coord (np.ndarray): Channel (x, y) coordinates with shape (chs, 2),
            e.g. from `tools.ch_map_to_coord`. NaN channels are ignored.
        normalise (bool, optional): If True, the entropy is divided by its
            maximum, log(chs), so it lies in [0, 1]. Default is True.

    Returns:
        entropy (np.ndarray): Spatial entropy with shape (samples,), in nats if
            not normalised.
    """

    amp, _ = _mapped(data, ch_coord)
    power = amp**2
    prob = power / np.sum(power, axis=1, keepdims=True)

    # Compute Shannon entropy (0 * log(0) = 0)
    entropy = np.sum(entr(prob), axis=1)
    if normalise:
        entropy /= np.log(amp.shape[1])

    return entropy


def get_spatial_cv(data: np.ndarray, ch_coord: np.ndarray) -> np.ndarray:
    """Compute the coefficient of variation (std / mean) of the amplitude across channels.

    Args:
        data (np.ndarray): Amplitude (e.g. RMS) with shape (samples, chs) or
            (windows, chs).
        ch_coord (np.ndarray): Channel (x, y) coordinates with shape (chs, 2),
            e.g. from `tools.ch_map_to_coord`. NaN channels are ignored.

    Returns:
        cv (np.ndarray): Coefficient of variation with shape (samples,).
    """

    amp, _ = _mapped(data, ch_coord)
    cv = np.std(amp, axis=1) / np.mean(amp, axis=1)

    return cv


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

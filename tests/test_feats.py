import numpy as np
import pytest

from emg_tools import feats, freq, tools


def test_compute_rms_of_sine(make_tone, timestamps, fs):
    data = np.stack([make_tone(100, amp=1), make_tone(100, amp=3)], axis=-1)
    rms, win_ts = feats.compute_rms(data, timestamps, fs=fs)

    assert rms.shape == data.shape
    np.testing.assert_allclose(rms, [[1 / np.sqrt(2), 3 / np.sqrt(2)]] * len(data), rtol=1e-2)
    # Window timestamps are the window centres
    np.testing.assert_allclose(win_ts[:3], [0.05, 0.15, 0.25], atol=1e-3)


@pytest.mark.parametrize("step_s", [0.2, 1.0, 1.8])
def test_compute_rms_is_time_aligned(timestamps, fs, step_s):
    data = np.zeros((len(timestamps), 1))
    data[int(step_s * fs):] = 1
    rms, _ = feats.compute_rms(data, timestamps, fs=fs)

    crossing = timestamps[np.argmax(rms[:, 0] > 0.5)]
    assert crossing == pytest.approx(step_s, abs=0.01)


def test_compute_rms_shorter_than_window(rng, fs):
    data = rng.standard_normal((50, 2))
    rms, win_ts = feats.compute_rms(data, np.arange(50) / fs, fs=fs)

    assert win_ts.shape == (1,)
    np.testing.assert_allclose(rms, np.tile(np.sqrt(np.mean(data**2, axis=0)), (50, 1)))


@pytest.mark.parametrize("func", [feats.compute_rms, feats.compute_mav, feats.compute_wl,
                                  feats.compute_zc, feats.compute_ssc])
def test_windowed_output_shapes(func, emg, timestamps, fs):
    out, win_ts = func(emg, timestamps, fs=fs)
    out_win, win_ts_win = func(emg, timestamps, fs=fs, interp=False)

    assert out.shape == emg.shape
    assert out_win.shape == (len(win_ts), emg.shape[1])
    np.testing.assert_array_equal(win_ts, win_ts_win)


def test_compute_mav_and_wl_of_sine(make_tone, timestamps, fs):
    data = np.stack([make_tone(100, amp=1), make_tone(100, amp=3)], axis=-1)
    mav, _ = feats.compute_mav(data, timestamps, fs=fs, interp=False)
    wl, _ = feats.compute_wl(data, timestamps, fs=fs, interp=False)

    # MAV of a sine is 2 * amp / pi, WL is 4 * amp per period (10 per 0.1 s window)
    np.testing.assert_allclose(mav, [[2 / np.pi, 6 / np.pi]] * len(mav), rtol=1e-2)
    np.testing.assert_allclose(wl, [[40, 120]] * len(wl), rtol=2e-2)


def test_compute_zc_and_ssc_of_sine(make_tone, timestamps, fs):
    data = make_tone(100.3, amp=1)[:, None]   # off-grid frequency avoids exact zeros
    zc, _ = feats.compute_zc(data, timestamps, fs=fs, interp=False)
    ssc, _ = feats.compute_ssc(data, timestamps, fs=fs, interp=False)

    # Two crossings and two turning points per period, ~10 periods per window
    np.testing.assert_allclose(zc, 20, atol=1)
    np.testing.assert_allclose(ssc, 20, atol=1)


def test_compute_zc_and_ssc_threshold(rng, timestamps, fs):
    data = 0.01 * rng.standard_normal((len(timestamps), 1))   # baseline noise
    zc_all, _ = feats.compute_zc(data, timestamps, fs=fs, interp=False)
    zc_thr, _ = feats.compute_zc(data, timestamps, threshold=0.1, fs=fs, interp=False)
    ssc_all, _ = feats.compute_ssc(data, timestamps, fs=fs, interp=False)
    ssc_thr, _ = feats.compute_ssc(data, timestamps, threshold=0.1, fs=fs, interp=False)

    # Noise gives many counts, a threshold above the noise removes them
    assert zc_all.min() > 50
    assert ssc_all.min() > 50
    np.testing.assert_array_equal(zc_thr, 0)
    np.testing.assert_array_equal(ssc_thr, 0)


def test_get_center_of_gravity(ch_map):
    ch_coord = tools.ch_map_to_coord(ch_map, ied=8)
    amp = np.ones((2, ch_map.size))
    amp[1] = 0
    amp[1, ch_map[3, 1]] = 1   # all activity on one electrode

    np.testing.assert_allclose(feats.get_center_of_gravity(amp, ch_coord), [[12, 12], [8, 24]])


def test_spatial_features_ignore_unmapped_channels(ch_map):
    ch_coord = tools.ch_map_to_coord(ch_map)
    ch_coord = np.vstack([ch_coord, [np.nan, np.nan]])   # extra channel off the grid
    amp = np.ones((1, ch_map.size + 1))
    amp[0, -1] = 100

    np.testing.assert_allclose(feats.get_center_of_gravity(amp, ch_coord), [[1.5, 1.5]])
    np.testing.assert_allclose(feats.get_spatial_entropy(amp, ch_coord), [1])
    np.testing.assert_allclose(feats.get_spatial_cv(amp, ch_coord), [0], atol=1e-12)


def test_get_spatial_entropy_and_cv(ch_map):
    ch_coord = tools.ch_map_to_coord(ch_map)
    chs = ch_map.size
    amp = np.ones((2, chs))
    amp[1] = 0
    amp[1, 0] = 1   # single active channel

    entropy = feats.get_spatial_entropy(amp, ch_coord)
    np.testing.assert_allclose(entropy, [1, 0], atol=1e-12)
    np.testing.assert_allclose(
        feats.get_spatial_entropy(amp, ch_coord, normalise=False), entropy * np.log(chs)
    )
    np.testing.assert_allclose(feats.get_spatial_cv(amp, ch_coord), [0, np.sqrt(chs - 1)])


def test_get_mean_and_median_freq_of_tones(make_tone, fs):
    data = np.stack([make_tone(100), make_tone(250)], axis=-1)
    psd, xf = freq.get_psd(data, fs)

    np.testing.assert_allclose(feats.get_mean_freq(psd, xf), [100, 250], atol=1)
    np.testing.assert_allclose(feats.get_median_freq(psd, xf), [100, 250], atol=1)


def test_get_median_freq_splits_power(rng, fs):
    psd, xf = freq.get_psd(rng.standard_normal((4096, 2)), fs)
    median_freq = feats.get_median_freq(psd, xf)

    for ch in range(2):
        below = psd[xf <= median_freq[ch], ch].sum()
        assert below / psd[:, ch].sum() == pytest.approx(0.5, abs=0.01)


def test_get_band_power(make_tone, fs):
    data = make_tone(100, amp=2)[:, None]
    psd, xf = freq.get_psd(data, fs)

    # Tone power is amp^2 / 2, all of it inside the band and none outside
    np.testing.assert_allclose(feats.get_band_power(psd, xf, (50, 150)), [2], rtol=1e-2)
    np.testing.assert_allclose(feats.get_band_power(psd, xf, (200, 400)), [0], atol=1e-6)


def test_get_spectral_entropy(make_tone, rng, fs):
    data = np.stack([make_tone(100), rng.standard_normal(4096)], axis=-1)
    psd, _ = freq.get_psd(data, fs)
    entropy = feats.get_spectral_entropy(psd)

    # Single tone is close to 0, white noise is close to 1
    assert entropy[0] < 0.05
    assert entropy[1] > 0.9
    np.testing.assert_allclose(
        feats.get_spectral_entropy(psd, normalise=False), entropy * np.log(len(psd))
    )

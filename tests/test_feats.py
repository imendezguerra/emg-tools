import numpy as np
import pytest

from emg_tools import feats, freq


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

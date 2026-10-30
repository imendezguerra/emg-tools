import numpy as np
import pytest

from emg_tools import feats


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

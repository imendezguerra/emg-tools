import numpy as np
import pytest

from emg_tools import prepro

N_SAMPLES = 6 * 2048
EDGE = 2 * 2048  # the 1 Hz-wide powerline notch takes ~2 s to settle


def rms(x):
    """RMS per channel away from the filter edge transients."""
    return np.sqrt(np.mean(x[EDGE:-EDGE] ** 2, axis=0))


@pytest.mark.parametrize(
    ("filt", "kwargs", "pass_hz", "stop_hz"),
    [
        (prepro.bandpass_filter, {}, 100, 5),
        (prepro.bandpass_filter, {}, 100, 900),
        (prepro.highpass_filter, {}, 100, 2),
        (prepro.lowpass_filter, {}, 100, 900),
        (prepro.remove_powerline, {}, 100, 50),
        (prepro.remove_powerline, {"cutoff": 60}, 100, 60),
    ],
)
@pytest.mark.parametrize("filtfilt", [True, False])
def test_filters_along_time(make_tone, filt, kwargs, pass_hz, stop_hz, filtfilt):
    # One tone per channel: filtering across channels would mix or crash
    data = np.stack([make_tone(pass_hz, n_samples=N_SAMPLES), make_tone(stop_hz, n_samples=N_SAMPLES)], axis=-1)
    out = filt(data, filtfilt=filtfilt, **kwargs)

    assert out.shape == data.shape
    assert rms(out[:, 0]) == pytest.approx(rms(data[:, 0]), rel=0.1)
    assert rms(out[:, 1]) < 0.1 * rms(data[:, 1])


def test_filter_does_not_modify_input(emg):
    original = emg.copy()
    prepro.bandpass_filter(emg)
    np.testing.assert_array_equal(emg, original)

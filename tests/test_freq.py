import numpy as np
import pytest

from emg_tools import freq


def test_get_spectrum_amplitude(make_tone, fs):
    data = np.stack([make_tone(100, amp=2) + 0.5, make_tone(250, amp=1)], axis=-1)
    spectrum, xf = freq.get_spectrum(data, fs)

    assert spectrum.shape == (len(data) // 2, 2)
    assert xf.shape == (len(data) // 2,)
    assert xf[spectrum.argmax(axis=0)].tolist() == [100, 250]
    np.testing.assert_allclose(spectrum.max(axis=0), [2, 1], rtol=1e-6)
    assert spectrum[0, 0] == pytest.approx(0.5)  # DC offset


@pytest.mark.parametrize("welch", [False, True])
def test_get_psd_parseval(make_tone, emg, fs, welch):
    data = np.column_stack([make_tone(100, amp=2), emg[:, 0]])
    psd, xf = freq.get_psd(data, fs, welch=welch)

    assert psd.shape == (len(xf), 2)
    assert xf[psd[:, 0].argmax()] == pytest.approx(100, abs=fs / 256)
    # Integral of the PSD equals the signal power
    power = np.sum(psd, axis=0) * (xf[1] - xf[0])
    np.testing.assert_allclose(power, data.var(axis=0), rtol=0.05)


def test_get_psd_welch_nperseg(emg, fs):
    psd, xf = freq.get_psd(emg, fs, welch=True, nperseg=512)
    assert xf.shape == (257,)
    assert psd.shape == (257, emg.shape[1])

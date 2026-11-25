import numpy as np
import pytest

from emg_tools import similarity


@pytest.mark.parametrize("metric", ["corr", "cos"])
def test_get_similarity_recovers_lag(rng, metric):
    x = rng.standard_normal(2000)
    data = np.stack([x[20:1020], x[:1000], -x[20:1020]], axis=-1)
    sim, lags = similarity.get_similarity(data, metric=metric, max_lag=50)

    # Feature 1 is feature 0 delayed by 20 samples, feature 2 is feature 0 inverted
    assert lags[0, 1] == -20
    assert lags[1, 0] == 20
    assert lags[0, 2] == 0
    np.testing.assert_allclose(sim[[0, 0], [1, 2]], [1, -1], atol=1e-6)
    np.testing.assert_allclose(sim, sim.T)
    np.testing.assert_allclose(np.diag(sim), 1)


def test_get_similarity_without_alignment(rng):
    x = rng.standard_normal(2000)
    data = np.stack([x[20:1020], x[:1000]], axis=-1)
    sim, lags = similarity.get_similarity(data)

    assert np.all(lags == 0)
    assert abs(sim[0, 1]) < 0.2


def test_get_similarity_invalid_metric(rng):
    with pytest.raises(ValueError, match="metric must be"):
        similarity.get_similarity(rng.standard_normal((100, 2)), metric="l2")


def test_get_coherence(make_tone, rng, fs):
    noise = 0.1 * rng.standard_normal((4096, 3))
    tone = make_tone(100)
    data = np.stack([tone, np.roll(tone, 5), rng.standard_normal(4096)], axis=-1) + noise
    coherence, xf = similarity.get_coherence(data, fs)

    assert coherence.shape == (len(xf), 3, 3)
    np.testing.assert_allclose(coherence, coherence.transpose(0, 2, 1))
    np.testing.assert_allclose(coherence[:, [0, 1, 2], [0, 1, 2]], 1)

    # Shifted tones are coherent at 100 Hz, independent noise is not
    band_coh, band_xf = similarity.get_coherence(data, fs, band=(90, 110))
    assert band_coh.shape == (3, 3)
    assert band_xf.min() >= 90
    assert band_xf.max() <= 110
    assert band_coh[0, 1] > 0.9
    assert band_coh[0, 2] < 0.5

import numpy as np
import pytest

from emg_tools import imputation, prepro, tools


def test_find_outliers():
    data = np.array([10.0, 1, 2, 3, 4, 5, -8, 3, 2])
    np.testing.assert_array_equal(imputation.find_outliers(data), [0, 6])


def test_find_outliers_k_widens_fences():
    data = np.array([10.0, 1, 2, 3, 4, 5, -8, 3, 2])
    assert imputation.find_outliers(data, k=10).size == 0


def test_find_outliers_none():
    assert imputation.find_outliers(np.arange(10.0)).size == 0


@pytest.fixture
def ch_coord(ch_map):
    """(chs, 2) positions of the 4x4 grid with an 8 mm inter-electrode distance."""
    return tools.ch_map_to_coord(ch_map, ied=8)


@pytest.mark.parametrize(
    ("ch", "kwargs", "neigh"),
    [
        (5, {}, [0, 1, 2, 4, 6, 8, 9, 10]),  # inner channel: 8 surrounding
        (0, {}, [1, 4, 5]),  # corner
        (5, {"max_dist": 8}, [1, 4, 6, 9]),  # one ied: side neighbours only
        (5, {"exclude": [0, 6]}, [1, 2, 4, 8, 9, 10]),
    ],
)
def test_get_neighbours_grid(ch_coord, ch, kwargs, neigh):
    np.testing.assert_array_equal(imputation.get_neighbours(ch, ch_coord, **kwargs), neigh)


def test_get_neighbours_delaunay_irregular():
    # Channel 0 sits in the middle of a triangle, channel 4 is far outside it
    coord = np.array([[0.0, 0.0], [-1, -1], [1, -1], [0, 1.5], [10, 10]])
    np.testing.assert_array_equal(
        imputation.get_neighbours(0, coord, method="delaunay"), [1, 2, 3]
    )


def test_get_neighbours_delaunay_collinear_uses_2_nearest():
    coord = np.c_[np.arange(5.0), np.zeros(5)]
    np.testing.assert_array_equal(
        imputation.get_neighbours(2, coord, method="delaunay"), [1, 3]
    )


def test_get_neighbours_skips_unpositioned_channels(ch_coord):
    coord = np.where(np.arange(16)[:, None] == 6, np.nan, ch_coord)
    np.testing.assert_array_equal(
        imputation.get_neighbours(5, coord), [0, 1, 2, 4, 8, 9, 10]
    )


def test_get_neighbours_input_checks(ch_coord):
    with pytest.raises(ValueError, match="method must be"):
        imputation.get_neighbours(5, ch_coord, method="nearest")
    with pytest.raises(ValueError, match="needs a finite position"):
        imputation.get_neighbours(0, np.where(np.arange(16)[:, None] == 0, np.nan, ch_coord))


# Channel 0 is a corner with neighbours 1, 4 and 5. Bad channels are replaced in
# order, and channels still pending replacement are not used as neighbours.
@pytest.mark.parametrize(
    ("bad_ch", "neigh"),
    [([0], [1, 4, 5]), ([5, 0], [1, 4, 5]), (np.array([0, 5]), [1, 4])],
)
def test_replace_bad_ch_mean_of_neighbours(emg, ch_coord, bad_ch, neigh):
    original = emg.copy()
    out = imputation.replace_bad_ch(emg, bad_ch, ch_coord)

    np.testing.assert_allclose(out[:, 0], out[:, neigh].mean(axis=-1))
    np.testing.assert_array_equal(emg, original)  # input not modified


def test_replace_bad_ch_inner_channel_uses_8_neighbours(emg, ch_coord):
    out = imputation.replace_bad_ch(emg, [5], ch_coord)
    neigh = [0, 1, 2, 4, 6, 8, 9, 10]
    np.testing.assert_allclose(out[:, 5], emg[:, neigh].mean(axis=-1))


def test_replace_bad_ch_max_dist(emg, ch_coord):
    # A radius of one ied keeps only the 4 side neighbours
    out = imputation.replace_bad_ch(emg, [5], ch_coord, max_dist=8)
    np.testing.assert_allclose(out[:, 5], emg[:, [1, 4, 6, 9]].mean(axis=-1))


def test_replace_bad_ch_delaunay_irregular(rng):
    # Channel 0 sits in the middle of a triangle, channel 4 is far outside it
    coord = np.array([[0.0, 0.0], [-1, -1], [1, -1], [0, 1.5], [10, 10]])
    data = rng.standard_normal((50, 5))
    out = imputation.replace_bad_ch(data, [0], coord, neighbour_method="delaunay")
    np.testing.assert_allclose(out[:, 0], data[:, [1, 2, 3]].mean(axis=-1))


def test_replace_bad_ch_delaunay_collinear_uses_2_nearest(rng):
    coord = np.c_[np.arange(5.0), np.zeros(5)]
    data = rng.standard_normal((50, 5))
    out = imputation.replace_bad_ch(data, [2], coord, neighbour_method="delaunay")
    np.testing.assert_allclose(out[:, 2], data[:, [1, 3]].mean(axis=-1))


def test_replace_bad_ch_without_good_neighbours_is_zero():
    coord = np.array([[0.0, 0.0], [8, 0]])
    out = imputation.replace_bad_ch(np.ones((5, 2)), [0, 1], coord, max_dist=8)
    np.testing.assert_array_equal(out, 0)


def test_replace_bad_ch_skips_unpositioned_channels(emg, ch_coord):
    # Channel 16 is in the data but has no position
    data = np.c_[emg, 100 * np.ones(len(emg))]
    coord = np.r_[ch_coord, [[np.nan, np.nan]]]
    out = imputation.replace_bad_ch(data, [5], coord)
    np.testing.assert_allclose(out[:, 5], emg[:, [0, 1, 2, 4, 6, 8, 9, 10]].mean(axis=-1))


@pytest.mark.parametrize("func", [imputation.replace_bad_ch, imputation.interp_bad_ch])
def test_bad_ch_input_checks(emg, ch_coord, func):
    with pytest.raises(ValueError, match="Expected data"):
        func(emg, [0], ch_coord[:-1])  # one coordinate missing
    with pytest.raises(ValueError, match="needs a finite position"):
        func(emg, [0], np.where(np.arange(16)[:, None] == 0, np.nan, ch_coord))


def test_replace_bad_ch_invalid_method(emg, ch_coord):
    with pytest.raises(ValueError, match="method must be"):
        imputation.replace_bad_ch(emg, [0], ch_coord, neighbour_method="nearest")


def test_interp_bad_ch_recovers_smooth_field(ch_coord):
    # A linear field over the grid is reproduced exactly by a thin-plate spline
    field = ch_coord @ np.array([0.3, -0.2]) + 1
    data = np.outer(np.sin(np.arange(100) / 5), field)
    corrupted = data.copy()
    corrupted[:, [5, 10]] = 0

    out = imputation.interp_bad_ch(corrupted, [5, 10], ch_coord)
    np.testing.assert_allclose(out, data, atol=1e-8)


def test_interp_bad_ch_collinear_raises(rng):
    coord = np.c_[np.arange(5.0), np.zeros(5)]
    with pytest.raises(ValueError, match="single line"):
        imputation.interp_bad_ch(rng.standard_normal((50, 5)), [2], coord)


def test_get_loo_error(emg, ch_coord):
    # Smooth fields are predictable from their surroundings, white noise is not
    smooth = np.outer(np.sin(np.arange(100) / 5), ch_coord @ np.array([0.3, -0.2]) + 1)
    error, good = imputation.get_loo_error(smooth, ch_coord, bad_ch=[5])

    np.testing.assert_array_equal(good, np.setdiff1d(np.arange(16), [5]))
    assert error.max() < 1e-6

    noise_error, _ = imputation.get_loo_error(emg[:200], ch_coord)
    assert noise_error.min() > 0.5


def test_find_outliers_side_and_nan():
    data = np.array([10.0, 1, 2, 3, np.nan, 4, 5, -8, 3, 2])
    np.testing.assert_array_equal(imputation.find_outliers(data), [0, 7])
    np.testing.assert_array_equal(imputation.find_outliers(data, side='low'), [7])
    np.testing.assert_array_equal(imputation.find_outliers(data, side='high'), [0])


def test_get_loo_error_with_ref(emg, ch_coord):
    clean = emg[:512]
    error, good = imputation.get_loo_error(clean, ch_coord)
    error_ref, good_ref = imputation.get_loo_error(clean, ch_coord, ref=clean)
    np.testing.assert_allclose(error, error_ref)
    np.testing.assert_array_equal(good, good_ref)

    # Predicted from a clean reference, a corrupted channel doesn't affect the others
    data = clean.copy()
    data[:, 5] *= 10
    error_bad, _ = imputation.get_loo_error(data, ch_coord, ref=clean)
    error_no_ref, _ = imputation.get_loo_error(data, ch_coord)
    others = good != 5
    np.testing.assert_allclose(error_bad[others], error[others])
    assert not np.allclose(error_no_ref[others], error[others])


@pytest.fixture
def grid8():
    """Shared-source EMG on an 8x8 grid, with a contraction envelope, and its coordinates."""
    fs, n = 2048, 5 * 2048
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    env = np.sin(np.pi * t / t[-1])[:, None]
    data = env * (rng.standard_normal((n, 1)) + 0.3 * rng.standard_normal((n, 64)))
    coord = tools.ch_map_to_coord(np.arange(64).reshape(8, 8), ied=8)
    return data, coord, fs, t, rng


def test_find_bad_ch_clean(grid8):
    data, coord, fs, _, _ = grid8
    bad, scores, history = imputation.find_bad_ch(data, coord, fs)

    assert bad.size == 0
    assert history[-1]['status'] == 'converged'
    assert set(scores) == {'amp', 'corr', 'psd_corr'}


def test_find_bad_ch_faults(grid8):
    data, coord, fs, t, rng = grid8
    data[:, 5] = 0                                       # dead
    data[:, 20] += 3 * rng.standard_normal(len(t))       # noisy
    hf = prepro.bandpass_filter(rng.standard_normal((len(t), 1)), fs=fs, cutoff=[300, 500])
    data[:, 50] += 0.5 * np.std(data[:, 50]) * hf[:, 0] / np.std(hf)   # high-frequency noise
    bad, _, history = imputation.find_bad_ch(data, coord, fs)

    np.testing.assert_array_equal(bad, [5, 20, 50])
    final = history[-1]
    assert final['bad'] == bad.tolist()
    assert final['status'] == 'converged'
    assert not np.isin(final['eval_ch'], bad).any()
    assert len(final['loo_eval']) == len(history) - 1

    # Diagnostics of the last iteration
    last = history[-2]
    assert last['bad'] == last['flagged'] == bad.tolist()
    assert 50 in last['flagged_by']['psd_corr']
    assert 5 in last['flagged_by']['corr']               # flat channel: NaN correlation
    assert last['loo_error'].shape == last['dev_from_mean'].shape == (64,)


def test_find_bad_ch_max_frac(grid8):
    data, coord, fs, _, _ = grid8
    data[:, :20] = 0                                     # 20 of 64 dead
    with pytest.warns(UserWarning, match="max_frac"):
        bad, _, history = imputation.find_bad_ch(data, coord, fs)

    assert history[-1]['status'] == 'max_frac'
    assert history[-1]['bad'] == bad.tolist()


def test_find_bad_ch_k_per_score(grid8):
    data, coord, fs, t, rng = grid8
    hf = prepro.bandpass_filter(rng.standard_normal((len(t), 1)), fs=fs, cutoff=[300, 500])
    data[:, 50] += 0.5 * np.std(data[:, 50]) * hf[:, 0] / np.std(hf)

    # A wide fence for psd_corr alone stops it flagging the noisy channel
    _, _, history = imputation.find_bad_ch(data, coord, fs)
    assert 50 in history[0]['flagged_by']['psd_corr']
    _, _, history = imputation.find_bad_ch(data, coord, fs, k={'psd_corr': 1000})
    assert 50 not in history[0]['flagged_by']['psd_corr']
    with pytest.raises(ValueError, match="k keys"):
        imputation.find_bad_ch(data, coord, fs, k={'log_amp': 3})


def test_find_bad_ch_neighbour_method(grid8):
    data, coord, fs, t, rng = grid8
    data[:, 5] = 0                                       # dead
    data[:, 20] += 3 * rng.standard_normal(len(t))       # noisy

    # Both neighbour methods find the same faults on a regular grid
    for kw in ({'neighbour_method': 'delaunay'}, {'max_dist': 8}):
        bad, _, history = imputation.find_bad_ch(data, coord, fs, **kw)
        assert {5, 20} <= set(bad.tolist())

    # The neighbour method is passed to get_neighbours
    with pytest.raises(ValueError, match="method must be"):
        imputation.find_bad_ch(data, coord, fs, neighbour_method='knn')

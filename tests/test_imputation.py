import numpy as np
import pytest

from emg_tools import imputation, tools


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
    out = imputation.replace_bad_ch(data, [0], coord, method="delaunay")
    np.testing.assert_allclose(out[:, 0], data[:, [1, 2, 3]].mean(axis=-1))


def test_replace_bad_ch_delaunay_collinear_uses_2_nearest(rng):
    coord = np.c_[np.arange(5.0), np.zeros(5)]
    data = rng.standard_normal((50, 5))
    out = imputation.replace_bad_ch(data, [2], coord, method="delaunay")
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
    with pytest.raises(ValueError):
        func(emg, [0], ch_coord[:-1])  # one coordinate missing
    with pytest.raises(ValueError):
        func(emg, [0], np.where(np.arange(16)[:, None] == 0, np.nan, ch_coord))


def test_replace_bad_ch_invalid_method(emg, ch_coord):
    with pytest.raises(ValueError):
        imputation.replace_bad_ch(emg, [0], ch_coord, method="nearest")


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
    with pytest.raises(ValueError):
        imputation.interp_bad_ch(rng.standard_normal((50, 5)), [2], coord)


def test_get_loo_error(emg, ch_coord):
    # Smooth fields are predictable from their surroundings, white noise is not
    smooth = np.outer(np.sin(np.arange(100) / 5), ch_coord @ np.array([0.3, -0.2]) + 1)
    error, good = imputation.get_loo_error(smooth, ch_coord, bad_ch=[5])

    np.testing.assert_array_equal(good, np.setdiff1d(np.arange(16), [5]))
    assert error.max() < 1e-6

    noise_error, _ = imputation.get_loo_error(emg[:200], ch_coord)
    assert noise_error.min() > 0.5

import numpy as np
import pytest

from emg_tools import tools


def test_arrange_flatten_roundtrip(emg, ch_map):
    spatial = tools.arrange_data_spatially(emg, ch_map)
    assert spatial.shape == (*ch_map.shape, emg.shape[0])
    np.testing.assert_array_equal(spatial[1, 2], emg[:, ch_map[1, 2]])
    np.testing.assert_array_equal(tools.flatten_data_spatially(spatial, ch_map), emg)


def test_arrange_and_flatten_with_empty_electrodes():
    ch_map = np.array([[0, 1], [3, -1]])  # channel 2 unmapped, one empty electrode
    data = np.ones((5, 4))

    spatial = tools.arrange_data_spatially(data, ch_map)
    np.testing.assert_array_equal(spatial[1, 1], 0)

    flat = tools.flatten_data_spatially(spatial, ch_map)
    np.testing.assert_array_equal(flat[:, 2], 0)
    np.testing.assert_array_equal(flat[:, [0, 1, 3]], 1)


# Channel 0 is a corner with neighbours 1, 4 and 5. Bad channels are replaced in
# order, and channels still pending replacement are not used as neighbours.
@pytest.mark.parametrize(
    ("bad_ch", "neigh"),
    [([0], [1, 4, 5]), ([5, 0], [1, 4, 5]), (np.array([0, 5]), [1, 4])],
)
def test_replace_bad_ch_mean_of_neighbours(emg, ch_map, bad_ch, neigh):
    original = emg.copy()
    out = tools.replace_bad_ch(emg, bad_ch, ch_map)

    np.testing.assert_allclose(out[:, 0], out[:, neigh].mean(axis=-1))
    np.testing.assert_array_equal(emg, original)  # input not modified


def test_replace_bad_ch_3d_matches_2d(emg, ch_map):
    out_2d = tools.replace_bad_ch(emg, [5, 0], ch_map)
    out_3d = tools.replace_bad_ch(tools.arrange_data_spatially(emg, ch_map), [5, 0], ch_map)
    np.testing.assert_allclose(tools.flatten_data_spatially(out_3d, ch_map), out_2d)


def test_replace_bad_ch_without_good_neighbours_is_zero():
    out = tools.replace_bad_ch(np.ones((5, 2)), [0, 1], np.array([[0, 1]]))
    np.testing.assert_array_equal(out, 0)

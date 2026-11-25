import numpy as np

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


def test_ch_map_to_coord():
    ch_map = np.array([[0, 1, 2], [4, -1, 3]])  # one empty electrode
    coord = tools.ch_map_to_coord(ch_map, ied=(8, 4))

    assert coord.shape == (5, 2)
    np.testing.assert_array_equal(coord, [[0, 0], [8, 0], [16, 0], [16, 4], [0, 4]])

    coord = tools.ch_map_to_coord(np.array([[0, -1, 2]]))  # channel 1 not on the grid
    assert np.isnan(coord[1]).all()

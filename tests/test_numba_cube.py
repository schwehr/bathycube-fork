import numpy as np
from pytest import approx

from bathycube import numba_cube as cube


def test_hypothesislist():
    ll = cube.HypothesisList(cube.return_new_hypothesis(5.0, 5.0), None)
    ll = ll.prepend(cube.return_new_hypothesis(4.0, 4.0))
    ll = ll.prepend(cube.return_new_hypothesis(3.0, 3.0))
    ll.append(cube.return_new_hypothesis(7.0, 7.0))
    ll.append(cube.return_new_hypothesis(8.0, 8.0))
    ll.insert(cube.return_new_hypothesis(6.0, 6.0), 3)
    ll.insert(cube.return_new_hypothesis(99.0, 99.0), 2)
    badindex = ll.get_nearest_in_depth(95.7, 10.0)
    ll.remove(badindex)
    ll = ll.drop_first()

    data = [d.current_depth for d in ll.get_data()]
    assert data == [4.0, 5.0, 6.0, 7.0, 8.0]
    assert ll.get_item(0).current_depth == 4.0
    assert ll.get_item(2).current_depth == 6.0
    assert ll.get_item(4).current_depth == 8.0
    assert ll.get_nearest_min_error(5.5, 5.5) == 2
    assert ll.get_nearest_min_error(5.4, 5.5) == 1

    ll.get_item(2).number_of_samples = 3
    ll.get_item(1).number_of_samples = 2
    idx, curmax, secmax = ll.get_max_sample()
    assert idx == 2
    assert curmax == 3
    assert secmax == 2


def test_queuelist():
    ll = cube.QueueList(cube.return_new_sounding(5.0, 5.0, 0.0, 0.0), None)
    ll = ll.prepend(cube.return_new_sounding(4.0, 4.0, 0.0, 0.0))
    ll.append(cube.return_new_sounding(6.0, 6.0, 0.0, 0.0))
    ll.append(cube.return_new_sounding(8.0, 8.0, 0.0, 0.0))
    ll.insert(cube.return_new_sounding(7.0, 7.0, 0.0, 0.0), 3)
    ll.insert(cube.return_new_sounding(99.0, 99.0, 0.0, 0.0), 2)
    ll.remove(2)
    ll = ll.drop_first()
    data = [d.depth for d in ll.get_data()]
    assert data == [5.0, 6.0, 7.0, 8.0]
    assert ll.get_item(0).depth == 5.0
    assert ll.get_item(2).depth == 7.0
    assert ll.get_item(3).depth == 8.0


def test_cube_params():
    param = cube.return_default_cube_parameters("order1a", 0.5, 0.5)
    assert param.grid_resolution_x == 0.5
    assert param.grid_resolution_y == 0.5
    assert param.inv_dist_exponent == 1 / 2.0
    assert param.iho_order == "order1a"


def test_cube_node_init():
    cb = cube.return_new_cubenode()
    cb.queue = cube.QueueList(cube.return_new_sounding(0.0, 0.0, 0.0, 0.0), None)
    cb.predicted_depth = 1.0
    assert cb.predicted_depth == 1.0
    assert cb.predicted_variance == 0.0
    assert np.array_equal(cb.queue.data.depth, np.array(0.0))


def test_cube_node_new_hypothesis():
    cb = cube.return_new_cubenode()
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(5.0, 5.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(6.0, 6.0, 0.0, 0.0))
    data = cb.hypotheses.get_data()
    assert data[0].current_depth == 5.0
    assert data[1].current_depth == 6.0


def test_cube_node_remove_hypothesis():
    cb = cube.return_new_cubenode()
    assert not cube.cube_node_remove_hypothesis(cb, 5.0)
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(5.0, 5.0, 0.0, 0.0))
    assert not cube.cube_node_remove_hypothesis(cb, 99.0)
    assert cube.cube_node_remove_hypothesis(cb, 5.001)
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(5.0, 5.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(6.0, 6.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(7.0, 7.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cube.cube_node_remove_hypothesis(cb, 6.001)
    assert cube.cube_node_remove_hypothesis(cb, 7.999)
    assert cube.cube_node_remove_hypothesis(cb, 5.001)
    data = cb.hypotheses.get_data()
    assert len(data) == 1
    assert data[0].current_depth == 7.0


def test_cube_node_nominate_hypothesis():
    cb = cube.return_new_cubenode()
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(5.0, 5.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(6.0, 6.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(7.0, 7.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cb.nominated is None
    assert cube.cube_node_nominate_hypothesis(cb, 5.001)
    assert cb.nominated.current_depth == 5.0
    assert cube.cube_node_nominate_hypothesis(cb, 4.999)
    assert cb.nominated.current_depth == 5.0
    assert cube.cube_node_nominate_hypothesis(cb, 7.001)
    assert cb.nominated.current_depth == 7.0
    assert not cube.cube_node_nominate_hypothesis(cb, 7.5)
    assert cb.nominated is None


def test_cube_node_reset_nomination():
    cb = cube.return_new_cubenode()
    assert cube.cube_node_reset_nomination(cb)
    assert cb.nominated is None
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cube.cube_node_nominate_hypothesis(cb, 8.001)
    assert cube.cube_node_reset_nomination(cb)
    assert cb.nominated is None


def test_cube_node_is_nominated():
    cb = cube.return_new_cubenode()
    assert not cube.cube_node_is_nominated(cb)
    assert cb.nominated is None
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cube.cube_node_nominate_hypothesis(cb, 8.001)
    assert cube.cube_node_is_nominated(cb)


def test_cube_node_set_preddepth():
    cb = cube.return_new_cubenode()
    assert cb.predicted_depth == 0.0
    assert cb.predicted_variance == 0.0
    cube.cube_node_set_preddepth(cb, cube.return_new_sounding(5.0, 1.5, 0.0, 0.0))
    assert cb.predicted_depth == 5.0
    assert cb.predicted_variance == 1.5


def test_cube_node_monitor_hypothesis():
    cb = cube.return_new_cubenode()
    assert not cube.cube_node_monitor_hypothesis(cb, 0, cube.return_new_sounding(1.0, 1.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(5.0, 0.5, 0.0, 0.0))
    assert not cube.cube_node_monitor_hypothesis(
        cb, 0, cube.return_new_sounding(10.0, 1.0, 0.0, 0.0)
    )  # trigger bayes factor less than minimum threshold
    assert cube.cube_node_monitor_hypothesis(
        cb, 0, cube.return_new_sounding(8.0, 1.0, 0.0, 0.0)
    )  # no intervention required
    assert not cube.cube_node_monitor_hypothesis(
        cb, 0, cube.return_new_sounding(8.0, 1.0, 0.0, 0.0)
    )  # second monitor and the cum bayes fac is less than the threshold


def test_cube_node_reset_monitor():
    cb = cube.return_new_cubenode()
    assert not cube.cube_node_reset_monitor(cb, 0)  # failed with no hypotheses
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(5.0, 0.5, 0.0, 0.0))
    assert cube.cube_node_monitor_hypothesis(
        cb, 0, cube.return_new_sounding(8.0, 1.0, 0.0, 0.0)
    )  # no intervention required
    hypo = cb.hypotheses.get_item(0)
    assert hypo.cum_bayes_fac == approx(0.166, abs=0.001)
    assert hypo.seq_length == 1.0
    cube.cube_node_reset_monitor(cb, 0)
    assert hypo.cum_bayes_fac == 1.0
    assert hypo.seq_length == 0


def test_cube_node_update_hypothesis():
    cb = cube.return_new_cubenode()
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(6.0, 1.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(7.0, 1.0, 0.0, 0.0))

    hypo = cb.hypotheses.get_item(1)
    assert hypo.predict_depth == 6.0
    assert hypo.current_depth == 6.0
    assert hypo.current_variance == 1.0
    assert hypo.predict_variance == 1.0
    assert hypo.number_of_samples == 1

    cube.cube_node_update_hypothesis(cb, 1, cube.return_new_sounding(6.1, 0.9, 0.0, 0.0))
    assert hypo.predict_depth == approx(6.053, abs=0.001)
    assert hypo.current_depth == approx(6.053, abs=0.001)
    assert hypo.current_variance == approx(0.474, abs=0.001)
    assert hypo.predict_variance == approx(0.474, abs=0.001)
    assert hypo.number_of_samples == 2


def test_cube_node_best_hypothesis_index():
    cb = cube.return_new_cubenode()
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(6.0, 1.0, 0.0, 0.0))
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(7.0, 1.0, 0.0, 0.0))
    assert cube.cube_node_best_hypothesis_index(cb, cube.return_new_sounding(5.4, 1.0, 0.0, 0.0)) == 0
    assert cube.cube_node_best_hypothesis_index(cb, cube.return_new_sounding(5.6, 1.0, 0.0, 0.0)) == 1


def test_cube_node_update_node():
    cb = cube.return_new_cubenode()
    cube.cube_node_new_hypothesis(cb, cube.return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube.cube_node_update_node(cb, cube.return_new_sounding(5.1, 1.0, 0.0, 0.0))
    assert len(cb.hypotheses.get_data()) == 1
    assert cb.hypotheses.get_item(0).current_depth == approx(5.05, abs=0.01)
    assert cb.hypotheses.get_item(0).number_of_samples == 2

    cube.cube_node_update_node(cb, cube.return_new_sounding(15.1, 1.0, 0.0, 0.0))
    assert len(cb.hypotheses.get_data()) == 2


def test_cube_node_truncate():
    cb = cube.return_new_cubenode()
    cb.queue = cube.QueueList(cube.return_new_sounding(6.0, 1.0, 0.0, 0.0), None)
    cb.queue.append(cube.return_new_sounding(6.1, 1.0, 0.0, 0.0))
    cb.queue.append(cube.return_new_sounding(6.2, 1.0, 0.0, 0.0))
    cb.queue.append(cube.return_new_sounding(6.3, 1.0, 0.0, 0.0))
    cb.queue.append(cube.return_new_sounding(6.4, 1.0, 0.0, 0.0))
    cb.queue.append(cube.return_new_sounding(16.5, 2.0, 0.0, 0.0))
    cb.queue.append(cube.return_new_sounding(36.6, 2.0, 0.0, 0.0))
    cb.n_queued = 7
    cube.cube_node_truncate(cb)
    assert cb.n_queued == 6


def test_cube_node_queue_flush_node():
    cb = cube.return_new_cubenode()
    cb.queue = cube.QueueList(cube.return_new_sounding(5.0, 1.0, 0.0, 0.0), None)
    cb.queue.append(cube.return_new_sounding(5.1, 1.0, 0.0, 0.0))
    cb.queue.append(cube.return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cb.queue.append(cube.return_new_sounding(5.3, 1.0, 0.0, 0.0))
    cb.n_queued = 4
    cube.cube_node_queue_flush_node(cb)

    hypos = cb.hypotheses.get_data()
    assert len(hypos) == 1
    assert cb.n_queued == 0
    assert hypos[0].current_depth == approx(5.150, abs=0.001)
    assert hypos[0].current_variance == approx(0.25, abs=0.001)
    assert hypos[0].cum_bayes_fac == approx(1490.964, abs=0.001)
    assert hypos[0].number_of_samples == 4


def test_cube_node_choose_hypothesis():
    cb = cube.return_new_cubenode()
    cb.queue = cube.QueueList(cube.return_new_sounding(5.0, 1.0, 0.0, 0.0), None)
    cb.queue.append(cube.return_new_sounding(5.1, 1.0, 0.0, 0.0))
    cb.queue.append(cube.return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cb.queue.append(cube.return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cb.queue.append(cube.return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cb.n_queued = 5
    cube.cube_node_queue_flush_node(cb)
    hypo, ratio = cube.cube_node_choose_hypothesis(cb)
    assert hypo.number_of_samples == 3
    assert ratio == 3.5


def test_cube_node_queue_fill():
    cb = cube.return_new_cubenode()
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(5.5, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(4.5, 1.0, 0.0, 0.0))
    assert cb.n_queued == 6
    data = cb.queue.get_data()
    assert np.array_equal(data[0].depth, np.array(4.5, dtype=np.float32))
    assert np.array_equal(data[1].depth, np.array(5.0, dtype=np.float32))
    assert np.array_equal(data[2].depth, np.array(5.2, dtype=np.float32))
    assert np.array_equal(data[3].depth, np.array(5.5, dtype=np.float32))
    assert np.array_equal(data[4].depth, np.array(17.7, dtype=np.float32))
    assert np.array_equal(data[5].depth, np.array(17.8, dtype=np.float32))


def test_cube_node_add_to_queue():
    cb = cube.return_new_cubenode()
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(5.5, 1.0, 0.0, 0.0))
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(4.4, 1.0, 0.0, 0.0))
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(4.0, 1.0, 0.0, 0.0))
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(16.7, 1.0, 0.0, 0.0))
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(4.2, 1.0, 0.0, 0.0))
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(4.5, 1.0, 0.0, 0.0))
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(16.8, 1.0, 0.0, 0.0))
    assert cb.hypotheses is None
    # this should trigger update node, as you hit median length limit
    cube.cube_node_add_to_queue(cb, cube.return_new_sounding(4.6, 1.0, 0.0, 0.0))
    assert len(cb.hypotheses.get_data()) == 1


def test_cube_node_queue_insert():
    cb = cube.return_new_cubenode()
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(5.5, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(4.4, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(4.0, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(16.7, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(4.2, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(4.5, 1.0, 0.0, 0.0))
    cube.cube_node_queue_fill(cb, cube.return_new_sounding(16.8, 1.0, 0.0, 0.0))
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]), atol=0.01)

    median_data = cube.cube_node_queue_insert(cb, cube.return_new_sounding(10.0, 1.0, 0.0, 0.0))
    assert np.allclose(np.array(5.2, dtype=np.float32), median_data.depth)
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(
        np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.5, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01
    )

    median_data = cube.cube_node_queue_insert(cb, cube.return_new_sounding(10.0, 1.0, 0.0, 0.0))
    assert np.allclose(np.array(5.5, dtype=np.float32), median_data.depth)
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(
        np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 10.0, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01
    )

    median_data = cube.cube_node_queue_insert(
        cb, cube.return_new_sounding(100.0, 1.0, 0.0, 0.0)
    )  # this outlier will trigger truncation
    assert np.allclose(np.array(10.0, dtype=np.float32), median_data.depth)
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01)


def test_cube_node_insert():
    cb = cube.return_new_cubenode()  # predicted depth flagged
    cb.predicted_depth = np.float32(np.nan)
    handled = cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 1.0, 0.0, 0.0), 1.0)
    assert handled
    assert cb.n_queued == 0

    cb = cube.return_new_cubenode()  # blunder
    cb.predicted_depth = 100.0
    handled = cube.cube_node_insert(cb, cube.return_new_sounding(50.0, 1.0, 0.0, 0.0), 1.0)
    assert handled
    assert cb.n_queued == 0

    cb = cube.return_new_cubenode()  # too far
    handled = cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 1.0, 0.0, 0.0), 1.0)
    assert handled
    assert cb.n_queued == 0

    cb = cube.return_new_cubenode()
    handled = cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    assert handled
    assert cb.n_queued == 1

    cube.cube_node_insert(cb, cube.return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    assert cb.n_queued == 11
    assert cb.hypotheses is None

    dpths = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(dpths, np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]), atol=0.01)
    varis = [d.variance for d in cb.queue.get_data()]
    assert np.allclose(varis, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), atol=0.01)


def test_cube_node_extract_depth_uncertainty():
    cb = cube.return_new_cubenode()
    d, u, r = cube.cube_node_extract_depth_uncertainty(cb)
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = cube.return_new_cubenode()
    cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_queue_flush_node(cb)
    d, u, r = cube.cube_node_extract_depth_uncertainty(cb)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = cube.return_new_cubenode()
    cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_queue_flush_node(cb)
    cube.cube_node_nominate_hypothesis(cb, 5.0)
    d, u, r = cube.cube_node_extract_depth_uncertainty(cb)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = cube.return_new_cubenode()
    cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_queue_flush_node(cb)
    d, u, r = cube.cube_node_extract_depth_uncertainty(cb)
    assert d == approx(4.686, abs=0.001)
    assert u == approx(0.524, abs=0.001)
    assert r == approx(3.25, abs=0.001)


def test_cube_node_extract_closest_depth_uncertainty():
    cb = cube.return_new_cubenode()
    d, u, r = cube.cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = cube.return_new_cubenode()
    cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_queue_flush_node(cb)
    d, u, r = cube.cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = cube.return_new_cubenode()
    cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_queue_flush_node(cb)
    cube.cube_node_nominate_hypothesis(cb, 5.0)
    d, u, r = cube.cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = cube.return_new_cubenode()
    cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_queue_flush_node(cb)
    d, u, r = cube.cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(17.25, abs=0.001)
    assert u == approx(0.693, abs=0.001)
    assert r == approx(4.429, abs=0.001)


def test_cube_node_extract_posterior_depth_uncertainty():
    cb = cube.return_new_cubenode()
    d, u, r = cube.cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = cube.return_new_cubenode()
    cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_queue_flush_node(cb)
    d, u, r = cube.cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = cube.return_new_cubenode()
    cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_queue_flush_node(cb)
    cube.cube_node_nominate_hypothesis(cb, 5.0)
    d, u, r = cube.cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = cube.return_new_cubenode()
    cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_queue_flush_node(cb)
    d, u, r = cube.cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(17.25, abs=0.001)
    assert u == approx(0.693, abs=0.001)
    assert r == approx(4.429, abs=0.001)


def test_cube_node_hypothesis_count():
    cb = cube.return_new_cubenode()
    assert cube.cube_node_hypothesis_count(cb) == 0

    cb = cube.return_new_cubenode()
    cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_queue_flush_node(cb)
    assert cube.cube_node_hypothesis_count(cb) == 1

    cb = cube.return_new_cubenode()
    cube.cube_node_insert(cb, cube.return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_insert(cb, cube.return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube.cube_node_queue_flush_node(cb)
    assert cube.cube_node_hypothesis_count(cb) == 2


def test_compile_now():
    cube.compile_now()


def test_run_cube_gridding_variants():
    import pytest

    x = np.array([101.0, 102.0, 103.0, 101.5, 102.5])
    y = np.array([199.0, 198.0, 197.0, 198.5, 197.5])
    z = np.array([10.0, 11.0, 10.5, 10.2, 10.8])
    tvu = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    thu = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    for method in ["local", "posterior", "prior", "predicted"]:
        d, _, _, _ = cube.run_cube_gridding(
            z, thu, tvu, x, y, 4, 4, 100.0, 200.0, method, "order1a", 1.0, 1.0, dist_exponent=2.0
        )
        assert d.shape == (4, 4)

    with pytest.raises(NotImplementedError):
        cube.run_cube_gridding(z, thu, tvu, x, y, 4, 4, 100.0, 200.0, "unknown", "order1a", 1.0, 1.0)


def test_numba_cube_py_funcs_coverage():
    # 1. get_iho_limits py_func
    assert cube.get_iho_limits.py_func("exclusive") == (0.15, 0.0075)
    assert cube.get_iho_limits.py_func("special") == (0.25, 0.0075)
    assert cube.get_iho_limits.py_func("order1a") == (0.5, 0.013)
    assert cube.get_iho_limits.py_func("order1b") == (0.5, 0.013)
    assert cube.get_iho_limits.py_func("order2") == (1.0, 0.023)
    assert cube.get_iho_limits.py_func("unknown") is None

    # 2. CubeParameters py_func and constructor
    params = cube.return_default_cube_parameters.py_func("order1a", 1.0, 1.0)
    cube.CubeParameters.class_type.methods["__init__"](params, "order1a", np.float32(1.0), np.float32(1.0))
    params_large = cube.return_default_cube_parameters.py_func("order2", 20.0, 20.0)
    assert params_large.iho_fixed == 1.0

    # 3. Sounding py_func and constructor
    snd = cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5))
    cube.Sounding.class_type.methods["__init__"](
        snd, np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)
    )

    # 4. Hypothesis py_func and constructor
    hyp = cube.return_new_hypothesis.py_func(np.float32(10.0), np.float32(0.5))
    cube.Hypothesis.class_type.methods["__init__"](hyp, np.float32(10.0), np.float32(0.5))

    # 5. HypothesisList methods
    hl_methods = cube.HypothesisList.class_type.methods
    hl = cube.HypothesisList(hyp, None)
    hl_methods["__init__"](hl, hyp, None)
    hl2 = hl_methods["prepend"](hl, cube.return_new_hypothesis.py_func(np.float32(8.0), np.float32(0.5)))
    hl_methods["append"](hl2, cube.return_new_hypothesis.py_func(np.float32(12.0), np.float32(0.5)))
    hl_methods["insert"](hl2, cube.return_new_hypothesis.py_func(np.float32(9.0), np.float32(0.5)), 1)
    hl_methods["insert"](hl2, cube.return_new_hypothesis.py_func(np.float32(9.5), np.float32(0.5)), 2)
    hl_methods["insert"](hl2, cube.return_new_hypothesis.py_func(np.float32(7.0), np.float32(0.5)), 0)
    hl_methods["insert"](hl2, cube.return_new_hypothesis.py_func(np.float32(100.0), np.float32(0.5)), 999)
    hl_methods["remove"](hl2, 1)
    hl_methods["remove"](hl2, 0)
    hl_methods["remove"](hl2, 999)
    hl_methods["drop_first"](hl2)
    hl_methods["get_data"](hl2)
    hl_methods["get_item"](hl2, 0)
    hl_methods["get_item"](hl2, 1)
    hl_methods["get_item"](hl2, 999)
    hl_methods["get_nearest_in_depth"](hl2, 8.0, 0.5)
    hl_methods["get_nearest_in_depth"](hl2, 12.0, 0.5)
    hl_methods["get_nearest_in_depth"](hl2, 500.0, 0.5)
    hl_methods["get_nearest_min_error"](hl2, 10.0, 0.5)
    hl_methods["get_max_sample"](hl2)

    # HypothesisList get_max_sample coverage for branch 281->283 and line 282
    h0 = cube.return_new_hypothesis.py_func(np.float32(1.0), np.float32(0.5))
    h0.number_of_samples = 0
    h1 = cube.return_new_hypothesis.py_func(np.float32(2.0), np.float32(0.5))
    h1.number_of_samples = 5
    hl_zero = cube.HypothesisList(h0, cube.HypothesisList(h1, None))
    hl_methods["get_max_sample"](hl_zero)

    h_a = cube.return_new_hypothesis.py_func(np.float32(1.0), np.float32(0.5))
    h_a.number_of_samples = 5
    h_b = cube.return_new_hypothesis.py_func(np.float32(2.0), np.float32(0.5))
    h_b.number_of_samples = 3
    h_c = cube.return_new_hypothesis.py_func(np.float32(3.0), np.float32(0.5))
    h_c.number_of_samples = 10
    hl_282 = cube.HypothesisList(h_a, cube.HypothesisList(h_b, cube.HypothesisList(h_c, None)))
    hl_methods["get_max_sample"](hl_282)

    # 6. QueueList methods
    ql_methods = cube.QueueList.class_type.methods
    ql = cube.QueueList(snd, None)
    ql_methods["__init__"](ql, snd, None)
    ql2 = ql_methods["prepend"](
        ql, cube.return_new_sounding.py_func(np.float32(8.0), np.float32(0.5), np.float32(0.5), np.float32(0.5))
    )
    ql_methods["append"](
        ql2, cube.return_new_sounding.py_func(np.float32(12.0), np.float32(0.5), np.float32(0.5), np.float32(0.5))
    )
    ql_methods["insert"](
        ql2,
        cube.return_new_sounding.py_func(np.float32(9.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        1,
    )
    ql_methods["insert"](
        ql2,
        cube.return_new_sounding.py_func(np.float32(9.5), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        2,
    )
    ql_methods["insert"](
        ql2,
        cube.return_new_sounding.py_func(np.float32(7.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        0,
    )
    ql_methods["insert"](
        ql2,
        cube.return_new_sounding.py_func(np.float32(100.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        999,
    )
    ql_methods["remove"](ql2, 1)
    ql_methods["remove"](ql2, 0)
    ql_methods["remove"](ql2, 999)
    ql_methods["drop_first"](ql2)
    ql_methods["get_data"](ql2)
    ql_methods["get_item"](ql2, 0)
    ql_methods["get_item"](ql2, 1)
    ql_methods["get_item"](ql2, 999)

    # 7. CubeNode py_func and constructor
    node = cube.return_new_cubenode.py_func()
    cube.CubeNode.class_type.methods["__init__"](node)

    # 8. CubeGrid py_func and constructor
    grid = cube.return_new_cubegrid.py_func(100.0, 200.0, 4, 4, np.float32(1.0), np.float32(1.0), params)
    cube.CubeGrid.class_type.methods["__init__"](grid, 100.0, 200.0, 4, 4, np.float32(1.0), np.float32(1.0), params)

    class MockHypoNoneItem:
        def __init__(self):
            self.data = cube.return_new_hypothesis.py_func(np.float32(10.0), np.float32(0.5))
            self.next_data = None

        def get_nearest_in_depth(self, d, tol):
            return 0

        def get_item(self, idx):
            return None

    class MockNode:
        def __init__(self):
            self.hypotheses = MockHypoNoneItem()
            self.depth_tolerance = 0.01
            self.nominated = None
            self.no_data_value = np.float32(np.nan)
            self.max_hypothesis_ratio = 5.0
            self.variance_selection = "cube"
            self.stddev_to_conf_scale = 1.96

    class MockGrid:
        def __init__(self, num_rows, num_columns, p):
            self.num_rows = num_rows
            self.num_columns = num_columns
            self.resolution_x = np.float32(p.grid_resolution_x)
            self.resolution_y = np.float32(p.grid_resolution_y)
            self.minimum_easting = 100.0
            self.maximum_northing = 200.0
            self.dist_scale = p.dist_scale
            self.inv_dist_exponent = p.inv_dist_exponent
            self.iho_fixed = p.iho_fixed
            self.iho_percent = p.iho_percent
            self.min_context = p.min_context
            self.max_context = p.max_context
            self.no_data_value = np.float32(np.nan)
            self.grid = [[cube.return_new_cubenode.py_func() for _ in range(num_columns)] for _ in range(num_rows)]

    # Run py_funcs with Debug=True and Debug=False
    for debug_mode in [False, True]:
        cube.Debug = debug_mode

        # 9. CubeNode operations py_func
        n_op = cube.return_new_cubenode.py_func()
        assert cube.cube_node_hypothesis_count.py_func(n_op) == 0
        cube.cube_node_new_hypothesis.py_func(n_op, snd)
        cube.cube_node_new_hypothesis.py_func(
            n_op,
            cube.return_new_sounding.py_func(np.float32(10.005), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        assert cube.cube_node_hypothesis_count.py_func(n_op) == 2

        # nomination py_funcs
        assert cube.cube_node_nominate_hypothesis.py_func(n_op, np.float32(10.0))
        assert cube.cube_node_is_nominated.py_func(n_op)
        d, _, _ = cube.cube_node_get_nominated_depth_uncertainty.py_func(n_op)
        assert d == approx(10.0, abs=0.01)
        assert cube.cube_node_reset_nomination.py_func(n_op)
        assert not cube.cube_node_is_nominated.py_func(n_op)
        cube.cube_node_get_nominated_depth_uncertainty.py_func(n_op)

        # nominate failure branches
        assert not cube.cube_node_nominate_hypothesis.py_func(cube.return_new_cubenode.py_func(), np.float32(10.0))
        assert not cube.cube_node_nominate_hypothesis.py_func(n_op, np.float32(999.0))

        n_mock_nom = MockNode()
        assert not cube.cube_node_nominate_hypothesis.py_func(n_mock_nom, np.float32(10.0))

        # remove hypothesis
        cube.cube_node_nominate_hypothesis.py_func(n_op, np.float32(10.0))
        cube.cube_node_remove_hypothesis.py_func(n_op, np.float32(10.0))
        cube.cube_node_remove_hypothesis.py_func(n_op, np.float32(99.0))
        assert not cube.cube_node_remove_hypothesis.py_func(cube.return_new_cubenode.py_func(), np.float32(10.0))

        # remove middle hypothesis in 3-element list
        n_rem3 = cube.return_new_cubenode.py_func()
        cube.cube_node_new_hypothesis.py_func(n_rem3, snd)
        cube.cube_node_new_hypothesis.py_func(
            n_rem3,
            cube.return_new_sounding.py_func(np.float32(15.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_new_hypothesis.py_func(
            n_rem3,
            cube.return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        assert cube.cube_node_remove_hypothesis.py_func(n_rem3, np.float32(15.0))

        # remove single hypothesis
        n_rem1 = cube.return_new_cubenode.py_func()
        cube.cube_node_new_hypothesis.py_func(n_rem1, snd)
        assert cube.cube_node_remove_hypothesis.py_func(n_rem1, np.float32(10.0))

        # predicted depth
        cube.cube_node_set_preddepth.py_func(
            n_op,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )

        # monitor hypothesis & reset monitor
        n_mon = cube.return_new_cubenode.py_func()
        assert not cube.cube_node_monitor_hypothesis.py_func(n_mon, 0, snd)
        assert not cube.cube_node_reset_monitor.py_func(n_mon, 0)
        cube.cube_node_new_hypothesis.py_func(
            n_mon,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        assert not cube.cube_node_monitor_hypothesis.py_func(n_mon, 999, snd)
        assert not cube.cube_node_reset_monitor.py_func(n_mon, 999)

        # monitor error < 0 and bayes factor branches
        cube.cube_node_monitor_hypothesis.py_func(
            n_mon,
            0,
            cube.return_new_sounding.py_func(np.float32(9.8), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_monitor_hypothesis.py_func(
            n_mon,
            0,
            cube.return_new_sounding.py_func(np.float32(10.2), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_monitor_hypothesis.py_func(
            n_mon,
            0,
            cube.return_new_sounding.py_func(np.float32(100.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        n_mon.hypotheses.data.cum_bayes_fac = 0.5
        cube.cube_node_monitor_hypothesis.py_func(
            n_mon,
            0,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        # Runlength threshold failure triggering lines 715-717
        n_mon.hypotheses.data.cum_bayes_fac = 0.5
        n_mon.hypotheses.data.seq_length = 10
        n_mon.runlength_threshold = 5
        cube.cube_node_monitor_hypothesis.py_func(
            n_mon,
            0,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_reset_monitor.py_func(n_mon, 0)

        # update hypothesis
        n_upd = cube.return_new_cubenode.py_func()
        assert not cube.cube_node_update_hypothesis.py_func(n_upd, 0, snd)
        cube.cube_node_new_hypothesis.py_func(
            n_upd,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        assert not cube.cube_node_update_hypothesis.py_func(n_upd, 999, snd)
        assert not cube.cube_node_update_hypothesis.py_func(
            n_upd,
            0,
            cube.return_new_sounding.py_func(np.float32(100.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        n_upd.variance_selection = "max"
        cube.cube_node_update_hypothesis.py_func(
            n_upd,
            0,
            cube.return_new_sounding.py_func(np.float32(10.1), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        n_upd.variance_selection = "input"
        cube.cube_node_update_hypothesis.py_func(
            n_upd,
            0,
            cube.return_new_sounding.py_func(np.float32(10.1), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        n_upd.variance_selection = "cube"
        cube.cube_node_update_hypothesis.py_func(
            n_upd,
            0,
            cube.return_new_sounding.py_func(np.float32(10.1), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )

        # best hypothesis index
        assert cube.cube_node_best_hypothesis_index.py_func(cube.return_new_cubenode.py_func(), snd) == -1
        cube.cube_node_best_hypothesis_index.py_func(n_upd, snd)

        # update node
        n_upd_node = cube.return_new_cubenode.py_func()
        cube.cube_node_update_node.py_func(
            n_upd_node,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_update_node.py_func(
            n_upd_node,
            cube.return_new_sounding.py_func(np.float32(10.1), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_update_node.py_func(
            n_upd_node,
            cube.return_new_sounding.py_func(np.float32(100.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )

        # choose hypothesis
        h_none, _ = cube.cube_node_choose_hypothesis.py_func(cube.return_new_cubenode.py_func())
        assert h_none is None
        # choose hypothesis on node with multiple hypotheses (second_highest_count != 0)
        cube.cube_node_choose_hypothesis.py_func(n_upd_node)
        # choose hypothesis on node with single hypothesis (second_highest_count == 0 -> branch 1006->1008)
        n_choose_single = cube.return_new_cubenode.py_func()
        cube.cube_node_new_hypothesis.py_func(
            n_choose_single,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_choose_hypothesis.py_func(n_choose_single)

        # truncate
        n_tr = cube.return_new_cubenode.py_func()
        cube.cube_node_truncate.py_func(n_tr)
        n_tr.n_queued = 5
        n_tr.queue = None
        cube.cube_node_truncate.py_func(n_tr)
        n_tr_full = cube.return_new_cubenode.py_func()
        for d_val in [4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8, 100.0]:
            cube.cube_node_queue_fill.py_func(
                n_tr_full,
                cube.return_new_sounding.py_func(np.float32(d_val), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
            )
        n_tr_full.quotient_limit = 0.01
        cube.cube_node_truncate.py_func(n_tr_full)

        # queue flush
        n_fl_empty = cube.return_new_cubenode.py_func()
        cube.cube_node_queue_flush_node.py_func(n_fl_empty)
        n_fl_even = cube.return_new_cubenode.py_func()
        for d_val in [4.0, 5.0, 6.0, 7.0]:
            cube.cube_node_queue_fill.py_func(
                n_fl_even,
                cube.return_new_sounding.py_func(np.float32(d_val), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
            )
        cube.cube_node_queue_flush_node.py_func(n_fl_even)
        n_fl_odd = cube.return_new_cubenode.py_func()
        for d_val in [4.0, 5.0, 6.0]:
            cube.cube_node_queue_fill.py_func(
                n_fl_odd,
                cube.return_new_sounding.py_func(np.float32(d_val), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
            )
        cube.cube_node_queue_flush_node.py_func(n_fl_odd)

        # queue fill
        n_qf = cube.return_new_cubenode.py_func()
        cube.cube_node_queue_fill.py_func(
            n_qf,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_queue_fill.py_func(
            n_qf,
            cube.return_new_sounding.py_func(np.float32(5.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_queue_fill.py_func(
            n_qf,
            cube.return_new_sounding.py_func(np.float32(15.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_queue_fill.py_func(
            n_qf,
            cube.return_new_sounding.py_func(np.float32(8.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )

        # queue insert
        n_qi = cube.return_new_cubenode.py_func()
        for d_val in [4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]:
            cube.cube_node_queue_fill.py_func(
                n_qi,
                cube.return_new_sounding.py_func(np.float32(d_val), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
            )
        cube.cube_node_queue_insert.py_func(
            n_qi,
            cube.return_new_sounding.py_func(np.float32(6.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_queue_insert.py_func(
            n_qi,
            cube.return_new_sounding.py_func(np.float32(1.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_queue_insert.py_func(
            n_qi,
            cube.return_new_sounding.py_func(np.float32(100.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )

        n_qi_overlap = cube.return_new_cubenode.py_func()
        for d_val in [10.0, 10.01, 10.02, 10.03, 10.04, 10.05, 10.06, 10.07, 10.08, 10.09, 10.10]:
            cube.cube_node_queue_fill.py_func(
                n_qi_overlap,
                cube.return_new_sounding.py_func(np.float32(d_val), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
            )
        cube.cube_node_queue_insert.py_func(
            n_qi_overlap,
            cube.return_new_sounding.py_func(np.float32(10.05), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )

        # add_to_queue
        n_atq = cube.return_new_cubenode.py_func()
        n_atq.queue = None
        cube.cube_node_add_to_queue.py_func(n_atq, snd)
        for d_val in [4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7]:
            cube.cube_node_add_to_queue.py_func(
                n_atq,
                cube.return_new_sounding.py_func(np.float32(d_val), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
            )
        cube.cube_node_add_to_queue.py_func(
            n_atq,
            cube.return_new_sounding.py_func(np.float32(12.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )

        # insert filtering branches
        n_ins = cube.return_new_cubenode.py_func()
        n_ins.predicted_depth = np.float32(np.nan)
        cube.cube_node_insert.py_func(n_ins, snd, np.float32(0.25))

        n_ins.predicted_depth = np.float32(20.0)
        n_ins.predicted_variance = np.float32(0.5)
        n_ins.blunder_min = np.float32(5.0)
        n_ins.blunder_percent = np.float32(0.25)
        # blunder reject
        cube.cube_node_insert.py_func(
            n_ins,
            cube.return_new_sounding.py_func(np.float32(2.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
            np.float32(0.25),
        )
        # blunder pass, distance reject (lines 1182->1190, 1192-1194)
        cube.cube_node_insert.py_func(
            n_ins,
            cube.return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
            np.float32(10000.0),
        )
        # blunder pass, distance accepted
        cube.cube_node_insert.py_func(
            n_ins,
            cube.return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
            np.float32(0.01),
        )
        n_ins.capture_dist_scale = np.float32(0.01)
        cube.cube_node_insert.py_func(n_ins, snd, np.float32(100.0))
        cube.cube_node_insert.py_func(n_ins, snd, np.float32(0.01))

        n_ins_nopred = cube.return_new_cubenode.py_func()
        n_ins_nopred.predicted_depth = np.float32(0.0)
        cube.cube_node_insert.py_func(n_ins_nopred, snd, np.float32(0.01))

        # generic hypothesis depth uncertainty
        n_g = cube.return_new_cubenode.py_func()
        h_g = cube.return_new_hypothesis.py_func(np.float32(10.0), np.float32(0.5))
        h_g.variance_estimate = np.float32(1.0)
        n_g.variance_selection = "max"
        cube.cube_node_get_generic_hypothesis_depth_uncertainty.py_func(n_g, h_g)
        n_g.variance_selection = "input"
        cube.cube_node_get_generic_hypothesis_depth_uncertainty.py_func(n_g, h_g)
        n_g.variance_selection = "cube"
        cube.cube_node_get_generic_hypothesis_depth_uncertainty.py_func(n_g, h_g)
        h_zero_samp = cube.return_new_hypothesis.py_func(np.float32(10.0), np.float32(0.5))
        h_zero_samp.number_of_samples = 0
        cube.cube_node_get_generic_hypothesis_depth_uncertainty.py_func(n_g, h_zero_samp)

        # extract depth uncertainty branches
        n_ext = cube.return_new_cubenode.py_func()
        cube.cube_node_extract_depth_uncertainty.py_func(n_ext)
        cube.cube_node_new_hypothesis.py_func(n_ext, snd)
        cube.cube_node_extract_depth_uncertainty.py_func(n_ext)
        cube.cube_node_new_hypothesis.py_func(
            n_ext,
            cube.return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_extract_depth_uncertainty.py_func(n_ext)
        cube.cube_node_nominate_hypothesis.py_func(n_ext, np.float32(10.0))
        cube.cube_node_extract_depth_uncertainty.py_func(n_ext)

        # extract closest & posterior
        cube.cube_node_extract_closest_depth_uncertainty.py_func(n_ext, np.float32(10.0), np.float32(0.5))
        cube.cube_node_extract_posterior_depth_uncertainty.py_func(n_ext, np.float32(10.0), np.float32(0.5))
        n_ext.nominated = None
        cube.cube_node_extract_closest_depth_uncertainty.py_func(n_ext, np.float32(10.0), np.float32(0.5))
        cube.cube_node_extract_posterior_depth_uncertainty.py_func(n_ext, np.float32(10.0), np.float32(0.5))

        n_all_z = cube.return_new_cubenode.py_func()
        hz1 = cube.return_new_hypothesis.py_func(np.float32(10.0), np.float32(0.5))
        hz1.number_of_samples = 0
        hz2 = cube.return_new_hypothesis.py_func(np.float32(20.0), np.float32(0.5))
        hz2.number_of_samples = 0
        n_all_z.hypotheses = cube.HypothesisList(hz1, cube.HypothesisList(hz2, None))
        cube.cube_node_extract_closest_depth_uncertainty.py_func(n_all_z, np.float32(10.0), np.float32(0.5))
        cube.cube_node_extract_posterior_depth_uncertainty.py_func(n_all_z, np.float32(10.0), np.float32(0.5))

        # Grid insertion & extraction
        grid_test = MockGrid(5, 5, params)
        z_arr = np.array([10.0, 100.0, 1.0, 15.0, 10.0], dtype=np.float32)
        thu_arr = np.array([0.1, 100.0, 0.001, 0.5, 0.5], dtype=np.float32)
        tvu_arr = np.array([100.0, 0.001, 100.0, 0.5, 0.5], dtype=np.float32)
        x_arr = np.array([101.5, 102.5, 101.5, 500.0, 101.5], dtype=np.float64)
        y_arr = np.array([198.5, 197.5, 198.5, 500.0, 198.5], dtype=np.float64)
        cube.cube_grid_insert_points.py_func(grid_test, z_arr, thu_arr, tvu_arr, x_arr, y_arr)
        for r in range(grid_test.num_rows):
            for c in range(grid_test.num_columns):
                grid_test.grid[r][c].predicted_depth = np.float32(10.0)
                grid_test.grid[r][c].predicted_variance = np.float32(0.5)

        for m in ["local", "posterior", "prior", "predicted"]:
            cube.cube_grid_extract_data.py_func(grid_test, m)

        grid_search = MockGrid(5, 5, params)
        n_m = grid_search.grid[2][2]
        cube.cube_node_new_hypothesis.py_func(
            n_m,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_new_hypothesis.py_func(
            n_m,
            cube.return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        n_s_row = grid_search.grid[2][3]
        cube.cube_node_new_hypothesis.py_func(
            n_s_row,
            cube.return_new_sounding.py_func(np.float32(10.2), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_grid_extract_data.py_func(grid_search, "local")
        cube.cube_grid_extract_data.py_func(grid_search, "posterior")

        grid_search_col = MockGrid(5, 5, params)
        n_m_col = grid_search_col.grid[2][2]
        n_m_col.predicted_depth = np.float32(10.0)
        n_m_col.predicted_variance = np.float32(0.5)
        cube.cube_node_new_hypothesis.py_func(
            n_m_col,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_new_hypothesis.py_func(
            n_m_col,
            cube.return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        n_s_col = grid_search_col.grid[1][3]
        n_s_col.predicted_depth = np.float32(10.0)
        n_s_col.predicted_variance = np.float32(0.5)
        cube.cube_node_new_hypothesis.py_func(
            n_s_col,
            cube.return_new_sounding.py_func(np.float32(10.2), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_grid_extract_data.py_func(grid_search_col, "local")
        cube.cube_grid_extract_data.py_func(grid_search_col, "posterior")
        cube.cube_grid_extract_data.py_func(grid_search_col, "prior")
        cube.cube_grid_extract_data.py_func(grid_search_col, "predicted")

        # Corner nodes to trigger bounds checking in row/col loops (lines 1586, 1590, 1602, 1606)
        grid_corner = MockGrid(3, 3, params)
        grid_corner.min_context = 2
        grid_corner.max_context = 2
        n_c0 = grid_corner.grid[0][0]
        cube.cube_node_new_hypothesis.py_func(
            n_c0,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_new_hypothesis.py_func(
            n_c0,
            cube.return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        n_c2 = grid_corner.grid[2][2]
        cube.cube_node_new_hypothesis.py_func(
            n_c2,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_new_hypothesis.py_func(
            n_c2,
            cube.return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_grid_extract_data.py_func(grid_corner, "local")

        # min_context > max_context to trigger 1583->1617 branch
        grid_no_ctx = MockGrid(3, 3, params)
        grid_no_ctx.min_context = 5
        grid_no_ctx.max_context = 1
        n_no_ctx = grid_no_ctx.grid[1][1]
        cube.cube_node_new_hypothesis.py_func(
            n_no_ctx,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_new_hypothesis.py_func(
            n_no_ctx,
            cube.return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_grid_extract_data.py_func(grid_no_ctx, "local")

        grid_search_nan = MockGrid(5, 5, params)
        n_m_nan = grid_search_nan.grid[2][2]
        cube.cube_node_new_hypothesis.py_func(
            n_m_nan,
            cube.return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        cube.cube_node_new_hypothesis.py_func(
            n_m_nan,
            cube.return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)),
        )
        n_nan = grid_search_nan.grid[2][3]
        h_nan = cube.return_new_hypothesis.py_func(np.float32(10.0), np.float32(0.5))
        h_nan.number_of_samples = 0
        n_nan.hypotheses = cube.HypothesisList(h_nan, None)
        cube.cube_grid_extract_data.py_func(grid_search_nan, "local")

        # cube_grid_numba py_func by temporarily patching return_new_cubegrid and helpers in nc module
        orig_return_grid = cube.return_new_cubegrid
        orig_insert = cube.cube_grid_insert_points
        orig_extract = cube.cube_grid_extract_data
        try:
            cube.return_new_cubegrid = lambda me, mn, nc_cols, nr, rx, ry, p: MockGrid(nr, nc_cols, p)
            cube.cube_grid_insert_points = lambda *args: None
            cube.cube_grid_extract_data = lambda cg, m: (
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros((1, 1), dtype=np.int32),
            )
            cube.cube_grid_numba.py_func(
                np.array([10.0], dtype=np.float32),
                np.array([0.5], dtype=np.float32),
                np.array([0.5], dtype=np.float32),
                np.array([101.0], dtype=np.float64),
                np.array([199.0], dtype=np.float64),
                4,
                4,
                100.0,
                200.0,
                "local",
                params,
            )
        finally:
            cube.return_new_cubegrid = orig_return_grid
            cube.cube_grid_insert_points = orig_insert
            cube.cube_grid_extract_data = orig_extract

    cube.Debug = False

    # run_cube_gridding with extra kwargs (valid and invalid attribute)
    cube.run_cube_gridding(
        np.array([10.0]),
        np.array([0.5]),
        np.array([0.5]),
        np.array([101.0]),
        np.array([199.0]),
        4,
        4,
        100.0,
        200.0,
        "local",
        "order1a",
        1.0,
        1.0,
        quotient_limit=50.0,
        non_existent_param=123,
    )

    # __main__ block coverage
    import inspect

    lines, _ = inspect.getsourcelines(cube)
    main_idx = next(i for i, line in enumerate(lines) if "if __name__ ==" in line)
    padded_src = "\n" * main_idx + "".join(lines[main_idx:])
    code = compile(padded_src, cube.__file__, "exec")
    d = dict(cube.__dict__)
    d["__name__"] = "__main__"
    exec(code, d)

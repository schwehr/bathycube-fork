from pytest import approx

from bathycube.numba_cube import *


def test_hypothesislist():
    ll = HypothesisList(return_new_hypothesis(5.0, 5.0), None)
    ll = ll.prepend(return_new_hypothesis(4.0, 4.0))
    ll = ll.prepend(return_new_hypothesis(3.0, 3.0))
    ll.append(return_new_hypothesis(7.0, 7.0))
    ll.append(return_new_hypothesis(8.0, 8.0))
    ll.insert(return_new_hypothesis(6.0, 6.0), 3)
    ll.insert(return_new_hypothesis(99.0, 99.0), 2)
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
    ll = QueueList(return_new_sounding(5.0, 5.0, 0.0, 0.0), None)
    ll = ll.prepend(return_new_sounding(4.0, 4.0, 0.0, 0.0))
    ll.append(return_new_sounding(6.0, 6.0, 0.0, 0.0))
    ll.append(return_new_sounding(8.0, 8.0, 0.0, 0.0))
    ll.insert(return_new_sounding(7.0, 7.0, 0.0, 0.0), 3)
    ll.insert(return_new_sounding(99.0, 99.0, 0.0, 0.0), 2)
    ll.remove(2)
    ll = ll.drop_first()
    data = [d.depth for d in ll.get_data()]
    assert data == [5.0, 6.0, 7.0, 8.0]
    assert ll.get_item(0).depth == 5.0
    assert ll.get_item(2).depth == 7.0
    assert ll.get_item(3).depth == 8.0


def test_cube_params():
    param = return_default_cube_parameters('order1a', 0.5, 0.5)
    assert param.grid_resolution_x == 0.5
    assert param.grid_resolution_y == 0.5
    assert param.inv_dist_exponent == 1 / 2.0
    assert param.iho_order == 'order1a'


def test_cube_node_init():
    cb = return_new_cubenode()
    cb.queue = QueueList(return_new_sounding(0.0, 0.0, 0.0, 0.0), None)
    cb.predicted_depth = 1.0
    assert cb.predicted_depth == 1.0
    assert cb.predicted_variance == 0.0
    assert np.array_equal(cb.queue.data.depth, np.array(0.0))


def test_cube_node_new_hypothesis():
    cb = return_new_cubenode()
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 5.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(6.0, 6.0, 0.0, 0.0))
    data = cb.hypotheses.get_data()
    assert data[0].current_depth == 5.0
    assert data[1].current_depth == 6.0


def test_cube_node_remove_hypothesis():
    cb = return_new_cubenode()
    assert not cube_node_remove_hypothesis(cb, 5.0)
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 5.0, 0.0, 0.0))
    assert not cube_node_remove_hypothesis(cb, 99.0)
    assert cube_node_remove_hypothesis(cb, 5.001)
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 5.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(6.0, 6.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(7.0, 7.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cube_node_remove_hypothesis(cb, 6.001)
    assert cube_node_remove_hypothesis(cb, 7.999)
    assert cube_node_remove_hypothesis(cb, 5.001)
    data = cb.hypotheses.get_data()
    assert len(data) == 1
    assert data[0].current_depth == 7.0


def test_cube_node_nominate_hypothesis():
    cb = return_new_cubenode()
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 5.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(6.0, 6.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(7.0, 7.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cb.nominated is None
    assert cube_node_nominate_hypothesis(cb, 5.001)
    assert cb.nominated.current_depth == 5.0
    assert cube_node_nominate_hypothesis(cb, 4.999)
    assert cb.nominated.current_depth == 5.0
    assert cube_node_nominate_hypothesis(cb, 7.001)
    assert cb.nominated.current_depth == 7.0
    assert not cube_node_nominate_hypothesis(cb, 7.5)
    assert cb.nominated is None


def test_cube_node_reset_nomination():
    cb = return_new_cubenode()
    assert cube_node_reset_nomination(cb)
    assert cb.nominated is None
    cube_node_new_hypothesis(cb, return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cube_node_nominate_hypothesis(cb, 8.001)
    assert cube_node_reset_nomination(cb)
    assert cb.nominated is None


def test_cube_node_is_nominated():
    cb = return_new_cubenode()
    assert not cube_node_is_nominated(cb)
    assert cb.nominated is None
    cube_node_new_hypothesis(cb, return_new_sounding(8.0, 8.0, 0.0, 0.0))
    assert cube_node_nominate_hypothesis(cb, 8.001)
    assert cube_node_is_nominated(cb)


def test_cube_node_set_preddepth():
    cb = return_new_cubenode()
    assert cb.predicted_depth == 0.0
    assert cb.predicted_variance == 0.0
    cube_node_set_preddepth(cb, return_new_sounding(5.0, 1.5, 0.0, 0.0))
    assert cb.predicted_depth == 5.0
    assert cb.predicted_variance == 1.5


def test_cube_node_monitor_hypothesis():
    cb = return_new_cubenode()
    assert not cube_node_monitor_hypothesis(cb, 0, return_new_sounding(1.0, 1.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 0.5, 0.0, 0.0))
    assert not cube_node_monitor_hypothesis(cb, 0, return_new_sounding(10.0, 1.0, 0.0, 0.0))  # trigger bayes factor less than minimum threshold
    assert cube_node_monitor_hypothesis(cb, 0, return_new_sounding(8.0, 1.0, 0.0, 0.0))  # no intervention required
    assert not cube_node_monitor_hypothesis(cb, 0, return_new_sounding(8.0, 1.0, 0.0, 0.0))  # second monitor and the cum bayes fac is less than the threshold


def test_cube_node_reset_monitor():
    cb = return_new_cubenode()
    assert not cube_node_reset_monitor(cb, 0)  # failed with no hypotheses
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 0.5, 0.0, 0.0))
    assert cube_node_monitor_hypothesis(cb, 0, return_new_sounding(8.0, 1.0, 0.0, 0.0))  # no intervention required
    hypo = cb.hypotheses.get_item(0)
    assert hypo.cum_bayes_fac == approx(0.166, abs=0.001)
    assert hypo.seq_length == 1.0
    cube_node_reset_monitor(cb, 0)
    assert hypo.cum_bayes_fac == 1.0
    assert hypo.seq_length == 0


def test_cube_node_update_hypothesis():
    cb = return_new_cubenode()
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(6.0, 1.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(7.0, 1.0, 0.0, 0.0))

    hypo = cb.hypotheses.get_item(1)
    assert hypo.predict_depth == 6.0
    assert hypo.current_depth == 6.0
    assert hypo.current_variance == 1.0
    assert hypo.predict_variance == 1.0
    assert hypo.number_of_samples == 1

    cube_node_update_hypothesis(cb, 1, return_new_sounding(6.1, 0.9, 0.0, 0.0))
    assert hypo.predict_depth == approx(6.053, abs=0.001)
    assert hypo.current_depth == approx(6.053, abs=0.001)
    assert hypo.current_variance == approx(0.474, abs=0.001)
    assert hypo.predict_variance == approx(0.474, abs=0.001)
    assert hypo.number_of_samples == 2


def test_cube_node_best_hypothesis_index():
    cb = return_new_cubenode()
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(6.0, 1.0, 0.0, 0.0))
    cube_node_new_hypothesis(cb, return_new_sounding(7.0, 1.0, 0.0, 0.0))
    assert cube_node_best_hypothesis_index(cb, return_new_sounding(5.4, 1.0, 0.0, 0.0)) == 0
    assert cube_node_best_hypothesis_index(cb, return_new_sounding(5.6, 1.0, 0.0, 0.0)) == 1


def test_cube_node_update_node():
    cb = return_new_cubenode()
    cube_node_new_hypothesis(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_update_node(cb, return_new_sounding(5.1, 1.0, 0.0, 0.0))
    assert len(cb.hypotheses.get_data()) == 1
    assert cb.hypotheses.get_item(0).current_depth == approx(5.05, abs=0.01)
    assert cb.hypotheses.get_item(0).number_of_samples == 2

    cube_node_update_node(cb, return_new_sounding(15.1, 1.0, 0.0, 0.0))
    assert len(cb.hypotheses.get_data()) == 2


def test_cube_node_truncate():
    cb = return_new_cubenode()
    cb.queue = QueueList(return_new_sounding(6.0, 1.0, 0.0, 0.0), None)
    cb.queue.append(return_new_sounding(6.1, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(6.2, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(6.3, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(6.4, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(16.5, 2.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(36.6, 2.0, 0.0, 0.0))
    cb.n_queued = 7
    cube_node_truncate(cb)
    assert cb.n_queued == 6


def test_cube_node_queue_flush_node():
    cb = return_new_cubenode()
    cb.queue = QueueList(return_new_sounding(5.0, 1.0, 0.0, 0.0), None)
    cb.queue.append(return_new_sounding(5.1, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(5.3, 1.0, 0.0, 0.0))
    cb.n_queued = 4
    cube_node_queue_flush_node(cb)

    hypos = cb.hypotheses.get_data()
    assert len(hypos) == 1
    assert cb.n_queued == 0
    assert hypos[0].current_depth == approx(5.150, abs=0.001)
    assert hypos[0].current_variance == approx(0.25, abs=0.001)
    assert hypos[0].cum_bayes_fac == approx(1490.964, abs=0.001)
    assert hypos[0].number_of_samples == 4


def test_cube_node_choose_hypothesis():
    cb = return_new_cubenode()
    cb.queue = QueueList(return_new_sounding(5.0, 1.0, 0.0, 0.0), None)
    cb.queue.append(return_new_sounding(5.1, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cb.queue.append(return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cb.n_queued = 5
    cube_node_queue_flush_node(cb)
    hypo, ratio = cube_node_choose_hypothesis(cb)
    assert hypo.number_of_samples == 3
    assert ratio == 3.5


def test_cube_node_queue_fill():
    cb = return_new_cubenode()
    cube_node_queue_fill(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(5.5, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(4.5, 1.0, 0.0, 0.0))
    assert cb.n_queued == 6
    data = cb.queue.get_data()
    assert np.array_equal(data[0].depth, np.array(4.5, dtype=np.float32))
    assert np.array_equal(data[1].depth, np.array(5.0, dtype=np.float32))
    assert np.array_equal(data[2].depth, np.array(5.2, dtype=np.float32))
    assert np.array_equal(data[3].depth, np.array(5.5, dtype=np.float32))
    assert np.array_equal(data[4].depth, np.array(17.7, dtype=np.float32))
    assert np.array_equal(data[5].depth, np.array(17.8, dtype=np.float32))


def test_cube_node_add_to_queue():
    cb = return_new_cubenode()
    cube_node_add_to_queue(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(5.5, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(4.4, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(4.0, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(16.7, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(4.2, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(4.5, 1.0, 0.0, 0.0))
    cube_node_add_to_queue(cb, return_new_sounding(16.8, 1.0, 0.0, 0.0))
    assert cb.hypotheses is None
    # this should trigger update node, as you hit median length limit
    cube_node_add_to_queue(cb, return_new_sounding(4.6, 1.0, 0.0, 0.0))
    assert len(cb.hypotheses.get_data()) == 1


def test_cube_node_queue_insert():
    cb = return_new_cubenode()
    cube_node_queue_fill(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(17.7, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(5.2, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(5.5, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(17.8, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(4.4, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(4.0, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(16.7, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(4.2, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(4.5, 1.0, 0.0, 0.0))
    cube_node_queue_fill(cb, return_new_sounding(16.8, 1.0, 0.0, 0.0))
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]), atol=0.01)

    median_data = cube_node_queue_insert(cb, return_new_sounding(10.0, 1.0, 0.0, 0.0))
    assert np.allclose(np.array(5.2, dtype=np.float32), median_data.depth)
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.5, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01)

    median_data = cube_node_queue_insert(cb, return_new_sounding(10.0, 1.0, 0.0, 0.0))
    assert np.allclose(np.array(5.5, dtype=np.float32), median_data.depth)
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 10.0, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01)

    median_data = cube_node_queue_insert(cb, return_new_sounding(100.0, 1.0, 0.0, 0.0))  # this outlier will trigger truncation
    assert np.allclose(np.array(10.0, dtype=np.float32), median_data.depth)
    data = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01)


def test_cube_node_insert():
    cb = return_new_cubenode()  # predicted depth flagged
    cb.predicted_depth = np.float32(np.nan)
    handled = cube_node_insert(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0), 1.0)
    assert handled
    assert cb.n_queued == 0

    cb = return_new_cubenode()  # blunder
    cb.predicted_depth = 100.0
    handled = cube_node_insert(cb, return_new_sounding(50.0, 1.0, 0.0, 0.0), 1.0)
    assert handled
    assert cb.n_queued == 0

    cb = return_new_cubenode()  # too far
    handled = cube_node_insert(cb, return_new_sounding(5.0, 1.0, 0.0, 0.0), 1.0)
    assert handled
    assert cb.n_queued == 0

    cb = return_new_cubenode()
    handled = cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    assert handled
    assert cb.n_queued == 1

    cube_node_insert(cb, return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    assert cb.n_queued == 11
    assert cb.hypotheses is None

    dpths = [d.depth for d in cb.queue.get_data()]
    assert np.allclose(dpths, np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]), atol=0.01)
    varis = [d.variance for d in cb.queue.get_data()]
    assert np.allclose(varis, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), atol=0.01)


def test_cube_node_extract_depth_uncertainty():
    cb = return_new_cubenode()
    d, u, r = cube_node_extract_depth_uncertainty(cb)
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_depth_uncertainty(cb)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    cube_node_nominate_hypothesis(cb, 5.0)
    d, u, r = cube_node_extract_depth_uncertainty(cb)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_depth_uncertainty(cb)
    assert d == approx(4.686, abs=0.001)
    assert u == approx(0.524, abs=0.001)
    assert r == approx(3.25, abs=0.001)


def test_cube_node_extract_closest_depth_uncertainty():
    cb = return_new_cubenode()
    d, u, r = cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    cube_node_nominate_hypothesis(cb, 5.0)
    d, u, r = cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_closest_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(17.25, abs=0.001)
    assert u == approx(0.693, abs=0.001)
    assert r == approx(4.429, abs=0.001)


def test_cube_node_extract_posterior_depth_uncertainty():
    cb = return_new_cubenode()
    d, u, r = cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    cube_node_nominate_hypothesis(cb, 5.0)
    d, u, r = cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.4, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(4.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(16.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    d, u, r = cube_node_extract_posterior_depth_uncertainty(cb, 15.0, 0.5)
    assert d == approx(17.25, abs=0.001)
    assert u == approx(0.693, abs=0.001)
    assert r == approx(4.429, abs=0.001)


def test_cube_node_hypothesis_count():
    cb = return_new_cubenode()
    assert cube_node_hypothesis_count(cb) == 0

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    assert cube_node_hypothesis_count(cb) == 1

    cb = return_new_cubenode()
    cube_node_insert(cb, return_new_sounding(5.0, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.7, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.2, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(5.5, 0.0, 0.5, 0.5), 0.25)
    cube_node_insert(cb, return_new_sounding(17.8, 0.0, 0.5, 0.5), 0.25)
    cube_node_queue_flush_node(cb)
    assert cube_node_hypothesis_count(cb) == 2


def test_compile_now():
    compile_now()


def test_run_cube_gridding_variants():
    import pytest
    numpoints = 5
    x = np.array([101.0, 102.0, 103.0, 101.5, 102.5])
    y = np.array([199.0, 198.0, 197.0, 198.5, 197.5])
    z = np.array([10.0, 11.0, 10.5, 10.2, 10.8])
    tvu = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    thu = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    for method in ['local', 'posterior', 'prior', 'predicted']:
        d, u, r, n = run_cube_gridding(
            z, thu, tvu, x, y, 4, 4, 100.0, 200.0, method, 'order1a', 1.0, 1.0, dist_exponent=2.0
        )
        assert d.shape == (4, 4)

    with pytest.raises(NotImplementedError):
        run_cube_gridding(
            z, thu, tvu, x, y, 4, 4, 100.0, 200.0, 'unknown', 'order1a', 1.0, 1.0
        )


def test_numba_cube_py_funcs_coverage():
    # 1. get_iho_limits py_func
    assert get_iho_limits.py_func('exclusive') == (0.15, 0.0075)
    assert get_iho_limits.py_func('special') == (0.25, 0.0075)
    assert get_iho_limits.py_func('order1a') == (0.5, 0.013)
    assert get_iho_limits.py_func('order1b') == (0.5, 0.013)
    assert get_iho_limits.py_func('order2') == (1.0, 0.023)

    # 2. CubeParameters py_func and constructor
    params = return_default_cube_parameters.py_func('order1a', 1.0, 1.0)
    CubeParameters.class_type.methods['__init__'](params, 'order1a', np.float32(1.0), np.float32(1.0))

    # 3. Sounding py_func and constructor
    snd = return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5))
    Sounding.class_type.methods['__init__'](snd, np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5))

    # 4. Hypothesis py_func and constructor
    hyp = return_new_hypothesis.py_func(np.float32(10.0), np.float32(0.5))
    Hypothesis.class_type.methods['__init__'](hyp, np.float32(10.0), np.float32(0.5))

    # 5. HypothesisList methods
    hl_methods = HypothesisList.class_type.methods
    hl = HypothesisList(hyp, None)
    hl_methods['__init__'](hl, hyp, None)
    hl2 = hl_methods['prepend'](hl, return_new_hypothesis.py_func(np.float32(8.0), np.float32(0.5)))
    hl_methods['append'](hl2, return_new_hypothesis.py_func(np.float32(12.0), np.float32(0.5)))
    hl_methods['insert'](hl2, return_new_hypothesis.py_func(np.float32(9.0), np.float32(0.5)), 1)
    hl_methods['insert'](hl2, return_new_hypothesis.py_func(np.float32(7.0), np.float32(0.5)), 0)
    hl_methods['remove'](hl2, 1)
    hl_methods['remove'](hl2, 0)
    hl_dropped = hl_methods['drop_first'](hl2)
    data = hl_methods['get_data'](hl2)
    item = hl_methods['get_item'](hl2, 0)
    nearest_idx = hl_methods['get_nearest_in_depth'](hl2, 10.0, 0.5)
    min_err_idx = hl_methods['get_nearest_min_error'](hl2, 10.0, 0.5)
    max_s_idx, curmax, secmax = hl_methods['get_max_sample'](hl2)

    # 6. QueueList methods
    ql_methods = QueueList.class_type.methods
    ql = QueueList(snd, None)
    ql_methods['__init__'](ql, snd, None)
    ql2 = ql_methods['prepend'](ql, return_new_sounding.py_func(np.float32(8.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    ql_methods['append'](ql2, return_new_sounding.py_func(np.float32(12.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    ql_methods['insert'](ql2, return_new_sounding.py_func(np.float32(9.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)), 1)
    ql_methods['insert'](ql2, return_new_sounding.py_func(np.float32(7.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)), 0)
    ql_methods['remove'](ql2, 1)
    ql_methods['remove'](ql2, 0)
    ql_dropped = ql_methods['drop_first'](ql2)
    qdata = ql_methods['get_data'](ql2)
    qitem = ql_methods['get_item'](ql2, 0)

    # 7. CubeNode py_func and constructor
    node = return_new_cubenode.py_func()
    CubeNode.class_type.methods['__init__'](node)

    # 8. CubeGrid py_func and constructor
    grid = return_new_cubegrid.py_func(100.0, 200.0, 4, 4, np.float32(1.0), np.float32(1.0), params)
    CubeGrid.class_type.methods['__init__'](grid, 100.0, 200.0, 4, 4, np.float32(1.0), np.float32(1.0), params)

    # 9. CubeNode operations py_func
    cube_node_new_hypothesis.py_func(node, snd)
    cube_node_new_hypothesis.py_func(node, return_new_sounding.py_func(np.float32(10.005), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    assert cube_node_hypothesis_count.py_func(node) == 2

    # nomination py_funcs
    assert cube_node_nominate_hypothesis.py_func(node, np.float32(10.0))
    assert cube_node_is_nominated.py_func(node)
    d, u, r = cube_node_get_nominated_depth_uncertainty.py_func(node)
    assert d == approx(10.0, abs=0.01)
    assert cube_node_reset_nomination.py_func(node)
    assert not cube_node_is_nominated.py_func(node)
    cube_node_get_nominated_depth_uncertainty.py_func(node)

    # remove hypothesis
    cube_node_nominate_hypothesis.py_func(node, np.float32(10.0))
    cube_node_remove_hypothesis.py_func(node, np.float32(10.0))
    cube_node_remove_hypothesis.py_func(node, np.float32(99.0))

    # predicted depth
    cube_node_set_preddepth.py_func(node, return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))

    # monitor hypothesis & reset monitor
    node2 = return_new_cubenode.py_func()
    assert not cube_node_monitor_hypothesis.py_func(node2, 0, snd)
    assert not cube_node_reset_monitor.py_func(node2, 0)
    cube_node_new_hypothesis.py_func(node2, return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_monitor_hypothesis.py_func(node2, 0, return_new_sounding.py_func(np.float32(12.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_reset_monitor.py_func(node2, 0)

    # update hypothesis & choose hypothesis & best hypothesis index
    cube_node_update_hypothesis.py_func(node2, 0, return_new_sounding.py_func(np.float32(10.1), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_best_hypothesis_index.py_func(node2, snd)
    best_h, best_r = cube_node_choose_hypothesis.py_func(node2)

    # update node
    cube_node_update_node.py_func(node2, return_new_sounding.py_func(np.float32(10.2), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_update_node.py_func(node2, return_new_sounding.py_func(np.float32(25.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))

    # queue operations
    node_q = return_new_cubenode.py_func()
    for d in [4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]:
        cube_node_queue_fill.py_func(node_q, return_new_sounding.py_func(np.float32(d), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_queue_insert.py_func(node_q, return_new_sounding.py_func(np.float32(6.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_queue_insert.py_func(node_q, return_new_sounding.py_func(np.float32(4.1), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_truncate.py_func(node_q)
    cube_node_queue_flush_node.py_func(node_q)

    # cube_node_insert filtering branches
    node_f = return_new_cubenode.py_func()
    node_f.predicted_depth = np.float32(np.nan)
    assert cube_node_insert.py_func(node_f, snd, np.float32(0.25))
    assert node_f.n_queued == 0

    node_f.predicted_depth = np.float32(20.0)
    node_f.predicted_variance = np.float32(0.5)
    node_f.blunder_min = np.float32(5.0)
    node_f.blunder_percent = np.float32(0.25)
    assert cube_node_insert.py_func(node_f, return_new_sounding.py_func(np.float32(2.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)), np.float32(0.25))
    assert node_f.n_queued == 0

    node_f.predicted_depth = np.float32(1.0)
    node_f.capture_dist_scale = np.float32(0.01)
    assert cube_node_insert.py_func(node_f, snd, np.float32(100.0))
    assert node_f.n_queued == 0


    # generic hypothesis extraction & variance_selection
    node_ext = return_new_cubenode.py_func()
    h_ext = return_new_hypothesis.py_func(np.float32(10.0), np.float32(0.5))
    h_ext.variance_estimate = np.float32(1.0)
    node_ext.variance_selection = 'max'
    cube_node_get_generic_hypothesis_depth_uncertainty.py_func(node_ext, h_ext)
    node_ext.variance_selection = 'input'
    cube_node_get_generic_hypothesis_depth_uncertainty.py_func(node_ext, h_ext)
    h_zero = return_new_hypothesis.py_func(np.float32(10.0), np.float32(0.5))
    h_zero.number_of_samples = 0
    cube_node_get_generic_hypothesis_depth_uncertainty.py_func(node_ext, h_zero)

    # extract_depth_uncertainty branches (nominated, empty, single, multiple)
    node_e = return_new_cubenode.py_func()
    cube_node_extract_depth_uncertainty.py_func(node_e)
    cube_node_new_hypothesis.py_func(node_e, snd)
    cube_node_extract_depth_uncertainty.py_func(node_e)
    cube_node_new_hypothesis.py_func(node_e, return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_extract_depth_uncertainty.py_func(node_e)
    cube_node_nominate_hypothesis.py_func(node_e, np.float32(10.0))
    cube_node_extract_depth_uncertainty.py_func(node_e)

    # extract_closest_depth_uncertainty & extract_posterior_depth_uncertainty
    cube_node_extract_closest_depth_uncertainty.py_func(node_e, np.float32(10.0), np.float32(0.5))
    cube_node_extract_posterior_depth_uncertainty.py_func(node_e, np.float32(10.0), np.float32(0.5))
    node_e.nominated = None
    cube_node_extract_closest_depth_uncertainty.py_func(node_e, np.float32(10.0), np.float32(0.5))
    cube_node_extract_posterior_depth_uncertainty.py_func(node_e, np.float32(10.0), np.float32(0.5))

    # remove hypothesis branches
    node_rem = return_new_cubenode.py_func()
    assert not cube_node_remove_hypothesis.py_func(node_rem, np.float32(10.0))
    cube_node_new_hypothesis.py_func(node_rem, snd)
    cube_node_nominate_hypothesis.py_func(node_rem, np.float32(10.0))
    assert cube_node_remove_hypothesis.py_func(node_rem, np.float32(10.0))

    node_rem2 = return_new_cubenode.py_func()
    cube_node_new_hypothesis.py_func(node_rem2, snd)
    cube_node_new_hypothesis.py_func(node_rem2, return_new_sounding.py_func(np.float32(15.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_new_hypothesis.py_func(node_rem2, return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    assert cube_node_remove_hypothesis.py_func(node_rem2, np.float32(15.0))

    # nominate hypothesis when None
    node_nom_empty = return_new_cubenode.py_func()
    assert not cube_node_nominate_hypothesis.py_func(node_nom_empty, np.float32(10.0))

    # monitor hypothesis branches (runlength and variance test)
    node_mon = return_new_cubenode.py_func()
    assert not cube_node_monitor_hypothesis.py_func(node_mon, 0, snd)
    cube_node_new_hypothesis.py_func(node_mon, return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    # Monitor with variance threshold failure
    cube_node_monitor_hypothesis.py_func(node_mon, 0, return_new_sounding.py_func(np.float32(100.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    # Trigger runlength failure
    for _ in range(6):
        cube_node_monitor_hypothesis.py_func(node_mon, 0, return_new_sounding.py_func(np.float32(10.5), np.float32(0.5), np.float32(0.5), np.float32(0.5)))

    # update hypothesis variance modes
    node_upd = return_new_cubenode.py_func()
    assert not cube_node_update_hypothesis.py_func(node_upd, 0, snd)
    cube_node_new_hypothesis.py_func(node_upd, return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    node_upd.variance_selection = 'max'
    cube_node_update_hypothesis.py_func(node_upd, 0, return_new_sounding.py_func(np.float32(10.1), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    node_upd.variance_selection = 'input'
    cube_node_update_hypothesis.py_func(node_upd, 0, return_new_sounding.py_func(np.float32(10.1), np.float32(0.5), np.float32(0.5), np.float32(0.5)))

    # best hypothesis index when None
    assert cube_node_best_hypothesis_index.py_func(return_new_cubenode.py_func(), snd) == -1

    # truncate branches (<3 points, None queue, outlier quotient)
    node_tr = return_new_cubenode.py_func()
    cube_node_truncate.py_func(node_tr)
    node_tr.n_queued = 5
    node_tr.queue = None
    cube_node_truncate.py_func(node_tr)

    node_outlier = return_new_cubenode.py_func()
    for d in [4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8, 100.0]:
        cube_node_queue_fill.py_func(node_outlier, return_new_sounding.py_func(np.float32(d), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    node_outlier.quotient_limit = 0.01
    cube_node_truncate.py_func(node_outlier)


    # queue flush branches (0 queued, odd queued)
    node_q_odd = return_new_cubenode.py_func()
    cube_node_queue_flush_node.py_func(node_q_odd)
    for d in [4.0, 5.0, 6.0]:
        cube_node_queue_fill.py_func(node_q_odd, return_new_sounding.py_func(np.float32(d), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_queue_flush_node.py_func(node_q_odd)

    # choose hypothesis when None
    h_none, r_none = cube_node_choose_hypothesis.py_func(return_new_cubenode.py_func())
    assert h_none is None

    # HypothesisList get_item and get_max_sample branches
    h_a = return_new_hypothesis.py_func(np.float32(5.0), np.float32(0.5))
    h_a.number_of_samples = 2
    h_b = return_new_hypothesis.py_func(np.float32(6.0), np.float32(0.5))
    h_b.number_of_samples = 10
    h_c = return_new_hypothesis.py_func(np.float32(7.0), np.float32(0.5))
    h_c.number_of_samples = 5
    hl_branch = HypothesisList(h_a, HypothesisList(h_b, HypothesisList(h_c, None)))
    assert HypothesisList.class_type.methods['get_item'](hl_branch, 1).current_depth == 6.0
    assert HypothesisList.class_type.methods['get_item'](hl_branch, 10) is None
    idx_m, cur_m, sec_m = HypothesisList.class_type.methods['get_max_sample'](hl_branch)
    assert idx_m == 1 and cur_m == 10 and sec_m == 5

    # QueueList get_item branch
    ql_branch = QueueList(snd, QueueList(return_new_sounding.py_func(np.float32(12.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)), None))
    assert QueueList.class_type.methods['get_item'](ql_branch, 1).depth == 12.0
    assert QueueList.class_type.methods['get_item'](ql_branch, 10) is None

    # add_to_queue branches (n_queued >= median_length)

    node_add_q = return_new_cubenode.py_func()
    for d in [4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]:
        cube_node_add_to_queue.py_func(node_add_q, return_new_sounding.py_func(np.float32(d), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_add_to_queue.py_func(node_add_q, return_new_sounding.py_func(np.float32(6.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))

    # hypothesis count when None
    assert cube_node_hypothesis_count.py_func(return_new_cubenode.py_func()) == 0

    # extract closest / posterior when nearest_hypo is None (all 0 samples)
    node_all_zero = return_new_cubenode.py_func()
    h_z1 = return_new_hypothesis.py_func(np.float32(10.0), np.float32(0.5))
    h_z1.number_of_samples = 0
    h_z2 = return_new_hypothesis.py_func(np.float32(20.0), np.float32(0.5))
    h_z2.number_of_samples = 0
    node_all_zero.hypotheses = HypothesisList(h_z1, HypothesisList(h_z2, None))
    d_z, u_z, r_z = cube_node_extract_closest_depth_uncertainty.py_func(node_all_zero, np.float32(10.0), np.float32(0.5))
    assert np.isnan(d_z)
    d_zp, u_zp, r_zp = cube_node_extract_posterior_depth_uncertainty.py_func(node_all_zero, np.float32(10.0), np.float32(0.5))
    assert np.isnan(d_zp)

    # 10. CubeGrid insert and extract py_func
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
            self.grid = [[return_new_cubenode.py_func() for _ in range(num_columns)] for _ in range(num_rows)]

    grid2 = MockGrid(4, 4, params)
    z = np.array([10.0, 15.0, 10.0], dtype=np.float32)
    thu = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    tvu = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    x = np.array([101.5, 500.0, 102.5], dtype=np.float64)
    y = np.array([198.5, 500.0, 197.5], dtype=np.float64)
    cube_grid_insert_points.py_func(grid2, z, thu, tvu, x, y)

    for m in ['local', 'posterior', 'prior', 'predicted']:
        cube_grid_extract_data.py_func(grid2, m)

    # spatial row and column search in extract_data py_func
    grid3 = MockGrid(5, 5, params)
    n_multi = grid3.grid[2][2]
    cube_node_new_hypothesis.py_func(n_multi, return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_new_hypothesis.py_func(n_multi, return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    n_single = grid3.grid[2][3]
    cube_node_new_hypothesis.py_func(n_single, return_new_sounding.py_func(np.float32(10.2), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_grid_extract_data.py_func(grid3, 'local')
    cube_grid_extract_data.py_func(grid3, 'posterior')

    # column search in extract_data py_func
    grid_col = MockGrid(5, 5, params)
    n_multi_col = grid_col.grid[2][2]
    cube_node_new_hypothesis.py_func(n_multi_col, return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_new_hypothesis.py_func(n_multi_col, return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    n_single_col = grid_col.grid[3][2]  # placed in target_cols offset
    cube_node_new_hypothesis.py_func(n_single_col, return_new_sounding.py_func(np.float32(10.2), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_grid_extract_data.py_func(grid_col, 'local')

    # MockGrid with no neighbors found (defaults to extract_depth_uncertainty)
    grid_none = MockGrid(5, 5, params)
    n_multi_none = grid_none.grid[2][2]
    cube_node_new_hypothesis.py_func(n_multi_none, return_new_sounding.py_func(np.float32(10.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_node_new_hypothesis.py_func(n_multi_none, return_new_sounding.py_func(np.float32(20.0), np.float32(0.5), np.float32(0.5), np.float32(0.5)))
    cube_grid_extract_data.py_func(grid_none, 'local')





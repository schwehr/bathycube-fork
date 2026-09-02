from pytest import approx

from bathycube.cube import *


def test_cube_params():
    param = CubeParameters()
    param.initialize("order1a", 0.5, 0.5)
    assert param.grid_resolution_x == 0.5
    assert param.grid_resolution_y == 0.5
    assert param.inv_dist_exponent == 1 / 2.0
    assert param.iho_order == "order1a"


def test_cube_node_init():
    cb = CubeNode()
    cb.predicted_depth = 1.0
    assert cb.predicted_depth == 1.0
    assert cb.predicted_variance == 0.0


def test_cube_node_new_hypothesis():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 5.0)
    cb.add_hypothesis(6.0, 6.0)
    data = cb.hypotheses
    assert data[0].current_depth == 5.0
    assert data[1].current_depth == 6.0


def test_cube_node_remove_hypothesis():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 5.0)
    cb.remove_hypothesis(99.0)
    assert len(cb.hypotheses) == 1

    cb.remove_hypothesis(5.001)
    assert len(cb.hypotheses) == 0

    cb.add_hypothesis(5.0, 5.0)
    cb.add_hypothesis(6.0, 6.0)
    cb.add_hypothesis(7.0, 7.0)
    cb.add_hypothesis(8.0, 8.0)
    cb.remove_hypothesis(6.001)
    cb.remove_hypothesis(7.999)
    cb.remove_hypothesis(5.001)
    data = cb.hypotheses
    assert len(data) == 1
    assert data[0].current_depth == 7.0


def test_cube_node_nominate_hypothesis():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 5.0)
    cb.add_hypothesis(6.0, 6.0)
    cb.add_hypothesis(7.0, 7.0)
    cb.add_hypothesis(8.0, 8.0)

    assert cb.nominated is None
    cb.nominate_hypothesis(5.001)
    assert cb.nominated.current_depth == 5.0

    cb.nominate_hypothesis(4.999)
    assert cb.nominated.current_depth == 5.0

    cb.nominate_hypothesis(7.001)
    assert cb.nominated.current_depth == 7.0

    cb.nominate_hypothesis(7.5)
    assert cb.nominated is None


def test_cube_node_reset_nomination():
    cb = CubeNode()
    cb.clear_nomination()
    assert cb.nominated is None

    cb.add_hypothesis(8.0, 8.0)
    cb.nominate_hypothesis(8.001)
    cb.clear_nomination()
    assert cb.nominated is None


def test_cube_node_is_nominated():
    cb = CubeNode()
    assert not cb.has_nomination()
    assert cb.nominated is None

    cb.add_hypothesis(8.0, 8.0)
    cb.nominate_hypothesis(8.001)
    assert cb.has_nomination()


def test_cube_node_set_preddepth():
    cb = CubeNode()
    assert cb.predicted_depth == 0.0
    assert cb.predicted_variance == 0.0
    cb.predicted_depth = 5.0
    cb.predicted_variance = 1.5
    assert cb.predicted_depth == 5.0
    assert cb.predicted_variance == 1.5


def test_cube_node_monitor_hypothesis():
    cb = CubeNode()
    assert not cb.monitor_hypothesis(0, 1.0, 1.0)
    cb.add_hypothesis(5.0, 0.5)
    assert not cb.monitor_hypothesis(0, 10.0, 1.0)
    assert cb.monitor_hypothesis(0, 8.0, 1.0)  # no intervention required
    assert not cb.monitor_hypothesis(0, 8.0, 1.0)  # second monitor and the cum bayes fac is less than the threshold


def test_cube_node_reset_monitor():
    cb = CubeNode()
    assert not cb.reset_monitor(0)  # failed with no hypotheses
    cb.add_hypothesis(5.0, 0.5)
    assert cb.monitor_hypothesis(0, 8.0, 1.0)  # no intervention required
    hypo = cb.hypotheses[0]
    assert hypo.cum_bayes_fac == approx(0.166, abs=0.001)
    assert hypo.seq_length == 1.0
    cb.reset_monitor(0)
    assert hypo.cum_bayes_fac == 1.0
    assert hypo.seq_length == 0


def test_cube_node_update_hypothesis():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 1.0)
    cb.add_hypothesis(6.0, 1.0)
    cb.add_hypothesis(7.0, 1.0)

    hypo = cb.hypotheses[1]
    assert hypo.predict_depth == 6.0
    assert hypo.current_depth == 6.0
    assert hypo.current_variance == 1.0
    assert hypo.predict_variance == 1.0
    assert hypo.number_of_points == 1

    cb.update_hypothesis(1, 6.1, 0.9)
    assert hypo.predict_depth == approx(6.053, abs=0.001)
    assert hypo.current_depth == approx(6.053, abs=0.001)
    assert hypo.current_variance == approx(0.474, abs=0.001)
    assert hypo.predict_variance == approx(0.474, abs=0.001)
    assert hypo.number_of_points == 2


def test_cube_node_best_hypothesis_index():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 1.0)
    cb.add_hypothesis(6.0, 1.0)
    cb.add_hypothesis(7.0, 1.0)

    assert cb.best_hypothesis_index(5.4, 1.0) == 0
    assert cb.best_hypothesis_index(5.6, 1.0) == 1


def test_cube_node_update_node():
    cb = CubeNode()
    cb.add_hypothesis(5.0, 1.0)
    cb.update_node(5.1, 1.0)

    assert len(cb.hypotheses) == 1
    assert cb.hypotheses[0].current_depth == approx(5.05, abs=0.01)
    assert cb.hypotheses[0].number_of_points == 2

    cb.update_node(15.1, 1.0)
    assert len(cb.hypotheses) == 2


def test_cube_node_truncate():
    cb = CubeNode()
    cb.add_to_queue(6.0, 1.0)
    cb.add_to_queue(6.1, 1.0)
    cb.add_to_queue(6.2, 1.0)
    cb.add_to_queue(6.3, 1.0)
    cb.add_to_queue(6.4, 1.0)
    cb.add_to_queue(16.5, 2.0)
    cb.add_to_queue(36.6, 2.0)
    assert cb.n_queued == 7
    cb.truncate()
    assert cb.n_queued == 6


def test_cube_node_queue_flush_node():
    cb = CubeNode()
    cb.add_to_queue(5.0, 1.0)
    cb.add_to_queue(5.1, 1.0)
    cb.add_to_queue(5.2, 1.0)
    cb.add_to_queue(5.3, 1.0)
    cb.n_queued = 4
    cb.flush_queue()

    hypos = cb.hypotheses
    assert len(hypos) == 1
    assert cb.n_queued == 0
    assert hypos[0].current_depth == approx(5.150, abs=0.001)
    assert hypos[0].current_variance == approx(0.25, abs=0.001)
    assert hypos[0].cum_bayes_fac == approx(1490.966, abs=0.001)
    assert hypos[0].number_of_points == 4


def test_cube_node_choose_hypothesis():
    cb = CubeNode()
    cb.add_to_queue(5.0, 1.0)
    cb.add_to_queue(5.1, 1.0)
    cb.add_to_queue(5.2, 1.0)
    cb.add_to_queue(17.7, 1.0)
    cb.add_to_queue(17.8, 1.0)
    cb.n_queued = 5
    cb.flush_queue()

    hypo, ratio = cb.choose_hypothesis()
    assert hypo.number_of_points == 3
    assert ratio == 3.5


def test_cube_node_queue_fill():
    cb = CubeNode()
    cb.add_to_queue(5.0, 1.0)
    cb.add_to_queue(17.7, 1.0)
    cb.add_to_queue(5.2, 1.0)
    cb.add_to_queue(5.5, 1.0)
    cb.add_to_queue(17.8, 1.0)
    cb.add_to_queue(4.5, 1.0)

    assert cb.n_queued == 6
    data = cb.queue
    assert np.allclose(data[0][0], np.array(4.5, dtype=np.float32))
    assert np.allclose(data[1][0], np.array(5.0, dtype=np.float32))
    assert np.allclose(data[2][0], np.array(5.2, dtype=np.float32))
    assert np.allclose(data[3][0], np.array(5.5, dtype=np.float32))
    assert np.allclose(data[4][0], np.array(17.7, dtype=np.float32))
    assert np.allclose(data[5][0], np.array(17.8, dtype=np.float32))


def test_cube_node_add_to_queue():
    cb = CubeNode()
    cb.add_to_queue(5.0, 1.0)
    cb.add_to_queue(17.7, 1.0)
    cb.add_to_queue(5.2, 1.0)
    cb.add_to_queue(5.5, 1.0)
    cb.add_to_queue(17.8, 1.0)
    cb.add_to_queue(4.4, 1.0)
    cb.add_to_queue(4.0, 1.0)
    cb.add_to_queue(16.7, 1.0)
    cb.add_to_queue(4.2, 1.0)
    cb.add_to_queue(4.5, 1.0)
    cb.add_to_queue(16.8, 1.0)

    assert cb.hypotheses == []
    # this should trigger update node, as you hit median length limit
    cb.add_to_queue(4.6, 1.0)
    assert len(cb.hypotheses) == 1


def test_cube_node_queue_insert():
    cb = CubeNode()
    cb.add_to_queue(5.0, 1.0)
    cb.add_to_queue(17.7, 1.0)
    cb.add_to_queue(5.2, 1.0)
    cb.add_to_queue(5.5, 1.0)
    cb.add_to_queue(17.8, 1.0)
    cb.add_to_queue(4.4, 1.0)
    cb.add_to_queue(4.0, 1.0)
    cb.add_to_queue(16.7, 1.0)
    cb.add_to_queue(4.2, 1.0)
    cb.add_to_queue(4.5, 1.0)
    cb.add_to_queue(16.8, 1.0)
    data = [d[0] for d in cb.queue]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]), atol=0.01)

    median_data = cb.queue_insert(10.0, 1.0)
    assert np.allclose(np.array(5.2, dtype=np.float32), median_data[0])
    data = [d[0] for d in cb.queue]
    assert np.allclose(
        np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.5, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01
    )

    median_data = cb.queue_insert(10.0, 1.0)
    assert np.allclose(np.array(5.5, dtype=np.float32), median_data[0])
    data = [d[0] for d in cb.queue]
    assert np.allclose(
        np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 10.0, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01
    )

    median_data = cb.queue_insert(100.0, 1.0)  # this outlier will trigger truncation
    assert np.allclose(np.array(10.0, dtype=np.float32), median_data[0])
    data = [d[0] for d in cb.queue]
    assert np.allclose(np.array(data), np.array([4.0, 4.2, 4.4, 4.5, 5.0, 10.0, 16.7, 16.8, 17.7, 17.8]), atol=0.01)


def test_cube_add_point_to_node():
    cb = CubeNode()  # predicted depth flagged
    cb.predicted_depth = np.float32(np.nan)
    cb.add_point_to_node(5.0, 0.0, 0.0, 1.0)
    assert cb.n_queued == 0

    cb = CubeNode()  # blunder
    cb.predicted_depth = 100.0
    cb.add_point_to_node(50.0, 0.0, 0.0, 1.0)
    assert cb.n_queued == 0

    cb = CubeNode()  # too far
    cb.add_point_to_node(5.0, 0.0, 0.0, 1.0)
    assert cb.n_queued == 0

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    assert cb.n_queued == 1

    cb.add_point_to_node(17.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.8, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.4, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.8, 0.5, 0.5, 0.25)
    assert cb.n_queued == 11
    assert cb.hypotheses == []

    dpths = [d[0] for d in cb.queue]
    assert np.allclose(dpths, np.array([4.0, 4.2, 4.4, 4.5, 5.0, 5.2, 5.5, 16.7, 16.8, 17.7, 17.8]), atol=0.01)
    varis = [d[1] for d in cb.queue]
    assert np.allclose(varis, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), atol=0.01)


def test_cube_node_extract_depth_uncertainty():
    cb = CubeNode()
    d, u, r = cb.extract_node_value(("depth", "uncertainty", "ratio"))
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_node_value(("depth", "uncertainty", "ratio"))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    cb.nominate_hypothesis(5.0)
    d, u, r = cb.extract_node_value(("depth", "uncertainty", "ratio"))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.8, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.4, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.8, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_node_value(("depth", "uncertainty", "ratio"))
    assert d == approx(4.686, abs=0.001)
    assert u == approx(0.524, abs=0.001)
    assert r == approx(3.25, abs=0.001)


def test_cube_node_extract_closest_depth_uncertainty():
    cb = CubeNode()
    d, u, r = cb.extract_closest_node_value(15.0, 0.5, ("depth", "uncertainty", "ratio"))
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_closest_node_value(15.0, 0.5, ("depth", "uncertainty", "ratio"))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    cb.nominate_hypothesis(5.0)
    d, u, r = cb.extract_closest_node_value(15.0, 0.5, ("depth", "uncertainty", "ratio"))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.8, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.4, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.8, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_closest_node_value(15.0, 0.5, ("depth", "uncertainty", "ratio"))
    assert d == approx(17.25, abs=0.001)
    assert u == approx(0.693, abs=0.001)
    assert r == approx(4.429, abs=0.001)


def test_cube_node_extract_posterior_depth_uncertainty():
    cb = CubeNode()
    d, u, r = cb.extract_posterior_weighted_node_value(15.0, 0.5, ("depth", "uncertainty", "ratio"))
    assert np.isnan(d)
    assert np.isnan(u)
    assert np.isnan(r)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_posterior_weighted_node_value(15.0, 0.5, ("depth", "uncertainty", "ratio"))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    cb.nominate_hypothesis(5.0)
    d, u, r = cb.extract_posterior_weighted_node_value(15.0, 0.5, ("depth", "uncertainty", "ratio"))
    assert d == approx(5.0, abs=0.001)
    assert u == approx(1.385, abs=0.001)
    assert r == approx(0.0, abs=0.001)

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.8, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.4, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(4.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(16.8, 0.5, 0.5, 0.25)
    cb.flush_queue()
    d, u, r = cb.extract_posterior_weighted_node_value(15.0, 0.5, ("depth", "uncertainty", "ratio"))
    assert d == approx(17.25, abs=0.001)
    assert u == approx(0.693, abs=0.001)
    assert r == approx(4.429, abs=0.001)


def test_cube_node_hypothesis_count():
    cb = CubeNode()
    assert cb.return_number_of_hypotheses() == 0

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.flush_queue()
    assert cb.return_number_of_hypotheses() == 1

    cb = CubeNode()
    cb.add_point_to_node(5.0, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.7, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.2, 0.5, 0.5, 0.25)
    cb.add_point_to_node(5.5, 0.5, 0.5, 0.25)
    cb.add_point_to_node(17.8, 0.5, 0.5, 0.25)
    cb.flush_queue()
    assert cb.return_number_of_hypotheses() == 2


def test_get_iho_limits_all_orders():
    assert get_iho_limits("exclusive") == (0.15, 0.0075)
    assert get_iho_limits("special") == (0.25, 0.0075)
    assert get_iho_limits("order1a") == (0.5, 0.013)
    assert get_iho_limits("order1b") == (0.5, 0.013)
    assert get_iho_limits("order2") == (1.0, 0.023)


def test_cube_parameters_io(tmp_path):
    import pytest

    param = CubeParameters()
    param.initialize("order1a", 1.0, 1.0)
    param.no_data_value = float(np.nan)
    filepath = str(tmp_path / "params.json")
    param.write_parameter_file(filepath)

    param2 = CubeParameters()
    param2.open_parameter_file(filepath)
    assert param2.grid_resolution_x == 1.0
    assert param2.grid_resolution_y == 1.0

    # Test error handling on write
    with pytest.raises(ValueError):
        param.write_parameter_file("/nonexistent_directory/params.json")

    # Test error handling on read invalid json
    bad_json = str(tmp_path / "bad.json")
    with open(bad_json, "w") as f:
        f.write("invalid json content")
    with pytest.raises(ValueError):
        param2.open_parameter_file(bad_json)

    # Test read json with no matching keys
    empty_json = str(tmp_path / "empty.json")
    with open(empty_json, "w") as f:
        f.write(json.dumps({"unknown_key": 123}))
    param2.open_parameter_file(empty_json)


def test_cube_node_additional_branches():
    import pytest

    cb = CubeNode()
    # Null hypothesis
    cb.add_hypothesis(10.0, 0.5, null_hypothesis=True)
    assert cb.hypotheses[0].number_of_points == 0
    # dump hypotheses
    cb.dump_hypotheses()

    # Extract answer from null hypothesis
    ans = cb._return_answer_from_hypothesis(cb.hypotheses[0], 1.0, ("depth", "uncertainty", "ratio", "n_hypotheses"))
    assert np.isnan(ans[0])
    assert np.isnan(ans[1])

    # Multiple matching hypotheses on remove raises ValueError
    cb2 = CubeNode()
    cb2.add_hypothesis(10.0, 0.5)
    cb2.add_hypothesis(10.005, 0.5)
    with pytest.raises(ValueError):
        cb2.remove_hypothesis(10.002)

    # Nominate hypothesis and then remove it
    cb3 = CubeNode()
    cb3.add_hypothesis(10.0, 0.5)
    cb3.nominate_hypothesis(10.0)
    assert cb3.nominated is not None
    cb3.remove_hypothesis(10.0)
    assert cb3.nominated is None

    # Nominate hypothesis with multiple candidates picking the closest
    cb4 = CubeNode()
    cb4.add_hypothesis(10.002, 0.5)
    cb4.add_hypothesis(10.006, 0.5)
    cb4.nominate_hypothesis(10.007)
    assert cb4.nominated.current_depth == 10.006

    # Return nominated answer with all value ids
    nom_ans = cb4._return_nominated_answer(("depth", "uncertainty", "ratio", "n_hypotheses"))
    assert nom_ans[0] == 10.006
    assert nom_ans[2] == 0.0
    assert nom_ans[3] == 2

    # Variance selection options ('max', 'input')
    cb5 = CubeNode()
    cb5.variance_selection = "max"
    cb5.add_hypothesis(10.0, 0.5)
    cb5.hypotheses[0].variance_estimate = 1.0
    ans_max = cb5.extract_node_value(("depth", "uncertainty", "ratio", "n_hypotheses"))
    assert ans_max[0] == 10.0

    cb5.variance_selection = "input"
    ans_input = cb5.extract_node_value(("depth", "uncertainty"))
    assert ans_input[0] == 10.0

    # return_depth and return_uncertainty
    assert cb5.return_depth() == 10.0
    assert ans_input[1] == cb5.return_uncertainty()

    # Choose hypothesis with second highest count branch
    cb6 = CubeNode()
    h1 = Hypothesis(10.0, 0.5)
    h1.number_of_points = 5
    h2 = Hypothesis(15.0, 0.5)
    h2.number_of_points = 3
    h3 = Hypothesis(20.0, 0.5)
    h3.number_of_points = 4
    cb6.hypotheses = [h1, h2, h3]
    best_h, ratio = cb6.choose_hypothesis()
    assert best_h == h1

    # Queue methods with use_queue = False or empty queue flush
    cb_no_q = CubeNode(use_queue=False)
    cb_no_q.queue_fill(10.0, 0.5)
    d, v = cb_no_q.queue_insert(10.0, 0.5)
    assert d == 10.0 and v == 0.5
    cb_no_q.flush_queue()

    cb_empty = CubeNode()
    cb_empty.flush_queue()


def test_cube_node_point_filtering():
    # Sounding rejected with NaN predicted depth
    cb = CubeNode()
    cb.predicted_depth = np.nan
    cb.add_point_to_node(10.0, 0.5, 0.5, 0.25)
    assert len(cb.queue) == 0

    # Sounding rejected as blunder
    cb2 = CubeNode()
    cb2.predicted_depth = 20.0
    cb2.predicted_variance = 0.5
    cb2.blunder_min = 5.0
    cb2.blunder_percent = 0.25
    cb2.add_point_to_node(2.0, 0.5, 0.5, 0.25)
    assert len(cb2.queue) == 0

    # Sounding rejected due to capture distance
    cb3 = CubeNode()
    cb3.predicted_depth = 1.0
    cb3.capture_dist_scale = 0.01
    # dist is sqrt(100) = 10m, max capture distance is max(0.01 * 1, 0.5) = 0.5m
    cb3.add_point_to_node(1.0, 0.5, 0.5, 100.0)
    assert len(cb3.queue) == 0


def test_cube_grid_and_gridding(tmp_path):
    import pytest

    param = CubeParameters()
    param.initialize("order1a", 1.0, 1.0)
    logfile = str(tmp_path / "test_cube.log")
    grid = CubeGrid(
        minimum_easting=100.0,
        maximum_northing=200.0,
        num_columns=4,
        num_rows=4,
        resolution_x=1.0,
        resolution_y=1.0,
        param=param,
        use_queue=True,
        logfile=logfile,
        debug=True,
    )

    assert grid.total_nodes_count == 16
    assert grid.empty_nodes_count == 16
    assert grid.populated_nodes_count == 0

    # Validate insert points with scalar, list, and mismatched arrays
    d, h, v, e, n = grid._validate_insert_points(10.0, 0.5, 0.5, 101.5, 198.5)
    assert isinstance(d, np.ndarray) and len(d) == 1

    d, h, v, e, n = grid._validate_insert_points([10.0, 11.0], [0.5, 0.5], [0.5, 0.5], [101.5, 102.5], [198.5, 197.5])
    assert len(d) == 2

    with pytest.raises(AssertionError):
        grid._validate_insert_points([10.0], [0.5, 0.5], [0.5], [101.5], [198.5])

    # Insert points including out of bounds soundings
    depths = np.array([10.0, 15.0, 10.0])
    thu = np.array([0.5, 0.5, 0.5])
    tvu = np.array([0.5, 0.5, 0.5])
    eastings = np.array([101.5, 500.0, 102.5])  # 500.0 is out of bounds
    northings = np.array([198.5, 500.0, 197.5])

    grid.insert_points(depths, thu, tvu, eastings, northings)
    grid.flush_node_queues()

    assert grid.populated_nodes_count > 0

    # Test all grid extraction methods
    for method in ["local", "posterior", "prior", "predicted"]:
        depth_grid = grid.get_grid_depth(method=method)
        unc_grid = grid.get_grid_uncertainty(method=method)
        ratio_grid = grid.get_grid_ratio(method=method)
        d_and_u = grid.get_grid_depth_and_uncertainty(method=method)
        assert depth_grid.shape == (4, 4)
        assert unc_grid.shape == (4, 4)
        assert ratio_grid.shape == (4, 4)
        assert len(d_and_u) == 2

    hyp_count_grid = grid.get_grid_number_hypotheses()
    assert hyp_count_grid.shape == (4, 4)

    # Test run_cube_gridding with valid methods and custom kwargs
    for method in ["local", "posterior", "prior", "predicted"]:
        dg, ug, rg, ng = run_cube_gridding(
            depth=np.array([10.0, 10.5]),
            horizontal_uncertainty=np.array([0.5, 0.5]),
            vertical_uncertainty=np.array([0.5, 0.5]),
            easting=np.array([101.5, 101.5]),
            northing=np.array([198.5, 198.5]),
            num_columns=4,
            num_rows=4,
            minimum_easting=100.0,
            maximum_northing=200.0,
            method=method,
            iho_order="order1a",
            grid_resolution_x=1.0,
            grid_resolution_y=1.0,
            dist_exponent=2.0,
        )
        assert dg.shape == (4, 4)

    # Test run_cube_gridding invalid method
    with pytest.raises(NotImplementedError):
        run_cube_gridding(
            depth=np.array([10.0]),
            horizontal_uncertainty=np.array([0.5]),
            vertical_uncertainty=np.array([0.5]),
            easting=np.array([101.5]),
            northing=np.array([198.5]),
            num_columns=4,
            num_rows=4,
            minimum_easting=100.0,
            maximum_northing=200.0,
            method="invalid_method",
            iho_order="order1a",
            grid_resolution_x=1.0,
            grid_resolution_y=1.0,
        )


def test_cube_grid_multihypothesis_spatial_search():
    param = CubeParameters()
    param.initialize("order1a", 1.0, 1.0)
    grid = CubeGrid(
        minimum_easting=100.0,
        maximum_northing=200.0,
        num_columns=5,
        num_rows=5,
        resolution_x=1.0,
        resolution_y=1.0,
        param=param,
        use_queue=False,
    )

    # Set up node (2, 2) with 2 hypotheses
    node_multi = grid.grid[2][2]
    node_multi.add_hypothesis(10.0, 0.5)
    node_multi.add_hypothesis(20.0, 0.5)

    # Set up node (2, 3) with 1 hypothesis (found during row search)
    node_single = grid.grid[2][3]
    node_single.add_hypothesis(10.2, 0.5)

    # Extract with 'local' and 'posterior'
    vals_local = grid.get_grid_values(("depth", "uncertainty"), method="local")
    assert not np.isnan(vals_local[0][2, 2])

    vals_posterior = grid.get_grid_values(("depth", "uncertainty"), method="posterior")
    assert not np.isnan(vals_posterior[0][2, 2])


def test_cube_logging_and_filters(tmp_path):
    import logging

    log_file = str(tmp_path / "test.log")
    logger = return_logger(logfile=log_file, loglevel=logging.DEBUG)
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    err_filter = StdErrFilter()
    out_filter = StdOutFilter()

    rec_info = logging.LogRecord("test", logging.INFO, "path", 1, "msg", (), None)
    rec_warn = logging.LogRecord("test", logging.WARNING, "path", 1, "msg", (), None)
    rec_err = logging.LogRecord("test", logging.ERROR, "path", 1, "msg", (), None)
    rec_crit = logging.LogRecord("test", logging.CRITICAL, "path", 1, "msg", (), None)
    rec_debug = logging.LogRecord("test", logging.DEBUG, "path", 1, "msg", (), None)

    assert not err_filter.filter(rec_info)
    assert not err_filter.filter(rec_debug)
    assert err_filter.filter(rec_warn)
    assert err_filter.filter(rec_err)
    assert err_filter.filter(rec_crit)

    assert out_filter.filter(rec_info)
    assert out_filter.filter(rec_debug)
    assert not out_filter.filter(rec_warn)
    assert not out_filter.filter(rec_err)
    assert not out_filter.filter(rec_crit)


def test_iho_limits():
    assert get_iho_limits("exclusive") == (0.15, 0.0075)
    assert get_iho_limits("special") == (0.25, 0.0075)
    assert get_iho_limits("order1a") == (0.5, 0.013)
    assert get_iho_limits("order1b") == (0.5, 0.013)
    assert get_iho_limits("order2") == (1.0, 0.023)
    assert get_iho_limits("unknown") is None


def test_cube_node_nominate_branches():
    cb = CubeNode()
    cb.add_hypothesis(5.005, 1.0)
    cb.add_hypothesis(5.002, 1.0)
    cb.add_hypothesis(5.008, 1.0)
    cb.nominate_hypothesis(5.0)
    assert cb.nominated.current_depth == 5.002


def test_cube_node_update_hypothesis_variance_selection():
    cb = CubeNode()
    cb.add_hypothesis(10.0, 0.5)
    cb.variance_selection = "max"
    assert cb.update_hypothesis(0, 10.1, 0.5)
    cb.variance_selection = "input"
    assert cb.update_hypothesis(0, 10.1, 0.5)


def test_cube_node_choose_hypothesis_branches():
    cb = CubeNode()
    h0 = Hypothesis(5.0, 0.5)
    h0.number_of_points = 2
    h1 = Hypothesis(6.0, 0.5)
    h1.number_of_points = 10
    h2 = Hypothesis(7.0, 0.5)
    h2.number_of_points = 5
    h3 = Hypothesis(8.0, 0.5)
    h3.number_of_points = 0
    cb.hypotheses = [h0, h1, h2, h3]
    best_hypo, ratio = cb.choose_hypothesis()
    assert best_hypo.current_depth == 6.0
    assert ratio > 0.0

    # Test hyp count <= second_highest_count
    cb2 = CubeNode()
    ha = Hypothesis(6.0, 0.5)
    ha.number_of_points = 10
    hb = Hypothesis(7.0, 0.5)
    hb.number_of_points = 5
    hc = Hypothesis(8.0, 0.5)
    hc.number_of_points = 2
    cb2.hypotheses = [ha, hb, hc]
    best2, ratio2 = cb2.choose_hypothesis()
    assert best2.current_depth == 6.0

    # Test single positive hypothesis
    cb3 = CubeNode()
    h_single = Hypothesis(6.0, 0.5)
    h_single.number_of_points = 10
    h_zero = Hypothesis(7.0, 0.5)
    h_zero.number_of_points = 0
    cb3.hypotheses = [h_single, h_zero]
    best3, ratio3 = cb3.choose_hypothesis()
    assert best3.current_depth == 6.0
    assert ratio3 == 0.0


def test_cube_node_queue_insert_truncate_branch():
    cb = CubeNode()
    cb.median_length = 11
    for d in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 100.0, 100.1, 100.2, 100.3, 100.4]:
        cb.queue_fill(d, 0.0001)
    cb.queue_insert(50.0, 0.0001)

    cb_no_trunc = CubeNode()
    cb_no_trunc.median_length = 11
    for d in [10.0, 10.01, 10.02, 10.03, 10.04, 10.05, 10.06, 10.07, 10.08, 10.09, 10.10]:
        cb_no_trunc.queue_fill(d, 1.0)
    cb_no_trunc.queue_insert(10.05, 1.0)


def test_cube_node_add_to_queue_no_queue():
    cb = CubeNode()
    cb.use_queue = False
    cb.add_to_queue(10.0, 0.5)
    assert len(cb.hypotheses) == 1


def test_cube_node_add_point_to_node_sounding_range():
    cb = CubeNode()
    cb.predicted_depth = 12.0
    cb.predicted_variance = 0.5
    cb.add_point_to_node(10.0, 0.5, 0.5, 0.01, sounding_range=5.0)
    assert len(cb.queue) == 1


def test_cube_node_return_answers_coverage():
    cb = CubeNode()
    cb.add_hypothesis(10.0, 0.5)
    cb.nominate_hypothesis(10.0)
    vals = cb._return_nominated_answer(("depth", "uncertainty", "ratio", "n_hypotheses", "unknown"))
    assert len(vals) == 4

    h = Hypothesis(10.0, 0.5)
    h.number_of_points = 5
    h.variance_estimate = 0.5
    for var_sel in ["max", "input", "cube"]:
        cb.variance_selection = var_sel
        ans = cb._return_answer_from_hypothesis(h, 0.5, ("depth", "uncertainty", "ratio", "n_hypotheses", "unknown"))
        assert len(ans) == 4

    h_empty = Hypothesis(10.0, 0.5)
    h_empty.number_of_points = 0
    ans_empty = cb._return_answer_from_hypothesis(h_empty, 0.0, ("depth", "uncertainty", "ratio", "n_hypotheses"))
    assert all(np.isnan(x) for x in ans_empty)


def test_cube_node_extract_closest_posterior_empty_hypos():
    cb = CubeNode()
    h1 = Hypothesis(10.0, 0.5)
    h1.number_of_points = 0
    h2 = Hypothesis(20.0, 0.5)
    h2.number_of_points = 0
    cb.hypotheses = [h1, h2]

    closest = cb.extract_closest_node_value(10.0, 0.5)
    assert all(np.isnan(x) for x in closest)

    posterior = cb.extract_posterior_weighted_node_value(10.0, 0.5)
    assert all(np.isnan(x) for x in posterior)


def test_cube_grid_insert_points_max_radius():
    param = CubeParameters()
    param.initialize("order1a", 1.0, 1.0)
    grid = CubeGrid(
        minimum_easting=100.0,
        maximum_northing=200.0,
        num_columns=4,
        num_rows=4,
        resolution_x=1.0,
        resolution_y=1.0,
        param=param,
        use_queue=False,
    )
    # Deep sounding with small horizontal uncertainty to hit radius > max_radius (line 1350)
    grid.insert_points(np.array([1000.0]), np.array([0.0001]), np.array([0.5]), np.array([101.5]), np.array([198.5]))

    # Sounding with 0.0 <= radius <= max_radius to hit line 1350->1352 fall-through
    grid.insert_points(np.array([10.0]), np.array([0.04137]), np.array([1.0]), np.array([101.5]), np.array([198.5]))


def test_cube_grid_boundary_context_searches():
    param = CubeParameters()
    param.initialize("order1a", 1.0, 1.0)
    param.min_context = 1
    param.max_context = 2

    # Grid with multi-hypothesis at (0, 0) to hit negative target_row and target_col offsets
    grid_bound = CubeGrid(
        minimum_easting=100.0,
        maximum_northing=200.0,
        num_columns=4,
        num_rows=4,
        resolution_x=1.0,
        resolution_y=1.0,
        param=param,
        use_queue=False,
    )
    n_corner = grid_bound.grid[0][0]
    n_corner.add_hypothesis(10.0, 0.5)
    n_corner.add_hypothesis(20.0, 0.5)
    n_corner.hypotheses[0].current_variance = 0.5
    n_corner.hypotheses[1].current_variance = 0.5
    for r in grid_bound.grid:
        for n in r:
            n.predicted_depth = 10.0
            n.predicted_variance = 0.5
    res_bound = grid_bound.get_grid_values(("depth",), method="local")
    assert res_bound[0].shape == (4, 4)

    # Test predicted extraction method on get_grid_values
    res_predicted = grid_bound.get_grid_values(("depth",), method="predicted")
    assert res_predicted[0].shape == (4, 4)


def test_cube_grid_more_spatial_searches_and_shortcuts():
    param = CubeParameters()
    param.initialize("order1a", 1.0, 1.0)
    param.min_context = 1
    param.max_context = 2

    # Column search branch
    grid_col = CubeGrid(
        minimum_easting=100.0,
        maximum_northing=200.0,
        num_columns=5,
        num_rows=5,
        resolution_x=1.0,
        resolution_y=1.0,
        param=param,
        use_queue=False,
    )
    for r in grid_col.grid:
        for n in r:
            n.predicted_depth = 10.0
            n.predicted_variance = 0.5
    node_multi = grid_col.grid[2][2]
    node_multi.add_hypothesis(10.0, 0.5)
    node_multi.add_hypothesis(20.0, 0.5)
    node_multi.hypotheses[0].current_variance = 0.5
    node_multi.hypotheses[1].current_variance = 0.5

    node_col_single = grid_col.grid[3][2]
    node_col_single.add_hypothesis(10.2, 0.5)
    node_col_single.hypotheses[0].current_variance = 0.5

    res_col_local = grid_col.get_grid_values(("depth", "uncertainty"), method="local")
    assert not np.isnan(res_col_local[0][2, 2])
    res_col_post = grid_col.get_grid_values(("depth", "uncertainty"), method="posterior")
    assert not np.isnan(res_col_post[0][2, 2])

    # No neighbor found branch (fallback to basic node value extraction)
    grid_none = CubeGrid(
        minimum_easting=100.0,
        maximum_northing=200.0,
        num_columns=5,
        num_rows=5,
        resolution_x=1.0,
        resolution_y=1.0,
        param=param,
        use_queue=False,
    )
    for r in grid_none.grid:
        for n in r:
            n.predicted_depth = 10.0
            n.predicted_variance = 0.5
    node_multi2 = grid_none.grid[2][2]
    node_multi2.add_hypothesis(10.0, 0.5)
    node_multi2.add_hypothesis(20.0, 0.5)
    node_multi2.hypotheses[0].current_variance = 0.5
    node_multi2.hypotheses[1].current_variance = 0.5

    res_none = grid_none.get_grid_values(("depth", "uncertainty"), method="local")
    assert not np.isnan(res_none[0][2, 2])

    # Test all shortcut methods across all method modes including unknown fallback
    for m in ["local", "posterior", "prior", "predicted"]:
        assert grid_none.get_grid_depth(method=m).shape == (5, 5)
        assert grid_none.get_grid_uncertainty(method=m).shape == (5, 5)
        assert grid_none.get_grid_ratio(method=m).shape == (5, 5)
        assert len(grid_none.get_grid_depth_and_uncertainty(method=m)) == 2

    assert grid_none.get_grid_depth(method="unknown") is None
    assert grid_none.get_grid_uncertainty(method="unknown") is None
    assert grid_none.get_grid_ratio(method="unknown") is None
    assert grid_none.get_grid_depth_and_uncertainty(method="unknown") is None

    # Test empty_nodes_count when some nodes are populated
    assert grid_none.empty_nodes_count == 24
    assert grid_none.total_nodes_count == 25


def test_run_cube_gridding_extra_kwargs_and_main():
    import runpy

    # Test extra kwargs not in CubeParameters
    dg, ug, rg, ng = run_cube_gridding(
        depth=np.array([10.0]),
        horizontal_uncertainty=np.array([0.5]),
        vertical_uncertainty=np.array([0.5]),
        easting=np.array([101.5]),
        northing=np.array([198.5]),
        num_columns=4,
        num_rows=4,
        minimum_easting=100.0,
        maximum_northing=200.0,
        method="local",
        iho_order="order1a",
        grid_resolution_x=1.0,
        grid_resolution_y=1.0,
        unknown_kwarg="ignored",
    )
    assert dg.shape == (4, 4)

    # Test running __main__ block
    import bathycube.cube

    runpy.run_path(bathycube.cube.__file__, run_name="__main__")

import pytest

from drl5g.config import config_from_dict, list_scenarios, load_config, load_scenario, save_config


def test_committed_scenarios_load():
    assert list_scenarios() == ["a", "b", "c", "d"]
    for scenario in list_scenarios():
        config = load_scenario(scenario)
        assert config.name == scenario.upper()
        assert config.env.throughput_norm_mbps == 50.0  # one service target for every cell


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="n_ues"):
        config_from_dict({"name": "X", "env": {"n_ues": 3}})
    with pytest.raises(ValueError, match="top-level"):
        config_from_dict({"name": "X", "agent": {}})


def test_rbg_must_divide_rbs():
    with pytest.raises(ValueError, match="multiple"):
        config_from_dict({"name": "X", "env": {"n_rb": 52}, "dqn": {"rbg_size": 5}})


def test_round_trip(tmp_path):
    config = load_scenario("c")
    save_config(config, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == config


def test_unknown_scenario_lists_alternatives():
    with pytest.raises(FileNotFoundError, match="available: a, b, c, d"):
        load_scenario("z")

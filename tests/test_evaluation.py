import numpy as np
import pytest

from drl5g.cli import main
from drl5g.config import EnvConfig, ScenarioConfig
from drl5g.evaluation import (
    evaluate_scenario,
    paired_difference,
    read_episodes,
    summarize,
    write_episodes,
)


@pytest.fixture
def config() -> ScenarioConfig:
    return ScenarioConfig(name="T", env=EnvConfig(n_ue=5, n_rb=10, max_steps=4))


def test_schedulers_see_the_same_drops(config):
    results = evaluate_scenario("t", config, ["RoundRobin", "PropFair"], seeds=[0, 1], episodes=3)
    rr = [r for r in results if r.scheduler == "RoundRobin"]
    pf = [r for r in results if r.scheduler == "PropFair"]
    # Static channel: PF settles on equal shares, exactly what RR does, on identical drops.
    assert [r.throughput_mbps for r in rr] == pytest.approx([r.throughput_mbps for r in pf])


def test_summary_and_paired_difference(config):
    results = evaluate_scenario("t", config, ["RoundRobin", "MaxRate"], seeds=[0], episodes=6)
    rows = {row["scheduler"]: row for row in summarize(results)}
    assert rows["RoundRobin"]["episodes"] == 6
    assert rows["MaxRate"]["starved"] == pytest.approx(0.8)  # one UE of five is served
    assert rows["RoundRobin"]["starved"] == 0.0

    diff, ci = paired_difference(results, "t", "MaxRate", "RoundRobin", metric="throughput_mbps")
    expected = np.mean([r.throughput_mbps for r in results if r.scheduler == "MaxRate"]) - np.mean(
        [r.throughput_mbps for r in results if r.scheduler == "RoundRobin"]
    )
    assert diff == pytest.approx(expected)
    assert diff > 0 and ci > 0


def test_episode_csv_round_trip(tmp_path, config):
    results = evaluate_scenario("t", config, ["Random"], seeds=[3], episodes=2)
    write_episodes(results, tmp_path / "episodes.csv")
    loaded = read_episodes(tmp_path / "episodes.csv")
    assert [r.seed for r in loaded] == [3, 3]
    assert loaded[0].reward == pytest.approx(results[0].reward)


def test_plot_command_draws_figures(tmp_path):
    main(
        [
            "evaluate",
            "--scenarios",
            "a",
            "b",
            "--schedulers",
            "RoundRobin",
            "Greedy",
            "--seeds",
            "1",
            "--episodes",
            "2",
            "--out",
            str(tmp_path / "eval"),
        ]
    )
    main(["plot", "--evaluation-dir", str(tmp_path / "eval"), "--out", str(tmp_path / "fig")])
    for name in ("objective.png", "tradeoff.png"):
        assert (tmp_path / "fig" / name).stat().st_size > 10_000

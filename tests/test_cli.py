import csv

from drl5g.cli import main


def test_demo_prints_ttis(capsys):
    main(["demo", "--steps", "2", "--scheduler", "PropFair"])
    out = capsys.readouterr().out
    assert "Scenario A" in out and out.strip().splitlines()[-1].lstrip().startswith("2")


def test_evaluate_writes_summary(tmp_path):
    main(
        [
            "evaluate",
            "--scenarios",
            "d",
            "--schedulers",
            "RoundRobin",
            "MaxRate",
            "--seeds",
            "2",
            "--episodes",
            "2",
            "--out",
            str(tmp_path),
        ]
    )
    with open(tmp_path / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["scheduler"] for r in rows] == ["RoundRobin", "MaxRate"]
    assert all(int(r["episodes"]) == 4 for r in rows)
    with open(tmp_path / "episodes.csv") as f:
        assert sum(1 for _ in f) == 1 + 2 * 4

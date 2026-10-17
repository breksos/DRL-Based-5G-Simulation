"""Command-line interface: ``drl5g {demo,train,evaluate,plot,gui}``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from drl5g import paths
from drl5g.config import ScenarioConfig, list_scenarios, load_scenario

log = logging.getLogger("drl5g")


def _scenario_ids(requested: Sequence[str]) -> list[str]:
    available = list_scenarios()
    ids = available if "all" in requested else [s.lower() for s in requested]
    unknown = sorted(set(ids) - set(available))
    if unknown:
        raise SystemExit(
            f"unknown scenario(s): {', '.join(unknown)} (available: {', '.join(available)})"
        )
    return ids


def scenario_title(scenario: str, config: ScenarioConfig) -> str:
    return f"{config.name} · {config.env.n_ue} UE / {config.env.n_rb} RB"


# -- subcommands ------------------------------------------------------------------
# Heavy modules (torch, matplotlib, tkinter) are imported inside the command
# that needs them, so `drl5g --help` and `drl5g demo` start instantly.
def cmd_demo(args: argparse.Namespace) -> None:
    from drl5g.env import FiveGEnv
    from drl5g.schedulers import make_scheduler

    config = load_scenario(args.scenario)
    env = FiveGEnv(config.env, seed=args.seed)
    scheduler = make_scheduler(args.scheduler, config, args.scenario, seed=args.seed)
    env.reset()
    scheduler.reset()
    print(f"Scenario {config.name} ({config.description}) · scheduler {scheduler.name}")
    print(f"{'TTI':>4}  {'reward':>7}  {'throughput (Mbit/s)':>19}  {'Jain':>6}")
    for tti in range(1, args.steps + 1):
        _, reward, _, _, info = env.step(scheduler.allocate(env.rb_capacity_bps))
        throughput = info["throughput_total_bps"] / 1e6
        print(f"{tti:>4}  {reward:>7.4f}  {throughput:>19.2f}  {info['fairness']:>6.3f}")


def cmd_train(args: argparse.Namespace) -> None:
    import torch

    from drl5g.rl.trainer import train

    # The networks are tiny: one thread is fastest and keeps runs bit-reproducible.
    torch.set_num_threads(1)

    out = Path(args.out_dir)
    for scenario in _scenario_ids(args.scenarios):
        config = load_scenario(scenario)
        if args.seed is not None:
            config = replace(config, train=replace(config.train, seed=args.seed))
        log.info("training scenario %s: %s", scenario, config.description)
        result = train(
            config,
            checkpoint_path=paths.model_path(scenario, out / "models"),
            log_path=paths.training_log_path(scenario, out / "logs"),
            episodes=args.episodes,
        )
        log.info(
            "scenario %s done: best validation objective %.4f at episode %d (%.0f s)",
            scenario,
            result.best_validation_objective,
            result.best_episode,
            result.seconds,
        )


def cmd_evaluate(args: argparse.Namespace) -> None:
    from drl5g.evaluation import evaluate_scenario, summarize, write_episodes, write_rows

    results = []
    for scenario in _scenario_ids(args.scenarios):
        config = load_scenario(scenario)
        checkpoint = args.checkpoint or paths.model_path(scenario, Path(args.models_dir))
        log.info(
            "evaluating scenario %s (%d seeds × %d episodes)", scenario, args.seeds, args.episodes
        )
        results += evaluate_scenario(
            scenario, config, args.schedulers, range(args.seeds), args.episodes, checkpoint
        )

    summary = summarize(results)
    out = Path(args.out)
    write_episodes(results, out / "episodes.csv")
    write_rows(summary, out / "summary.csv")
    _print_summary(summary)
    _print_dqn_margins(results, summary)
    log.info("wrote %s and %s", out / "episodes.csv", out / "summary.csv")


def _print_dqn_margins(results: list, summary: list[dict]) -> None:
    """Paired objective difference between the DQN and the best other scheduler."""
    from drl5g.evaluation import paired_difference

    for scenario in sorted({row["scenario"] for row in summary}):
        rows = [row for row in summary if row["scenario"] == scenario]
        rivals = [row for row in rows if row["scheduler"] != "DQN"]
        if len(rivals) == len(rows) or not rivals:
            continue
        best = max(rivals, key=lambda row: row["reward"])["scheduler"]
        diff, ci = paired_difference(results, scenario, "DQN", best)
        print(f"scenario {scenario}: DQN − {best} = {diff:+.4f} ± {ci:.4f} (paired, 95 % CI)")


SUMMARY_COLUMNS = (  # (header, key, alignment + width, value precision)
    ("scen", "scenario", "<5", ""),
    ("scheduler", "scheduler", "<12", ""),
    ("R", "reward", ">8", ".4f"),
    ("±95%", "reward_ci95", ">8", ".4f"),
    ("Mbit/s", "throughput_mbps", ">9", ".2f"),
    ("Jain", "fairness", ">7", ".3f"),
    ("min UE", "min_ue_mbps", ">8", ".2f"),
    ("starved", "starved", ">9", ".1%"),
)


def _print_summary(summary: list[dict]) -> None:
    header = "".join(f"{title:{width}}" for title, _, width, _ in SUMMARY_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in summary:
        print("".join(f"{row[key]:{width}{spec}}" for _, key, width, spec in SUMMARY_COLUMNS))


def cmd_plot(args: argparse.Namespace) -> None:
    from drl5g import plots

    summary_csv = Path(args.evaluation_dir) / "summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"{summary_csv} not found; run `drl5g evaluate` first")
    scenarios = list_scenarios()
    titles = {s: scenario_title(s, load_scenario(s)) for s in scenarios}
    out = Path(args.out)
    written = [
        plots.objective_dotplot(summary_csv, out / "objective.png", titles),
        plots.tradeoff_scatter(summary_csv, out / "tradeoff.png", titles),
    ]
    logs = {s: paths.training_log_path(s) for s in scenarios if paths.training_log_path(s).exists()}
    if logs:
        written.append(plots.training_curves(logs, summary_csv, out / "training.png", titles))
    for path in written:
        log.info("wrote %s", path)


def cmd_gui(args: argparse.Namespace) -> None:
    from drl5g.gui.app import main as gui_main

    gui_main()


# -- parser -------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    from drl5g.schedulers import SCHEDULER_NAMES

    parser = argparse.ArgumentParser(
        prog="drl5g", description="5G downlink resource-block scheduling with a DQN agent."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="print a few TTIs of one scheduler")
    demo.add_argument("--scenario", default="a")
    demo.add_argument("--scheduler", default="RoundRobin", choices=SCHEDULER_NAMES)
    demo.add_argument("--steps", type=int, default=10)
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=cmd_demo)

    train = sub.add_parser("train", help="train the DQN scheduler")
    train.add_argument("scenarios", nargs="+", help="scenario ids (e.g. a b) or 'all'")
    train.add_argument("--episodes", type=int, help="override the configured episode count")
    train.add_argument("--seed", type=int, help="override the configured training seed")
    train.add_argument(
        "--out-dir", default=str(paths.RESULTS_DIR), help="writes models/ and logs/ here"
    )
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="compare schedulers on shared UE drops")
    evaluate.add_argument("--scenarios", nargs="+", default=["all"])
    evaluate.add_argument(
        "--schedulers", nargs="+", default=list(SCHEDULER_NAMES), choices=SCHEDULER_NAMES
    )
    evaluate.add_argument("--seeds", type=int, default=5, help="number of seeds (0 … N-1)")
    evaluate.add_argument("--episodes", type=int, default=100, help="UE drops per seed")
    evaluate.add_argument("--models-dir", default=str(paths.MODELS_DIR))
    evaluate.add_argument(
        "--checkpoint",
        help="evaluate this DQN model on every scenario (cross-scenario transfer)",
    )
    evaluate.add_argument("--out", default=str(paths.EVALUATION_DIR))
    evaluate.set_defaults(func=cmd_evaluate)

    plot = sub.add_parser("plot", help="draw figures from evaluation and training CSVs")
    plot.add_argument("--evaluation-dir", default=str(paths.EVALUATION_DIR))
    plot.add_argument("--out", default=str(paths.FIGURES_DIR))
    plot.set_defaults(func=cmd_plot)

    gui = sub.add_parser("gui", help="open the live simulation window")
    gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

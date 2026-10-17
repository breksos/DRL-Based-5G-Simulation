"""Live Tkinter dashboard: watch any scheduler allocate resource blocks TTI by TTI.

Run with ``drl5g gui`` (or ``python -m drl5g gui``).
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, ttk

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from drl5g.config import ScenarioConfig, list_scenarios, load_scenario
from drl5g.env import FiveGEnv
from drl5g.schedulers import SCHEDULER_NAMES, Scheduler, make_scheduler

BG = "#1e1e2e"
PANEL_BG = "#2a2a3e"
FG = "#cdd6f4"
FG_MUTED = "#a6adc8"
ACCENT = "#89b4fa"
FAIRNESS = "#a6e3a1"
GRID = "#313244"
HISTORY_LEN = 200
RB_HISTORY_ROWS = 10
SNR_RANGE_DB = (-10.0, 35.0)


@dataclass
class RunState:
    """Counters and histories of the running simulation."""

    episode: int = 1
    tti: int = 0
    episode_reward: float = 0.0
    last_info: dict = field(default_factory=lambda: {"throughput_total_bps": 0.0, "fairness": 0.0})
    rewards: list[float] = field(default_factory=list)
    fairness: list[float] = field(default_factory=list)


class SimulationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("5G RB Scheduling · drl5g")
        self.root.configure(bg=BG)
        self.root.minsize(1260, 780)

        self._scenarios = {self._scenario_label(s): s for s in list_scenarios()}
        if not self._scenarios:
            raise SystemExit("no scenario_*.yaml files found in configs/")
        self._running = False
        self._resetting = False

        self._build_controls()
        self._build_canvas()
        self._reset()

    # -- layout -------------------------------------------------------------------
    @staticmethod
    def _scenario_label(scenario: str) -> str:
        cfg = load_scenario(scenario).env
        return f"{scenario.upper()}  ·  {cfg.n_ue} UE / {cfg.n_rb} RB"

    def _label(self, parent: tk.Widget, text: str, bold: bool = False, size: int = 9) -> tk.Label:
        font = ("Helvetica", size, "bold" if bold else "normal")
        return tk.Label(parent, text=text, font=font, bg=PANEL_BG, fg=FG)

    def _separator(self, parent: tk.Widget) -> None:
        tk.Frame(parent, bg="#45475a", height=1).pack(fill=tk.X, padx=8, pady=6)

    def _build_controls(self) -> None:
        panel = tk.Frame(self.root, bg=PANEL_BG, width=230)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        panel.pack_propagate(False)

        self._label(panel, "5G RB Scheduling", bold=True, size=13).pack(pady=(14, 2))
        self._label(panel, "drl5g live simulation", size=8).pack(pady=(0, 10))
        self._separator(panel)

        self._scenario_var = self._combobox(panel, "Scenario", list(self._scenarios))
        self._scheduler_var = self._combobox(panel, "Scheduler", list(SCHEDULER_NAMES), "DQN")
        for var in (self._scenario_var, self._scheduler_var):
            var.trace_add("write", lambda *_: self._reset())
        self._separator(panel)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Dark.TButton", background="#313244", foreground=FG, borderwidth=0, padding=4
        )
        style.map("Dark.TButton", background=[("active", "#45475a")])
        style.configure(
            "Run.TButton", background="#1e6f3e", foreground="white", borderwidth=0, padding=4
        )
        style.map("Run.TButton", background=[("active", "#2a9d59")])

        ttk.Button(panel, text="↺ Reset", style="Dark.TButton", command=self._reset).pack(
            padx=10, pady=(4, 2), fill=tk.X
        )
        row = tk.Frame(panel, bg=PANEL_BG)
        row.pack(padx=8, pady=(2, 4), fill=tk.X)
        ttk.Button(row, text="→ Step", style="Dark.TButton", command=self._step).pack(
            side=tk.LEFT, padx=2, expand=True, fill=tk.X
        )
        self._run_button = ttk.Button(
            row, text="▶ Run", style="Run.TButton", command=self._toggle_run
        )
        self._run_button.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        self._label(panel, "Speed (ms per TTI)", size=8).pack(anchor="w", padx=10, pady=(8, 0))
        self._delay_ms = tk.IntVar(value=120)
        ttk.Scale(
            panel,
            from_=20,
            to=800,
            orient="horizontal",
            command=lambda v: self._delay_ms.set(int(float(v))),
        ).pack(fill=tk.X, padx=10, pady=2)
        self._separator(panel)

        self._stats = self._readouts(
            panel, ("Episode", "TTI", "Episode reward", "Throughput", "Fairness")
        )
        self._separator(panel)
        self._params = self._readouts(panel, ("UEs", "RBs", "Cell radius"))

    def _combobox(
        self, parent: tk.Widget, title: str, values: list[str], default: str | None = None
    ) -> tk.StringVar:
        self._label(parent, title, bold=True).pack(anchor="w", padx=10)
        var = tk.StringVar(value=default if default in values else values[0])
        ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=26).pack(
            padx=10, pady=(2, 8)
        )
        return var

    def _readouts(self, parent: tk.Widget, names: tuple[str, ...]) -> dict[str, tk.Label]:
        labels = {}
        for name in names:
            row = tk.Frame(parent, bg=PANEL_BG)
            row.pack(fill=tk.X, padx=10, pady=1)
            tk.Label(
                row,
                text=f"{name}:",
                width=14,
                anchor="w",
                font=("Helvetica", 8),
                bg=PANEL_BG,
                fg=FG_MUTED,
            ).pack(side=tk.LEFT)
            value = tk.Label(
                row, text="—", anchor="w", font=("Helvetica", 8, "bold"), bg=PANEL_BG, fg=FG
            )
            value.pack(side=tk.LEFT)
            labels[name] = value
        return labels

    def _build_canvas(self) -> None:
        frame = tk.Frame(self.root, bg=BG)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._fig = Figure(figsize=(13.5, 8.2), dpi=96)
        self._fig.patch.set_facecolor(BG)
        self._canvas = FigureCanvasTkAgg(self._fig, master=frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -- simulation lifecycle ---------------------------------------------------------
    def _reset(self) -> None:
        if self._resetting:  # a fallback below changes a combobox, which re-enters here
            return
        self._resetting = True
        try:
            self._rebuild()
        finally:
            self._resetting = False

    def _rebuild(self) -> None:
        self._set_running(False)
        scenario = self._scenarios[self._scenario_var.get()]
        config = load_scenario(scenario)
        self._env = FiveGEnv(config.env)
        self._scheduler = self._make_scheduler(config, scenario)
        self._env.reset()
        self._scheduler.reset()
        self._state = RunState()
        self._allocation = np.zeros(config.env.n_rb, dtype=np.int64)
        self._rb_history = np.zeros((RB_HISTORY_ROWS, config.env.n_rb), dtype=np.int64)

        self._params["UEs"].config(text=str(config.env.n_ue))
        self._params["RBs"].config(text=str(config.env.n_rb))
        self._params["Cell radius"].config(text=f"{config.env.cell_radius:.0f} m")
        self._init_plots()
        self._refresh()

    def _make_scheduler(self, config: ScenarioConfig, scenario: str) -> Scheduler:
        name = self._scheduler_var.get()
        try:
            return make_scheduler(name, config, scenario)
        except (FileNotFoundError, ValueError) as error:  # missing or v1-format model
            messagebox.showwarning(
                "DQN model unavailable", f"{error}\n\nFalling back to round robin."
            )
            self._scheduler_var.set("RoundRobin")
            return make_scheduler("RoundRobin", config, scenario)

    def _step(self) -> None:
        self._allocation = self._scheduler.allocate(self._env.rb_capacity_bps)
        _, reward, terminated, truncated, info = self._env.step(self._allocation)
        state = self._state
        state.tti += 1
        state.episode_reward += reward
        state.last_info = info
        state.rewards.append(reward)
        state.fairness.append(info["fairness"])
        self._rb_history = np.roll(self._rb_history, -1, axis=0)
        self._rb_history[-1] = self._allocation

        if terminated or truncated:
            state.episode += 1
            state.tti = 0
            state.episode_reward = 0.0
            self._env.reset()
            self._scheduler.reset()
            self._init_plots()
        self._refresh()

    def _set_running(self, running: bool) -> None:
        self._running = running
        if hasattr(self, "_run_button"):
            self._run_button.config(text="⏸ Pause" if running else "▶ Run")

    def _toggle_run(self) -> None:
        self._set_running(not self._running)
        if self._running:
            self._tick()

    def _tick(self) -> None:
        if self._running:
            self._step()
            self.root.after(self._delay_ms.get(), self._tick)

    # -- plotting -------------------------------------------------------------------
    def _style(self, ax, title: str) -> None:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.set_title(title, color=FG, fontsize=9, fontweight="bold", pad=4)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.6)

    def _init_plots(self) -> None:
        """(Re)build all axes; called on reset and whenever UEs are re-dropped."""
        env = self._env
        n_ue, radius = env.n_ue, env.config.cell_radius
        self._fig.clear()
        self._ue_colors = [colormaps["tab20"](i % 20 / 20) for i in range(n_ue)]
        grid = GridSpec(
            3,
            3,
            figure=self._fig,
            hspace=0.52,
            wspace=0.38,
            left=0.05,
            right=0.97,
            top=0.95,
            bottom=0.07,
        )

        # Topology: gNB at the centre, a link per UE coloured by SNR.
        ax = self._ax_topology = self._fig.add_subplot(grid[0:2, 0:2])
        self._style(ax, "Network topology")
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), "--", color="#585b70", lw=0.9)
        ax.scatter([0], [0], marker="*", s=380, c=ACCENT, zorder=5)
        ax.text(0, radius * 0.09, "gNB", color=ACCENT, ha="center", fontsize=7.5, fontweight="bold")
        ax.set_xlim(-radius * 1.12, radius * 1.12)
        ax.set_ylim(-radius * 1.12, radius * 1.12)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
        x, y = env.distances * np.cos(env.angles), env.distances * np.sin(env.angles)
        self._links = LineCollection([[(0.0, 0.0), (xi, yi)] for xi, yi in zip(x, y, strict=True)])
        ax.add_collection(self._links)
        ax.scatter(x, y, c=self._ue_colors, s=140, zorder=4, edgecolors="#585b70", linewidths=0.7)
        for i in range(n_ue):
            ax.text(
                x[i],
                y[i] + 0.04 * radius,
                f"U{i}",
                color=FG,
                fontsize=6.5,
                ha="center",
                va="bottom",
            )
        self._snr_cmap, self._snr_norm = colormaps["RdYlGn"], Normalize(*SNR_RANGE_DB)
        colorbar = self._fig.colorbar(
            ScalarMappable(norm=self._snr_norm, cmap=self._snr_cmap),
            ax=ax,
            shrink=0.65,
            pad=0.02,
        )
        self._style_colorbar(colorbar, "SNR (dB)")

        # SNR per UE.
        ax = self._fig.add_subplot(grid[0, 2])
        self._style(ax, "SNR per UE")
        ax.barh(range(n_ue), env.snr_db, color=self._ue_colors, height=0.72)
        ax.set_xlim(*SNR_RANGE_DB)
        ax.set_yticks(range(n_ue), [f"U{i}" for i in range(n_ue)], fontsize=6.5)
        ax.set_xlabel("SNR (dB)", fontsize=8)

        # Throughput per UE in the last TTI.
        ax = self._ax_throughput = self._fig.add_subplot(grid[1, 2])
        self._style(ax, "Throughput per UE (last TTI)")
        self._throughput_bars = ax.bar(
            range(n_ue), np.zeros(n_ue), color=self._ue_colors, width=0.72
        )
        ax.set_xticks(range(n_ue), [f"U{i}" for i in range(n_ue)], fontsize=6.2)
        ax.set_ylabel("Mbit/s", fontsize=8)

        # RB allocation of the last few TTIs.
        ax = self._fig.add_subplot(grid[2, 0:2])
        self._style(ax, f"RB allocation (last {RB_HISTORY_ROWS} TTIs)")
        ax.grid(False)
        self._rb_image = ax.imshow(
            self._rb_history,
            aspect="auto",
            cmap=ListedColormap(self._ue_colors),
            vmin=0,
            vmax=max(n_ue - 1, 1),
            interpolation="nearest",
        )
        ax.set_ylabel("TTI (newest at bottom)", fontsize=8)
        ax.set_xlabel("RB index", fontsize=8)
        ticks = list(range(n_ue)) if n_ue <= 12 else list(range(0, n_ue, 2))
        colorbar = self._fig.colorbar(
            self._rb_image, ax=ax, orientation="horizontal", pad=0.28, shrink=0.55, ticks=ticks
        )
        self._style_colorbar(colorbar, "UE index")

        # Reward and fairness history share one axis.
        ax = self._ax_history = self._fig.add_subplot(grid[2, 2])
        self._style(ax, "Live history")
        (self._reward_line,) = ax.plot([], [], color=ACCENT, lw=1.6, label="Reward")
        (self._fairness_line,) = ax.plot([], [], color=FAIRNESS, lw=1.4, label="Fairness")
        ax.set_xlim(0, HISTORY_LEN)
        ax.set_ylim(-1.0, 1.0)  # the starvation penalty can push the reward below zero
        ax.axhline(0.0, color=FG_MUTED, lw=0.6)
        ax.set_xlabel("TTI", fontsize=8)
        ax.legend(fontsize=7, loc="lower left", facecolor=PANEL_BG, edgecolor=GRID, labelcolor=FG)

    def _style_colorbar(self, colorbar, label: str) -> None:
        colorbar.ax.tick_params(colors=FG, labelsize=7)
        colorbar.set_label(label, color=FG, fontsize=8)
        colorbar.outline.set_edgecolor(GRID)

    def _refresh(self) -> None:
        env, state = self._env, self._state
        served = np.bincount(self._allocation, minlength=env.n_ue) > 0
        colors = self._snr_cmap(self._snr_norm(env.snr_db))
        colors[:, 3] = np.where(served, 0.95, 0.30)
        self._links.set_colors(colors)
        self._links.set_linewidths(np.where(served, 3.5, 1.2))

        throughput_mbps = env.last_throughput_bps / 1e6
        for bar, value in zip(self._throughput_bars, throughput_mbps, strict=True):
            bar.set_height(value)
        self._ax_throughput.set_ylim(0, max(throughput_mbps.max() * 1.15, 0.5))
        self._rb_image.set_data(self._rb_history)

        rewards, fairness = state.rewards[-HISTORY_LEN:], state.fairness[-HISTORY_LEN:]
        self._reward_line.set_data(range(len(rewards)), rewards)
        self._fairness_line.set_data(range(len(fairness)), fairness)
        self._canvas.draw_idle()

        info = state.last_info
        self._stats["Episode"].config(text=str(state.episode))
        self._stats["TTI"].config(text=f"{state.tti} / {env.config.max_steps}")
        self._stats["Episode reward"].config(text=f"{state.episode_reward:.2f}")
        self._stats["Throughput"].config(text=f"{info['throughput_total_bps'] / 1e6:.1f} Mbit/s")
        self._stats["Fairness"].config(text=f"{info['fairness']:.3f}")


def main() -> None:
    root = tk.Tk()
    root.geometry("1400x840")
    SimulationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

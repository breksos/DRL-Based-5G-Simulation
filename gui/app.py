"""
5G Network Simulation — real-time GUI.

Run from project root:
    python gui/app.py
    python main.py --gui
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from pathlib import Path

from env.network_env import FiveGEnv
from baselines.schedulers import RandomScheduler, RoundRobinScheduler, ProportionalFairScheduler
from agents.dqn_scheduler import DQNScheduler
from utils.config import load_config

# ── palette ────────────────────────────────────────────────────────────────
BG        = '#1e1e2e'
PANEL_BG  = '#2a2a3e'
FG        = '#cdd6f4'
ACCENT    = '#89b4fa'
BTN_ACT   = '#a6e3a1'
PLOT_BG   = '#1e1e2e'
GRID_COL  = '#313244'
HIST_LEN  = 200   # steps to display in history plots

def _discover_scenarios() -> dict:
    """configs/ klasöründeki tüm scenario_*.yaml dosyalarını bulur."""
    import glob
    opts = {}
    files = sorted(glob.glob(os.path.join("configs", "scenario_*.yaml")))
    for f in files:
        name = os.path.basename(f).replace(".yaml", "")   # scenario_a
        letter = name.split("_")[-1].upper()               # A
        try:
            cfg = load_config(f)
            label = f"{letter}  ·  {cfg.n_ue} UE / {cfg.n_rb} RB"
        except Exception:
            label = letter
        opts[label] = name
    return opts if opts else {"A  ·  10 UE / 50 RB": "scenario_a"}

SCENARIO_OPTS = _discover_scenarios()
SCHED_OPTS = {
    "Random":            RandomScheduler,
    "Round Robin":       RoundRobinScheduler,
    "Proportional Fair": ProportionalFairScheduler,
    "DQN Agent":         DQNScheduler,
}


class SimApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("5G Network Simulation  ·  DRL Project")
        self.root.configure(bg=BG)
        self.root.minsize(1260, 780)

        self._env: FiveGEnv | None = None
        self._scheduler         = None
        self._state: np.ndarray | None = None
        self._running           = False
        self._step              = 0
        self._episode           = 1
        self._ep_reward         = 0.0
        self._last_action       = None
        self._last_info         = {"throughput_total": 0.0, "fairness": 1.0}
        self._ue_angles         = None
        self._reward_hist: list[float] = []
        self._fair_hist:   list[float] = []
        self._ep_reward_hist: list[float] = []

        self._build_ui()
        self._reset_env()

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        left = tk.Frame(self.root, bg=PANEL_BG, width=220)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        left.pack_propagate(False)
        self._build_controls(left)

        right = tk.Frame(self.root, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._build_canvas(right)

    def _build_controls(self, f):
        def lbl(parent, text, bold=False, size=9):
            font = ('Helvetica', size, 'bold' if bold else 'normal')
            return tk.Label(parent, text=text, font=font, bg=PANEL_BG, fg=FG)

        def sep():
            tk.Frame(f, bg='#45475a', height=1).pack(fill=tk.X, padx=8, pady=6)

        lbl(f, "5G Simulation", bold=True, size=13).pack(pady=(14, 2))
        lbl(f, "DRL Project", size=8).pack(pady=(0, 10))
        sep()

        # Scenario
        lbl(f, "Scenario", bold=True).pack(anchor='w', padx=10)
        self._scen_var = tk.StringVar(value=list(SCENARIO_OPTS)[0])
        cb1 = ttk.Combobox(f, textvariable=self._scen_var,
                            values=list(SCENARIO_OPTS), state='readonly', width=26)
        cb1.pack(padx=10, pady=(2, 8))
        self._scen_var.trace_add('write', lambda *_: self._reset_env())

        # Scheduler
        lbl(f, "Scheduler", bold=True).pack(anchor='w', padx=10)
        self._sched_var = tk.StringVar(value=list(SCHED_OPTS)[1])  # Round Robin default
        cb2 = ttk.Combobox(f, textvariable=self._sched_var,
                            values=list(SCHED_OPTS), state='readonly', width=26)
        cb2.pack(padx=10, pady=(2, 8))
        self._sched_var.trace_add('write', lambda *_: self._on_sched_change())

        # Mobility toggle
        self._mobility_var = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Mobility (±10 m walk)", variable=self._mobility_var,
                       bg=PANEL_BG, fg=FG, selectcolor=PANEL_BG,
                       activebackground=PANEL_BG, activeforeground=FG,
                       command=self._on_mobility_toggle).pack(anchor='w', padx=10, pady=2)

        sep()

        # Buttons
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TButton', background='#313244', foreground=FG,
                         borderwidth=0, padding=4)
        style.map('Dark.TButton', background=[('active', '#45475a')])
        style.configure('Run.TButton', background='#1e6f3e', foreground='white',
                         borderwidth=0, padding=4)
        style.map('Run.TButton', background=[('active', '#2a9d59')])

        btn_row1 = tk.Frame(f, bg=PANEL_BG)
        btn_row1.pack(padx=8, pady=(4, 2), fill=tk.X)
        ttk.Button(btn_row1, text="↺ Reset", style='Dark.TButton',
                   command=self._reset_env).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        ttk.Button(btn_row1, text="⟳ Refresh", style='Dark.TButton',
                   command=self._refresh_scenarios).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        btn_row2 = tk.Frame(f, bg=PANEL_BG)
        btn_row2.pack(padx=8, pady=(2, 4), fill=tk.X)
        ttk.Button(btn_row2, text="→ Step", style='Dark.TButton',
                   command=self._do_step).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        self._run_btn = ttk.Button(btn_row2, text="▶ Run", style='Run.TButton',
                                   command=self._toggle_run)
        self._run_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        # Speed
        lbl(f, "Speed (ms/step)", size=8).pack(anchor='w', padx=10, pady=(8, 0))
        speed_row = tk.Frame(f, bg=PANEL_BG)
        speed_row.pack(fill=tk.X, padx=10, pady=2)
        self._speed_var = tk.IntVar(value=120)
        self._speed_disp = tk.StringVar(value="120")
        def _on_speed(v):
            try:
                iv = int(round(float(v)))
            except (TypeError, ValueError):
                return
            iv = max(20, min(800, iv))
            self._speed_var.set(iv)
            self._speed_disp.set(str(iv))
        speed_scale = ttk.Scale(speed_row, from_=20, to=800, orient='horizontal',
                                command=_on_speed)
        speed_scale.set(120)
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._speed_lbl = tk.Label(speed_row, textvariable=self._speed_disp,
                                   width=4, bg=PANEL_BG, fg=FG, font=('Helvetica', 8))
        self._speed_lbl.pack(side=tk.LEFT)

        sep()

        # Stats
        stats = [("Episode",      "episode",    "1"),
                 ("Step",         "step",       "0 / 200"),
                 ("Ep. Reward",   "reward",     "0.00"),
                 ("Throughput",   "throughput", "0.0 Mbps"),
                 ("Fairness",     "fairness",   "1.000")]
        self._stat = {}
        for label, key, default in stats:
            row = tk.Frame(f, bg=PANEL_BG)
            row.pack(fill=tk.X, padx=10, pady=1)
            tk.Label(row, text=f"{label}:", width=11, anchor='w',
                     font=('Helvetica', 8), bg=PANEL_BG, fg='#a6adc8').pack(side=tk.LEFT)
            v = tk.Label(row, text=default, anchor='w',
                         font=('Helvetica', 8, 'bold'), bg=PANEL_BG, fg=FG)
            v.pack(side=tk.LEFT)
            self._stat[key] = v

        sep()

        # Env params (read-only)
        self._pinfo = {}
        for label, key in [("N_UE", "n_ue"), ("N_RB", "n_rb"), ("Cell radius", "radius")]:
            row = tk.Frame(f, bg=PANEL_BG)
            row.pack(fill=tk.X, padx=10, pady=1)
            tk.Label(row, text=f"{label}:", width=11, anchor='w',
                     font=('Helvetica', 8), bg=PANEL_BG, fg='#a6adc8').pack(side=tk.LEFT)
            v = tk.Label(row, text="—", anchor='w',
                         font=('Helvetica', 8), bg=PANEL_BG, fg='#b4befe')
            v.pack(side=tk.LEFT)
            self._pinfo[key] = v

    def _build_canvas(self, frame):
        self._fig = Figure(figsize=(13.5, 8.2), dpi=96)
        self._fig.patch.set_facecolor(BG)
        canvas = FigureCanvasTkAgg(self._fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas = canvas

    # ── plot initialisation ─────────────────────────────────────────────────
    def _init_plots(self):
        self._fig.clear()
        n_ue = self._env.n_ue
        n_rb  = self._env.n_rb
        r     = self._env.cell_radius

        # tab20 has 20 discrete colours at positions k/20 for k=0..19
        self._uc = [cm.tab20((i % 20) / 20) for i in range(n_ue)]

        gs = gridspec.GridSpec(3, 3, figure=self._fig,
                               hspace=0.52, wspace=0.38,
                               left=0.05, right=0.97, top=0.95, bottom=0.07)

        def style_ax(ax, title):
            ax.set_facecolor(PLOT_BG)
            ax.tick_params(colors=FG, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(GRID_COL)
            ax.xaxis.label.set_color(FG)
            ax.yaxis.label.set_color(FG)
            ax.set_title(title, color=FG, fontsize=9, fontweight='bold', pad=4)
            ax.grid(True, color=GRID_COL, lw=0.4, alpha=0.6)

        # ── Network Topology (top-left 2×2) ─────────────────────────────
        self._ax_topo = self._fig.add_subplot(gs[0:2, 0:2])
        style_ax(self._ax_topo, 'Network Topology')

        theta = np.linspace(0, 2*np.pi, 300)
        self._ax_topo.plot(r*np.cos(theta), r*np.sin(theta),
                           '--', color='#585b70', lw=0.9)
        self._ax_topo.scatter([0], [0], marker='*', s=380,
                              c=ACCENT, zorder=5)
        self._ax_topo.text(0, r*0.04, 'gNB', color=ACCENT,
                           ha='center', fontsize=7.5, fontweight='bold')
        self._ax_topo.set_xlim(-r*1.12, r*1.12)
        self._ax_topo.set_ylim(-r*1.12, r*1.12)
        self._ax_topo.set_aspect('equal')
        self._ax_topo.set_xlabel('x (m)', fontsize=8)
        self._ax_topo.set_ylabel('y (m)', fontsize=8)

        self._topo_lines = LineCollection([], lw=0.8, alpha=0.3)
        self._ax_topo.add_collection(self._topo_lines)

        self._topo_sc = self._ax_topo.scatter(
            np.zeros(n_ue), np.zeros(n_ue),
            c=np.zeros(n_ue), cmap='RdYlGn',
            s=130, vmin=0, vmax=30, zorder=4,
            edgecolors='#585b70', linewidths=0.5
        )
        cb = self._fig.colorbar(self._topo_sc, ax=self._ax_topo,
                                shrink=0.65, pad=0.02)
        cb.ax.tick_params(colors=FG, labelsize=7)
        cb.set_label('SNR (dB)', color=FG, fontsize=8)
        cb.outline.set_edgecolor(GRID_COL)

        self._topo_txts = [
            self._ax_topo.text(0, 0, f'U{i}', color=FG, fontsize=6.5,
                               ha='center', va='bottom')
            for i in range(n_ue)
        ]

        # ── SNR per UE (top-right) ────────────────────────────────────────
        self._ax_snr = self._fig.add_subplot(gs[0, 2])
        style_ax(self._ax_snr, 'SNR per UE')
        self._snr_bars = self._ax_snr.barh(
            range(n_ue), np.zeros(n_ue),
            color=self._uc, edgecolor='none', height=0.72
        )
        self._ax_snr.set_xlim(0, 31)
        self._ax_snr.set_yticks(range(n_ue))
        self._ax_snr.set_yticklabels([f'U{i}' for i in range(n_ue)], fontsize=6.5)
        self._ax_snr.set_xlabel('SNR (dB)', fontsize=8)
        self._ax_snr.axvline(20, color='#a6e3a1', lw=0.8, ls='--', alpha=0.6)

        # ── Throughput per UE (mid-right) ────────────────────────────────
        self._ax_tp = self._fig.add_subplot(gs[1, 2])
        style_ax(self._ax_tp, 'Throughput per UE')
        self._tp_bars = self._ax_tp.bar(
            range(n_ue), np.zeros(n_ue),
            color=self._uc, edgecolor='none', width=0.72
        )
        self._ax_tp.set_xticks(range(n_ue))
        self._ax_tp.set_xticklabels([f'U{i}' for i in range(n_ue)], fontsize=6.2)
        self._ax_tp.set_ylabel('Mbps', fontsize=8)
        self._ax_tp.set_ylim(0, 5)

        # ── RB Allocation heatmap (bottom-left 2 cols) ───────────────────
        self._ax_rb = self._fig.add_subplot(gs[2, 0:2])
        style_ax(self._ax_rb, 'Resource Block Allocation')
        self._ax_rb.grid(False)
        from matplotlib.colors import ListedColormap
        cmap_rb = ListedColormap(self._uc)   # same colours as topology/bars
        self._rb_im = self._ax_rb.imshow(
            np.zeros((1, n_rb)), aspect='auto',
            cmap=cmap_rb, vmin=0, vmax=max(n_ue - 1, 1),
            interpolation='nearest'
        )
        self._ax_rb.set_yticks([])
        self._ax_rb.set_xlabel('RB Index', fontsize=8, color=FG)
        cb_ticks = list(range(n_ue)) if n_ue <= 12 else list(range(0, n_ue, 2))
        cb2 = self._fig.colorbar(self._rb_im, ax=self._ax_rb,
                                  orientation='horizontal', pad=0.28,
                                  shrink=0.55, ticks=cb_ticks)
        cb2.ax.tick_params(colors=FG, labelsize=6.5)
        cb2.set_label('UE index', color=FG, fontsize=7)
        cb2.outline.set_edgecolor(GRID_COL)

        # ── History (bottom-right) ────────────────────────────────────────
        self._ax_hist = self._fig.add_subplot(gs[2, 2])
        style_ax(self._ax_hist, 'Live History')
        self._rew_line,  = self._ax_hist.plot([], [], color=ACCENT,    lw=1.6, label='Reward')
        self._fair_line, = self._ax_hist.plot([], [], color='#a6e3a1', lw=1.4, ls='--', label='Fairness×20')
        self._ax_hist.legend(fontsize=7, loc='upper left',
                             facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=FG)
        self._ax_hist.set_xlabel('Step', fontsize=8)
        self._ax_hist.set_xlim(0, HIST_LEN)
        self._ax_hist.set_ylim(0, 40)

        self._canvas.draw()

    # ── env / scheduler lifecycle ──────────────────────────────────────────
    def _reset_env(self):
        was_running = self._running
        self._running = False

        key  = self._scen_var.get()  if hasattr(self, '_scen_var')  else list(SCENARIO_OPTS)[0]
        skey = self._sched_var.get() if hasattr(self, '_sched_var') else 'Round Robin'
        mob  = self._mobility_var.get() if hasattr(self, '_mobility_var') else False

        cfg = load_config(Path('configs') / f'{SCENARIO_OPTS[key]}.yaml')

        self._env = FiveGEnv(
            n_ue=cfg.n_ue, n_rb=cfg.n_rb,
            cell_radius=cfg.cell_radius,
            alpha=cfg.alpha, beta=cfg.beta,
            max_steps=cfg.max_steps, seed=None
        )
        self._env.mobility = mob

        sched_cls = SCHED_OPTS.get(skey, RoundRobinScheduler)
        scenario_key = SCENARIO_OPTS[key][-1]  # 'scenario_a' → 'a'
        if sched_cls is DQNScheduler:
            self._scheduler = DQNScheduler(cfg.n_ue, cfg.n_rb, scenario=scenario_key)
            status = self._scheduler.model_info
            self.root.title(f"5G Network Simulation  ·  DQN ({status})")
        else:
            self._scheduler = sched_cls(cfg.n_ue, cfg.n_rb)
            self.root.title("5G Network Simulation  ·  DRL Project")

        self._state, _ = self._env.reset()
        self._ue_angles  = np.random.uniform(0, 2*np.pi, cfg.n_ue)
        self._last_action = np.zeros(cfg.n_rb, dtype=int)

        self._step = 0
        self._episode = 1
        self._ep_reward = 0.0
        self._reward_hist = []
        self._fair_hist   = []
        self._ep_reward_hist = []

        self._init_plots()
        self._update_plots()

        if hasattr(self, '_pinfo'):
            self._pinfo['n_ue'].config(text=str(cfg.n_ue))
            self._pinfo['n_rb'].config(text=str(cfg.n_rb))
            self._pinfo['radius'].config(text=f"{int(cfg.cell_radius)} m")
        self._update_labels()

        if hasattr(self, '_run_btn'):
            self._run_btn.config(text="▶ Run")

    def _refresh_scenarios(self):
        """Yeni YAML eklendikten sonra senaryo listesini yeniler."""
        global SCENARIO_OPTS
        SCENARIO_OPTS = _discover_scenarios()
        self._scen_var.set(list(SCENARIO_OPTS)[0])
        # Combobox değerlerini güncelle
        for widget in self.root.winfo_children():
            self._update_combobox(widget)
        self._reset_env()

    def _update_combobox(self, widget):
        if isinstance(widget, ttk.Combobox):
            if self._scen_var in (widget.cget('textvariable'),):
                widget.config(values=list(SCENARIO_OPTS))
        for child in widget.winfo_children():
            self._update_combobox(child)

    def _on_sched_change(self):
        if self._env is None:
            return
        skey = self._sched_var.get()
        cls  = SCHED_OPTS.get(skey, RoundRobinScheduler)
        if cls is DQNScheduler:
            scenario_key = SCENARIO_OPTS[self._scen_var.get()][-1]
            self._scheduler = DQNScheduler(self._env.n_ue, self._env.n_rb, scenario=scenario_key)
            self.root.title(f"5G Network Simulation  ·  DQN ({self._scheduler.model_info})")
        else:
            self._scheduler = cls(self._env.n_ue, self._env.n_rb)
            self.root.title("5G Network Simulation  ·  DRL Project")

    def _on_mobility_toggle(self):
        if self._env is not None:
            self._env.mobility = self._mobility_var.get()

    # ── simulation loop ────────────────────────────────────────────────────
    def _do_step(self):
        if self._env is None:
            return
        action = self._scheduler.allocate(self._state)
        self._state, reward, terminated, truncated, info = self._env.step(action)
        self._last_action = action
        self._last_info   = info

        self._step      += 1
        self._ep_reward += reward
        self._reward_hist.append(reward)
        self._fair_hist.append(info['fairness'])

        if terminated or truncated:
            self._ep_reward_hist.append(self._ep_reward)
            self._episode += 1
            self._ep_reward = 0.0
            self._step = 0
            self._state, _ = self._env.reset()
            self._scheduler.reset()
            self._ue_angles = np.random.uniform(0, 2*np.pi, self._env.n_ue)
            # refresh topology bounds (cell radius doesn't change mid-run)

        self._update_plots()
        self._update_labels()

    def _toggle_run(self):
        self._running = not self._running
        self._run_btn.config(text="⏸ Pause" if self._running else "▶ Run")
        if self._running:
            self._auto_step()

    def _auto_step(self):
        if not self._running:
            return
        self._do_step()
        self.root.after(int(self._speed_var.get()), self._auto_step)

    # ── plot updates ───────────────────────────────────────────────────────
    def _update_plots(self):
        n_ue   = self._env.n_ue
        snrs   = self._state[:n_ue]
        tps    = self._env._throughputs / 1e6
        dists  = self._env.distances
        angles = self._ue_angles

        # Topology
        x = dists * np.cos(angles)
        y = dists * np.sin(angles)
        self._topo_sc.set_offsets(np.c_[x, y])
        self._topo_sc.set_array(snrs)
        segs = [[(0.0, 0.0), (float(x[i]), float(y[i]))] for i in range(n_ue)]
        self._topo_lines.set_segments(segs)
        self._topo_lines.set_colors(self._uc)
        for i, t in enumerate(self._topo_txts):
            offset = dists[i] * 0.07
            t.set_position((x[i], y[i] + offset))

        # SNR bars
        for bar, v in zip(self._snr_bars, snrs):
            bar.set_width(float(v))

        # Throughput bars
        max_tp = max(float(tps.max()), 0.5)
        for bar, v in zip(self._tp_bars, tps):
            bar.set_height(float(v))
        self._ax_tp.set_ylim(0, max_tp * 1.3)

        # RB heatmap
        self._rb_im.set_data(self._last_action[np.newaxis, :].astype(float))

        # History
        hist_r = self._reward_hist[-HIST_LEN:]
        hist_f = self._fair_hist[-HIST_LEN:]
        xs = np.arange(len(hist_r))
        self._rew_line.set_data(xs, hist_r)
        self._fair_line.set_data(xs, [v * 20 for v in hist_f])
        self._ax_hist.set_xlim(0, max(HIST_LEN, len(xs)))
        if hist_r:
            top = max(float(np.max(hist_r)) * 1.25, 5.0)
            self._ax_hist.set_ylim(0, top)

        self._canvas.draw_idle()

    def _update_labels(self):
        tp = self._last_info.get('throughput_total', 0) / 1e6
        jf = self._last_info.get('fairness', 0)
        ms = self._env.max_steps if self._env else 200
        self._stat['episode'].config(text=str(self._episode))
        self._stat['step'].config(text=f"{self._step} / {ms}")
        self._stat['reward'].config(text=f"{self._ep_reward:.2f}")
        self._stat['throughput'].config(text=f"{tp:.1f} Mbps")
        self._stat['fairness'].config(text=f"{jf:.3f}")


# ── entry point ────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    root.geometry("1400x840")
    SimApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()

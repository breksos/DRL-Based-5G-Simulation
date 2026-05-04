"""
Grafik üretim fonksiyonları — rapor kalitesi (300 dpi).
Tüm grafikler PNG olarak kaydedilir.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_COLORS = {
    "DQN":    "#2196F3",
    "RR":     "#9E9E9E",
    "PF":     "#FF9800",
    "Random": "#F44336",
}

SCENARIO_LABELS = {
    "a": "Scenario A\n(10 UE, 50 RB)",
    "b": "Scenario B\n(15 UE, 75 RB)",
    "c": "Scenario C\n(20 UE, 100 RB)",
    "d": "Scenario D\n(20 UE, 50 RB)",
}

DPI = 300


def _save(fig, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


def bar_chart(methods, values, title, ylabel, save_path, errors=None):
    """Tek senaryo için basit bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    colors = [METHOD_COLORS.get(m, "#607D8B") for m in methods]
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white", linewidth=0.8)
    if errors:
        ax.errorbar(x, values, yerr=errors, fmt="none", color="black",
                    capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)


def grouped_bar_chart(scenarios, methods, values, title, ylabel, save_path, errors=None):
    """
    Cok senaryo x cok yontem grouped bar chart.
    values: {method: [val_a, val_b, val_c, val_d]}
    errors: {method: [std_a, ...]}  (opsiyonel)
    """
    n_s = len(scenarios)
    n_m = len(methods)
    x = np.arange(n_s)
    width = 0.7 / n_m
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, method in enumerate(methods):
        vals = values[method]
        errs = errors[method] if errors and method in errors else None
        ax.bar(x + offsets[i], vals, width=width * 0.92,
               color=METHOD_COLORS.get(method, "#607D8B"),
               label=method, edgecolor="white", linewidth=0.5)
        if errs:
            ax.errorbar(x + offsets[i], vals, yerr=errs, fmt="none",
                        color="black", capsize=3, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=11, framealpha=0.8)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, save_path)


def convergence_plot(csv_paths, save_path, title="DQN Training Convergence — All Scenarios"):
    """
    Egitim convergence egrisi (tum senaryolar tek grafik).
    csv_paths: {label: path}  — columns: episode, avg_reward_100
    """
    import pandas as pd
    scenario_colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for (label, path), color in zip(csv_paths.items(), scenario_colors):
        if not os.path.exists(path):
            print(f"  UYARI: {path} bulunamadi, atlandi.")
            continue
        df = pd.read_csv(path)
        ax.plot(df["episode"], df["avg_reward_100"],
                label=label, color=color, linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average Reward (100-ep rolling)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, save_path)


def convergence_plot_single(csv_path, save_path,
                             title="DQN Convergence", color="#2196F3"):
    """Tek senaryo convergence egrisi + plateau noktasi isareti."""
    import pandas as pd
    if not os.path.exists(csv_path):
        print(f"  UYARI: {csv_path} bulunamadi.")
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["episode"], df["avg_reward_100"],
            color=color, linewidth=1.5, alpha=0.85, label="Avg Reward (100-ep)")
    ax.fill_between(df["episode"],
                    df["avg_reward_100"] * 0.97, df["avg_reward_100"] * 1.03,
                    color=color, alpha=0.10)

    final = float(df["avg_reward_100"].iloc[-1])
    threshold = final * 0.90
    plateau_rows = df[df["avg_reward_100"] >= threshold]
    if not plateau_rows.empty:
        p_ep  = int(plateau_rows["episode"].iloc[0])
        p_val = float(plateau_rows["avg_reward_100"].iloc[0])
        ax.axvline(p_ep, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"Plateau (ep {p_ep})")
        offset = max(int(len(df) * 0.04), 50)
        ax.annotate(f"ep {p_ep}", xy=(p_ep, p_val),
                    xytext=(p_ep + offset, p_val * 0.96),
                    fontsize=9, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average Reward (100-ep rolling)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, save_path)


def boxplot_per_ue(data_dict, title, ylabel, save_path):
    """
    Per-UE throughput dagilimi boxplot.
    data_dict: {method: np.array of shape (N,)}  — per-episode per-UE values
    """
    labels = list(data_dict.keys())
    data   = [data_dict[k] for k in labels]
    colors = [METHOD_COLORS.get(k, "#607D8B") for k in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    whiskerprops=dict(linewidth=1.2),
                    medianprops=dict(color="white", linewidth=2.0),
                    flierprops=dict(marker="o", markersize=2, alpha=0.35))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, save_path)

"""
Tum grafikleri tek komutla uretir.

Calistir (repo kokundan):
    python evaluation/generate_all_plots.py

Girdi (otomatik bulunur):
    results/baselines/summary.csv
    results/baselines/per_ue_{scenario}.csv
    results/logs/train_scenario_{a,b,c,d}.csv

Cikti:
    results/plots/comparison_throughput_mbps.png   -- grouped bar (TP)
    results/plots/comparison_fairness.png           -- grouped bar (fairness)
    results/plots/convergence_all.png               -- tum senaryolar bir arada
    results/plots/convergence_scenario_{a,b,c,d}.png -- per-senaryo + plateau
    results/plots/per_ue_boxplot_scenario_{a,b,c,d}.png
    results/plots/summary_table.csv                -- iyilestirme yuzdeleriyle ozet
"""
import os
import sys
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from evaluation.plots import (
    grouped_bar_chart,
    convergence_plot,
    convergence_plot_single,
    boxplot_per_ue,
)

SCENARIOS     = ["a", "b", "c", "d"]
METHODS_ORDER = ["DQN", "RR", "PF", "Random"]
PLOTS_DIR     = os.path.join(ROOT, "results", "plots")
BASELINES_DIR = os.path.join(ROOT, "results", "baselines")
LOGS_DIR      = os.path.join(ROOT, "results", "logs")


# ---------------------------------------------------------------------------
def load_summary():
    path = os.path.join(BASELINES_DIR, "summary.csv")
    if not os.path.exists(path):
        print(f"HATA: {path} bulunamadi.")
        print("Once su komutu calistir:  python evaluation/evaluate_baselines.py")
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
def plot_comparison_bars(df):
    """Throughput ve fairness grouped bar chartlari -- tum senaryolar."""
    methods_present = [m for m in METHODS_ORDER if m in df["scheduler"].unique()]

    for metric, std_col, ylabel, title in [
        ("avg_throughput_mbps", "std_throughput_mbps",
         "Avg Throughput (Mbps)", "System Throughput Comparison"),
        ("avg_fairness", "std_fairness",
         "Avg Jain's Fairness Index", "Fairness Comparison"),
    ]:
        values = {}
        errors = {}
        for m in methods_present:
            vals, stds = [], []
            for s in SCENARIOS:
                row = df[(df["scenario"] == s) & (df["scheduler"] == m)]
                vals.append(float(row[metric].iloc[0])   if not row.empty else 0.0)
                stds.append(float(row[std_col].iloc[0])  if not row.empty else 0.0)
            values[m] = vals
            errors[m] = stds

        save_path = os.path.join(PLOTS_DIR, f"comparison_{metric.replace('avg_', '')}.png")
        grouped_bar_chart(
            scenarios=SCENARIOS,
            methods=methods_present,
            values=values,
            title=title,
            ylabel=ylabel,
            save_path=save_path,
            errors=errors,
        )


# ---------------------------------------------------------------------------
def plot_convergence_curves():
    """DQN egitim convergence egrileri."""
    csv_paths = {}
    for s in SCENARIOS:
        path = os.path.join(LOGS_DIR, f"train_scenario_{s}.csv")
        if os.path.exists(path):
            csv_paths[f"Scenario {s.upper()}"] = path

    if csv_paths:
        convergence_plot(
            csv_paths=csv_paths,
            save_path=os.path.join(PLOTS_DIR, "convergence_all.png"),
        )

    scenario_colors = {"a": "#2196F3", "b": "#FF9800", "c": "#4CAF50", "d": "#9C27B0"}
    for s in SCENARIOS:
        path = os.path.join(LOGS_DIR, f"train_scenario_{s}.csv")
        if os.path.exists(path):
            convergence_plot_single(
                csv_path=path,
                save_path=os.path.join(PLOTS_DIR, f"convergence_scenario_{s}.png"),
                title=f"DQN Convergence — Scenario {s.upper()}",
                color=scenario_colors.get(s, "#2196F3"),
            )


# ---------------------------------------------------------------------------
def plot_boxplots(df):
    """Per-UE throughput dagilimi boxplotlari."""
    for scenario in SCENARIOS:
        pu_path = os.path.join(BASELINES_DIR, f"per_ue_{scenario}.csv")
        if not os.path.exists(pu_path):
            print(f"  Atlandi (bulunamadi): {pu_path}")
            continue

        pu_df      = pd.read_csv(pu_path)
        data_dict  = {}
        for col in pu_df.columns:
            vals = pu_df[col].dropna().values.astype(float)
            if len(vals) > 0:
                data_dict[col] = vals

        if not data_dict:
            continue

        sc_rows = df[df["scenario"] == scenario]
        n_ue = int(sc_rows["n_ue"].iloc[0]) if not sc_rows.empty else "?"
        n_rb = int(sc_rows["n_rb"].iloc[0]) if not sc_rows.empty else "?"

        boxplot_per_ue(
            data_dict=data_dict,
            title=(f"Per-UE Throughput Distribution — "
                   f"Scenario {scenario.upper()} ({n_ue} UE, {n_rb} RB)"),
            ylabel="Per-UE Episode Throughput (Mbps)",
            save_path=os.path.join(PLOTS_DIR, f"per_ue_boxplot_scenario_{scenario}.png"),
        )


# ---------------------------------------------------------------------------
def export_summary_table(df):
    """RR baz alinarak iyilestirme yuzdeleriyle ozet tablo."""
    rows = []
    for scenario in SCENARIOS:
        sc_df  = df[df["scenario"] == scenario]
        rr_row = sc_df[sc_df["scheduler"] == "RR"]
        rr_tp  = float(rr_row["avg_throughput_mbps"].iloc[0]) if not rr_row.empty else 1.0

        for _, row in sc_df.iterrows():
            improv = ((row["avg_throughput_mbps"] - rr_tp) / rr_tp * 100) if rr_tp > 0 else 0.0
            rows.append({
                "scenario":                row["scenario"].upper(),
                "scheduler":               row["scheduler"],
                "n_ue":                    int(row["n_ue"]),
                "n_rb":                    int(row["n_rb"]),
                "avg_throughput_mbps":     round(row["avg_throughput_mbps"], 3),
                "std_throughput_mbps":     round(row["std_throughput_mbps"], 3),
                "avg_fairness":            round(row["avg_fairness"], 4),
                "std_fairness":            round(row["std_fairness"], 4),
                "improvement_over_rr_pct": round(improv, 2),
            })

    out_path = os.path.join(PLOTS_DIR, "summary_table.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Ozet tablo: {out_path}")

    # Terminal ciktisi
    print("\n" + "=" * 82)
    print(f"  {'Scen':<5} {'Sched':<7} {'UE':>3} {'RB':>4} "
          f"{'TP Mbps':>9} {'+-':>7} {'Fairness':>9} {'+-':>7} {'vs RR%':>7}")
    print("  " + "-" * 78)
    for r in rows:
        print(f"  {r['scenario']:<5} {r['scheduler']:<7} {r['n_ue']:>3} {r['n_rb']:>4} "
              f"{r['avg_throughput_mbps']:>9.3f} {r['std_throughput_mbps']:>7.3f} "
              f"{r['avg_fairness']:>9.4f} {r['std_fairness']:>7.4f} "
              f"{r['improvement_over_rr_pct']:>7.2f}%")
    print("=" * 82)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_summary()
    if df is None:
        return

    print("\n[1/4] Karsilastirma bar chartlari uretiliyor...")
    plot_comparison_bars(df)

    print("\n[2/4] Convergence egrileri uretiliyor...")
    plot_convergence_curves()

    print("\n[3/4] Per-UE boxplotlar uretiliyor...")
    plot_boxplots(df)

    print("\n[4/4] Ozet tablo uretiliyor...")
    export_summary_table(df)

    print(f"\nTum grafikler: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()

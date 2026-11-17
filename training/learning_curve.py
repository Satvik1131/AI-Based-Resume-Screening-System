"""
Learning Curve Analysis for Resume-JD Screening (Problem 2: Data Scarcity Diagnostic).

Evaluates whether model performance is bottlenecked by training data volume
or feature representation. Runs stratified 5-fold cross-validation across
sample sizes [20%, 40%, 60%, 80%, 100%].

Usage:
    py training/learning_curve.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all environments
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve

from app.config import FEATURES
from training.export_model import build_dataset_from_csv

RANDOM_STATE = 42
TRAIN_SIZES = [0.2, 0.4, 0.6, 0.8, 1.0]


def render_ascii_curve(sizes_abs, train_means, val_means):
    """Renders a lightweight ASCII terminal chart of the learning curve."""
    print("\n  [ASCII Learning Curve: Validation Macro-F1 vs Training Size]")
    min_v = min(min(val_means), min(train_means))
    max_v = max(max(val_means), max(train_means))
    span = max(max_v - min_v, 0.01)

    chart_width = 35
    for size, t_val, v_val in zip(sizes_abs, train_means, val_means):
        v_pos = int(((v_val - min_v) / span) * chart_width)
        bar = [" "] * (chart_width + 1)
        bar[min(v_pos, chart_width)] = "*"
        bar_str = "".join(bar)
        print(f"  {size:3d} samples |{bar_str}| Val: {v_val:.3f} (Train: {t_val:.3f})")
    print(f"  {'':11} {'|' + '-'*chart_width + '|'} (* = Validation F1)")


def main():
    csv_path = BASE_DIR / "resumeJD2_pairs.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    print("=" * 72)
    print("  DIAGNOSING PROBLEM 2: DATASET SIZE & LEARNING CURVE ANALYSIS".center(72))
    print("=" * 72)

    data = build_dataset_from_csv(str(csv_path))
    X = data[FEATURES]
    y = data["match_label"].str.lower().str.strip()

    print(f"\nTotal Dataset: {len(X)} samples, 13 boundary-aware features")
    for label, count in y.value_counts().items():
        print(f"  - {label:>15}: {count} ({count/len(y)*100:.1f}%)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    print("\nComputing 5-fold cross-validated learning curves across sizes:")
    print("  Train sizes: 20%, 40%, 60%, 80%, 100% of training folds...")

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        train_sizes=TRAIN_SIZES,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print(f"  {'Split %':<8} {'Train N':<9} {'Train Macro-F1':<22} {'Val Macro-F1':<22} {'Gap':<8}")
    print("-" * 72)
    for pct, n_s, tm, ts, vm, vs in zip(TRAIN_SIZES, train_sizes_abs, train_mean, train_std, val_mean, val_std):
        gap = tm - vm
        print(f"  {int(pct*100):<3}%     {n_s:<9} {tm:.3f} (+/-{ts:.3f})         {vm:.3f} (+/-{vs:.3f})         {gap:+.3f}")
    print("-" * 72)

    render_ascii_curve(train_sizes_abs, train_mean, val_mean)

    # ── Slope and Gap Diagnosis ───────────────────────────────────────────────
    last_delta = val_mean[-1] - val_mean[-2]
    first_to_last_delta = val_mean[-1] - val_mean[0]
    final_gap = train_mean[-1] - val_mean[-1]

    print("\n" + "=" * 72)
    print("  EMPIRICAL DIAGNOSIS FOR PROBLEM 2".center(72))
    print("=" * 72)
    print(f"  Final Validation Macro-F1: {val_mean[-1]:.3f} +/- {val_std[-1]:.3f}")
    print(f"  Last Slope (80% -> 100%):   {last_delta:+.3f} F1 points")
    print(f"  Overall Gain (20% -> 100%): {first_to_last_delta:+.3f} F1 points")
    print(f"  Train-Val Generalization Gap: {final_gap:.3f}")

    print("\n  INTERPRETATION:")
    if last_delta > 0.01:
        print("  [>] The validation curve is STILL CLIMBING at 100% data.")
        print("      This proves the model is DATA-CONSTRAINED. Additional training")
        print("      pairs (via pair recombination and synthetic generation) will")
        print("      directly translate into improved generalization and stability.")
    else:
        print("  [>] The validation curve is FLATTENING near 100% data.")
        print("      Raw quantity alone has diminishing returns; the bottleneck is")
        print("      class representation and boundary ambiguity. Targeted Partial Match")
        print("      spec augmentation (Fix 1) is significantly more critical than random data.")

    if final_gap > 0.15:
        print(f"  [>] Overfitting Gap ({final_gap:.3f}) is high.")
        print("      With only ~120 training Partial Match cases, trees memorize sample-specific")
        print("      phrasings. Expanding the border region will narrow this generalization gap.")

    # ── Plot Generation ───────────────────────────────────────────────────────
    out_dir = BASE_DIR / "presentation_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot = out_dir / "learning_curve.png"

    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    plt.plot(train_sizes_abs, train_mean, "o-", color="#1f77b4", label="Training Macro-F1", lw=2)
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#1f77b4")

    plt.plot(train_sizes_abs, val_mean, "s-", color="#2ca02c", label="Cross-Validation Macro-F1", lw=2)
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#2ca02c")

    plt.title("Learning Curve: Macro F1 vs Training Set Size (5-Fold CV)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Training Samples", fontsize=11)
    plt.ylabel("Macro-Averaged F1 Score", fontsize=11)
    plt.ylim(0.50, 1.02)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(out_plot)
    plt.close()
    print(f"\n  Saved learning curve visualization to:\n    {out_plot}")
    print("=" * 72)


if __name__ == "__main__":
    main()

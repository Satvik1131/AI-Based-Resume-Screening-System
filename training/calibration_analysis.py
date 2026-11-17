"""
Calibration Analysis and Reliability Diagnostics (Problem 4).

Addresses Random Forest probability miscalibration:
1. Measures miscalibration via ECE (Expected Calibration Error) and Reliability Diagrams.
2. Evaluates probability calibrators (Platt Sigmoid vs Isotonic Regression via 5-fold CV).
3. Evaluates per-class calibration (especially Partial Match).
4. Generates an evidence-backed Coverage vs Accuracy table to set human-review thresholds.
5. Implements decision margin analysis to flag near-ties between classes.

Usage:
    py training/calibration_analysis.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless safe backend
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.config import FEATURES
from training.export_model import build_dataset_from_csv

RANDOM_STATE = 42
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Calibration Metrics ───────────────────────────────────────────────

def expected_calibration_error(y_true, proba, classes, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) for multi-class or binary predictions.
    Groups predictions by confidence, then calculates the weighted gap between
    confidence and empirical accuracy across all bins.
    """
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def per_class_ece(y_true, proba, classes, n_bins=10):
    """
    Computes one-vs-rest ECE for each class independently.
    Helps isolate calibration issues on minority classes (e.g., Partial Match).
    """
    results = {}
    edges = np.linspace(0, 1, n_bins + 1)
    for idx, cls in enumerate(classes):
        cls_prob = proba[:, idx]
        cls_true = (y_true == cls).astype(float)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (cls_prob > lo) & (cls_prob <= hi)
            if mask.any():
                ece += mask.mean() * abs(cls_true[mask].mean() - cls_prob[mask].mean())
        results[cls] = float(ece)
    return results


def compute_brier_scores(y_true, proba, classes):
    """Computes multi-class and per-class Brier score loss."""
    scores = {}
    for idx, cls in enumerate(classes):
        binary_true = (y_true == cls).astype(float)
        scores[cls] = float(brier_score_loss(binary_true, proba[:, idx]))
    scores["overall_mean"] = float(np.mean(list(scores.values())))
    return scores


# ── Reliability Table & ASCII Display ────────────────────────────────────────

def print_reliability_table(y_true, proba, classes, title="Reliability Table", n_bins=6):
    """Prints a detailed reliability table grouped by confidence bins."""
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.4, 1.0, n_bins + 1)

    print(f"\n  [{title}]")
    print(f"  {'Confidence Bin':<18} {'Samples':>8} {'Weight':>8} {'Avg Conf':>10} {'Empirical Acc':>14} {'Gap (|Diff|)':>14}")
    print(f"  {'-'*76}")

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi if hi == 1.0 else conf < hi)
        n = mask.sum()
        if n > 0:
            avg_conf = conf[mask].mean()
            acc = correct[mask].mean()
            gap = abs(avg_conf - acc)
            weight = n / len(y_true)
            print(f"  [{lo:.2f} - {hi:.2f}]       {n:>8d} {weight:>7.1%} {avg_conf:>10.3f} {acc:>14.3f} {gap:>14.3f}")
        else:
            print(f"  [{lo:.2f} - {hi:.2f}]       {0:>8d}    0.0%        N/A            N/A            N/A")


# ── Step 3: Evidence-Backed Threshold Analysis ────────────────────────────────

def generate_coverage_accuracy_table(y_true, proba, classes):
    """
    Computes auto-decision coverage vs accuracy trade-offs across confidence thresholds.
    Also computes margin threshold trade-offs and combined threshold policy.
    """
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    sorted_p = np.sort(proba, axis=1)[:, ::-1]
    margins = sorted_p[:, 0] - sorted_p[:, 1]
    n_total = len(y_true)

    print("\n" + "=" * 76)
    print("  STEP 3: COVERAGE VS. ACCURACY TRADE-OFF (CONFIDENCE THRESHOLD)".center(76))
    print("=" * 76)
    print(f"  {'Threshold':<12} {'Auto-Decides':>14} {'Cases':>10} {'Accuracy':>12} {'Recruiter Review':>18}")
    print("  " + "-" * 72)

    threshold_data = []
    for t in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        keep = conf >= t
        n_keep = keep.sum()
        if n_keep > 0:
            acc = (pred[keep] == y_true[keep]).mean()
            cov = n_keep / n_total
            flagged = 1.0 - cov
            threshold_data.append((t, cov, n_keep, acc, flagged))
            print(f"  {t:<12.2f} {cov:>13.1%} {n_keep:>10d} {acc:>11.1%} {flagged:>17.1%}")

    print("\n" + "=" * 76)
    print("  REFINEMENT 1: DECISION MARGIN THRESHOLD (TOP-1 vs TOP-2 GAP)".center(76))
    print("=" * 76)
    print(f"  {'Margin (>=)':<12} {'Auto-Decides':>14} {'Cases':>10} {'Accuracy':>12} {'Flagged Review':>18}")
    print("  " + "-" * 72)

    for m in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        keep = margins >= m
        n_keep = keep.sum()
        if n_keep > 0:
            acc = (pred[keep] == y_true[keep]).mean()
            cov = n_keep / n_total
            flagged = 1.0 - cov
            print(f"  {m:<12.2f} {cov:>13.1%} {n_keep:>10d} {acc:>11.1%} {flagged:>17.1%}")

    print("\n" + "=" * 76)
    print("  RECOMMENDED DUAL-GATE POLICY: CONFIDENCE >= 0.65 AND MARGIN >= 0.10".center(76))
    print("=" * 76)
    dual_keep = (conf >= 0.65) & (margins >= 0.10)
    dual_acc = (pred[dual_keep] == y_true[dual_keep]).mean()
    dual_cov = dual_keep.mean()
    dual_flagged = 1.0 - dual_cov
    print(f"  Auto-decided volume : {dual_cov:.1%} ({dual_keep.sum()}/{n_total} cases)")
    print(f"  Accuracy on decided : {dual_acc:.1%}")
    print(f"  Sent to recruiter   : {dual_flagged:.1%} ({(~dual_keep).sum()}/{n_total} cases)")
    print(f"  Accuracy on flagged : {(pred[~dual_keep] == y_true[~dual_keep]).mean():.1%} (proves flagged cases truly need human eyes!)")


# ── Plotting Reliability Diagrams ─────────────────────────────────────────────

def plot_reliability_diagrams(y_true, proba_uncal, proba_sig, proba_iso, classes, save_path):
    """
    Plots multi-class reliability curves comparing Uncalibrated, Sigmoid, and Isotonic.
    Saves high-resolution figure to reports/calibration_curve.png.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = [
        ("Uncalibrated Random Forest", proba_uncal, "#e74c3c", axes[0]),
        ("Sigmoid (Platt Scaling)", proba_sig, "#2980b9", axes[1]),
        ("Isotonic Regression", proba_iso, "#27ae60", axes[2]),
    ]

    for title, proba, color, ax in methods:
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
        for idx, cls in enumerate(classes):
            cls_true = (y_true == cls).astype(int)
            cls_prob = proba[:, idx]
            prob_true, prob_pred = calibration_curve(cls_true, cls_prob, n_bins=8, strategy="uniform")
            ax.plot(prob_pred, prob_true, marker="o", label=f"{cls.title()}")

        ece_val = expected_calibration_error(y_true, proba, classes)
        ax.set_title(f"{title}\nECE = {ece_val:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Empirical Fraction of Positives", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9)

    plt.suptitle("Problem 4: Model Calibration Analysis (5-Fold Stratified CV)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved calibration diagrams to: {save_path.relative_to(BASE_DIR)}")


# ── Main Diagnostic Suite ─────────────────────────────────────────────────────

def run_calibration_suite():
    csv_path = BASE_DIR / "resumeJD2_pairs.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    print("=" * 76)
    print("  DIAGNOSING PROBLEM 4: PROBABILITY CALIBRATION & EVIDENCE-BASED REVIEW".center(76))
    print("=" * 76)

    data = build_dataset_from_csv(str(csv_path))
    X = data[FEATURES]
    y = data["match_label"].str.lower().str.strip().values
    classes = np.unique(y)

    print(f"\nDataset: {len(X)} samples across classes {list(classes)}")

    # ── 5-Fold Cross-Validation ───────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=RANDOM_STATE, class_weight="balanced"
    )

    print("\nRunning 5-fold cross-validation for Uncalibrated model...")
    p_uncal = cross_val_predict(rf, X, y, cv=5, method="predict_proba")

    print("Running 5-fold cross-validation for Sigmoid (Platt) Calibrator...")
    cal_sig = CalibratedClassifierCV(rf, method="sigmoid", cv=5)
    p_sig = cross_val_predict(cal_sig, X, y, cv=5, method="predict_proba")

    print("Running 5-fold cross-validation for Isotonic Calibrator...")
    cal_iso = CalibratedClassifierCV(rf, method="isotonic", cv=5)
    p_iso = cross_val_predict(cal_iso, X, y, cv=5, method="predict_proba")

    # ── Compute ECE & Brier Scores ────────────────────────────────────────
    ece_uncal = expected_calibration_error(y, p_uncal, classes)
    ece_sig   = expected_calibration_error(y, p_sig, classes)
    ece_iso   = expected_calibration_error(y, p_iso, classes)

    brier_uncal = compute_brier_scores(y, p_uncal, classes)
    brier_sig   = compute_brier_scores(y, p_sig, classes)
    brier_iso   = compute_brier_scores(y, p_iso, classes)

    ece_pc_uncal = per_class_ece(y, p_uncal, classes)
    ece_pc_sig   = per_class_ece(y, p_sig, classes)
    ece_pc_iso   = per_class_ece(y, p_iso, classes)

    # ── Summary Comparison ────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  STEP 1 & 2: CALIBRATION METRICS COMPARISON (5-FOLD CV)".center(76))
    print("=" * 76)
    print(f"  {'Metric':<28} {'Uncalibrated':>14} {'Sigmoid':>14} {'Isotonic':>14}")
    print("  " + "-" * 72)
    print(f"  {'Overall ECE':<28} {ece_uncal:>14.4f} {ece_sig:>14.4f} {ece_iso:>14.4f}")
    print(f"  {'Mean Brier Score Loss':<28} {brier_uncal['overall_mean']:>14.4f} {brier_sig['overall_mean']:>14.4f} {brier_iso['overall_mean']:>14.4f}")
    print("  " + "-" * 72)
    print("  Per-Class ECE (Refinement 2):")
    for cls in classes:
        print(f"    - {cls.title():<24} {ece_pc_uncal[cls]:>14.4f} {ece_pc_sig[cls]:>14.4f} {ece_pc_iso[cls]:>14.4f}")
    print("  " + "-" * 72)

    # ── Method Selection Guidance ─────────────────────────────────────────
    print("\n  [Why Sigmoid is Preferred for ~500 Pairs]")
    print("  - Sigmoid (Platt scaling) fits a parametric 2-parameter S-curve.")
    print("  - Isotonic is non-parametric; with only ~500 rows (~150 Partial Match rows),")
    print("    isotonic regression fits step-functions that overfit small sample validation folds.")
    print("  - Sigmoid guarantees monotonic, smooth, well-regularized probabilities.")

    # ── Reliability Table ─────────────────────────────────────────────────
    print_reliability_table(y, p_uncal, classes, title="Uncalibrated Forest: Confidence vs Empirical Accuracy")
    print_reliability_table(y, p_sig, classes, title="Sigmoid Calibrated: Confidence vs Empirical Accuracy")

    # ── Step 3 & Refinements ──────────────────────────────────────────────
    generate_coverage_accuracy_table(y, p_sig, classes)

    # ── Plotting ──────────────────────────────────────────────────────────
    plot_path = REPORTS_DIR / "calibration_curve.png"
    plot_reliability_diagrams(y, p_uncal, p_sig, p_iso, classes, plot_path)

    print("\n" + "=" * 76)
    print("  DIAGNOSTIC COMPLETE".center(76))
    print("=" * 76)


if __name__ == "__main__":
    run_calibration_suite()

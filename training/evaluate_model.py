"""
Evaluates the two-stage cascade classifier vs. the single-model baseline.

Performs 5-fold stratified cross-validation and reports:
  - Per-class precision / recall / F1 for both approaches
  - Confusion matrices
  - Head-to-head Partial Match comparison

Usage:
    python training/evaluate_model.py
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from app.config import FEATURES
from training.export_model import build_dataset_from_csv

RANDOM_STATE = 42
LABELS = ["match", "partial match", "no match"]


from sklearn.calibration import CalibratedClassifierCV
from training.calibration_analysis import expected_calibration_error, per_class_ece

# ── Single-model baseline (calibrated) ────────────────────────────────────────

def predict_single(X_train, y_train, X_test):
    """Baseline: single 3-way Random Forest calibrated with Platt scaling (cv=5)."""
    base_rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    calibrated = CalibratedClassifierCV(
        estimator=base_rf, method="sigmoid", cv=5
    )
    calibrated.fit(X_train, y_train)
    proba = calibrated.predict_proba(X_test)
    preds = calibrated.predict(X_test)
    return preds, proba, calibrated.classes_


# ── Cascade classifier (calibrated) ───────────────────────────────────────────

def _calibrate_threshold(X, y_binary, n_splits=5):
    """Find Stage-A threshold via OOF calibrated probabilities."""
    kfold = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    oof_proba = np.zeros(len(y_binary))

    for train_idx, val_idx in kfold.split(X, y_binary):
        base_rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=4,
            random_state=RANDOM_STATE, class_weight="balanced"
        )
        fold_model = CalibratedClassifierCV(
            estimator=base_rf, method="sigmoid", cv=5
        )
        fold_model.fit(X.iloc[train_idx], y_binary.iloc[train_idx])
        oof_proba[val_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.35, 0.80, 0.01):
        preds = (oof_proba >= t).astype(int)
        tp = ((preds == 1) & (y_binary == 1)).sum()
        fp = ((preds == 1) & (y_binary == 0)).sum()
        fn = ((preds == 0) & (y_binary == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if rec >= 0.88 and f1 > best_f1:
            best_f1 = f1
            best_t = t
    return round(best_t, 2)


def predict_cascade(X_train, y_train, X_test):
    """Two-stage cascade with calibrated probabilities at both stages."""
    # Stage A: Calibrated Match vs Rest
    y_a = (y_train == "match").astype(int)
    rf_a = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=4,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    model_a = CalibratedClassifierCV(
        estimator=rf_a, method="sigmoid", cv=5
    )
    model_a.fit(X_train, y_a)
    threshold = _calibrate_threshold(X_train, y_a)

    # Stage B: Calibrated Partial vs No-Match (non-Match subset)
    mask = y_train != "match"
    rf_b = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=4,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    model_b = CalibratedClassifierCV(
        estimator=rf_b, method="sigmoid", cv=5
    )
    model_b.fit(X_train[mask], y_train[mask])

    # Cascade inference with probability recording
    proba_a = model_a.predict_proba(X_test)
    match_idx = list(model_a.classes_).index(1)
    y_pred = []
    confs = []
    margins = []

    proba_b_all = model_b.predict_proba(X_test)

    for i in range(len(X_test)):
        p_match = proba_a[i][match_idx]
        if p_match >= threshold:
            y_pred.append("match")
            confs.append(p_match)
            margins.append(abs(p_match - (1.0 - p_match)))
        else:
            p_b = proba_b_all[i]
            sorted_idx = np.argsort(p_b)[::-1]
            pred_cls = model_b.classes_[sorted_idx[0]]
            y_pred.append(pred_cls)
            top_p = p_b[sorted_idx[0]]
            second_p = p_b[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0
            confs.append(top_p)
            margins.append(top_p - second_p)

    return np.array(y_pred), threshold, np.array(confs), np.array(margins)


# ── Pretty-print confusion matrix ────────────────────────────────────────────

def _print_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    header = f"{'':>15}" + "".join(f"{l:>15}" for l in LABELS)
    print(header)
    for i, label in enumerate(LABELS):
        row = f"  {label:>13}" + "".join(f"{cm[i][j]:>15}" for j in range(len(LABELS)))
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    csv_path = BASE_DIR / "resumeJD2_pairs.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    print("Building dataset (this may take a few minutes)...\n")
    data = build_dataset_from_csv(str(csv_path))
    X = data[FEATURES]
    y = data["match_label"].str.lower().str.strip()

    print(f"\nDataset: {len(X)} samples")
    for label, count in y.value_counts().items():
        print(f"  {label:>15}: {count}")

    # ── Stratified 5-fold CV ──────────────────────────────────────────────
    n_folds = 5
    kfold = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE
    )

    all_true_s, all_pred_s = [], []   # single-model
    all_proba_s = []
    all_true_c, all_pred_c = [], []   # cascade
    all_conf_c, all_margin_c = [], []
    classes_s = None

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pred_s, proba_s, classes_s = predict_single(X_tr, y_tr, X_te)
        all_true_s.extend(y_te)
        all_pred_s.extend(pred_s)
        all_proba_s.extend(proba_s)

        pred_c, threshold, confs_c, margins_c = predict_cascade(X_tr, y_tr, X_te)
        all_true_c.extend(y_te)
        all_pred_c.extend(pred_c)
        all_conf_c.extend(confs_c)
        all_margin_c.extend(margins_c)

        print(f"  Fold {fold}/{n_folds} done  "
              f"(cascade threshold: {threshold})")

    all_true_s = np.array(all_true_s)
    all_pred_s = np.array(all_pred_s)
    all_proba_s = np.array(all_proba_s)
    all_true_c = np.array(all_true_c)
    all_pred_c = np.array(all_pred_c)
    all_conf_c = np.array(all_conf_c)
    all_margin_c = np.array(all_margin_c)

    # ── Baseline report ───────────────────────────────────────────────────
    W = 70
    print("\n" + "=" * W)
    print("  BASELINE: Single 3-way Calibrated Random Forest (13 features)".center(W))
    print("=" * W)
    print(classification_report(
        all_true_s, all_pred_s, labels=LABELS, digits=3, zero_division=0
    ))
    ece_baseline = expected_calibration_error(all_true_s, all_proba_s, classes_s)
    print(f"  Expected Calibration Error (ECE): {ece_baseline:.4f}")
    print("\nConfusion Matrix (rows = actual, cols = predicted):")
    _print_cm(all_true_s, all_pred_s)

    # ── Cascade report ────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  CASCADE: Two-Stage Calibrated Classifier (13 features)".center(W))
    print("=" * W)
    print(classification_report(
        all_true_c, all_pred_c, labels=LABELS, digits=3, zero_division=0
    ))
    print("Confusion Matrix (rows = actual, cols = predicted):")
    _print_cm(all_true_c, all_pred_c)

    # ── Partial Match head-to-head ────────────────────────────────────────
    print("\n" + "-" * W)
    print("  PARTIAL MATCH FOCUS".center(W))
    print("-" * W)

    p_s, r_s, f_s, _ = precision_recall_fscore_support(
        all_true_s, all_pred_s, labels=["partial match"],
        average="macro", zero_division=0
    )
    p_c, r_c, f_c, _ = precision_recall_fscore_support(
        all_true_c, all_pred_c, labels=["partial match"],
        average="macro", zero_division=0
    )

    print(f"  {'Metric':<20} {'Baseline':>10} {'Cascade':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Precision':<20} {p_s:>10.3f} {p_c:>10.3f} {p_c-p_s:>+10.3f}")
    print(f"  {'Recall':<20} {r_s:>10.3f} {r_c:>10.3f} {r_c-r_s:>+10.3f}")
    print(f"  {'F1-score':<20} {f_s:>10.3f} {f_c:>10.3f} {f_c-f_s:>+10.3f}")

    # ── Human Review Policy from Evidence (Step 3) ────────────────────────
    print("\n" + "=" * W)
    print("  HUMAN REVIEW COVERAGE VS ACCURACY TRADE-OFF (CASCADE)".center(W))
    print("=" * W)
    print(f"  {'Threshold':<12} {'Auto-Decides':>14} {'Cases':>10} {'Accuracy':>12} {'Flagged':>14}")
    print("  " + "-" * 66)

    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        keep = all_conf_c >= t
        if keep.any():
            acc = (all_pred_c[keep] == all_true_c[keep]).mean()
            cov = keep.mean()
            print(f"  {t:<12.2f} {cov:>13.1%} {keep.sum():>10d} {acc:>11.1%} {(1.0-cov):>13.1%}")

    print("\n  Recommended Dual-Gate Policy (Confidence >= 0.65 and Margin >= 0.10):")
    dual_keep = (all_conf_c >= 0.65) & (all_margin_c >= 0.10)
    dual_acc = (all_pred_c[dual_keep] == all_true_c[dual_keep]).mean()
    print(f"  Auto-decides {dual_keep.mean():.1%} of resumes at {dual_acc:.1%} accuracy, sending {(~dual_keep).mean():.1%} to a human reviewer.")
    print()


if __name__ == "__main__":
    main()

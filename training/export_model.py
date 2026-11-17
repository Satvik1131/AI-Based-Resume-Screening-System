"""
Exports the two-stage cascade RandomForest classifier.

  Stage A: Match vs Rest (binary) — high-recall filter for confident Matches
  Stage B: Partial Match vs No Match (binary) — trained only on non-Match data

Saves to models/:
  stage_a_model.joblib   — binary classifier for confident Match detection
  stage_b_model.joblib   — binary classifier for Partial vs No-Match
  model_config.joblib    — features, calibrated threshold, metadata
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from app.config import (
    FEATURES, MODELS_DIR, EXPERIENCE_UNKNOWN_SENTINEL,
    CONFIDENCE_THRESHOLD, MIN_SKILL_EVIDENCE,
    OVERQUALIFICATION_RATIO_THRESHOLD
)
from app.services.skill_matcher import (
    extract_skills_exact, get_matched_skills, get_missing_skills,
    skill_match_score, identify_domain, compute_semantic_similarity
)
from app.services.feature_builder import extract_jd_experience

RANDOM_STATE = 42


# ── Dataset Construction ──────────────────────────────────────────────────────

def build_dataset_from_csv(csv_path: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Engineers the full 13-feature vector for every resume-JD pair in the CSV.
    Includes the 3 boundary-aware features (skill_gap_signed, experience_gap_years,
    skill_balance_ratio) alongside the original 10.
    Caches results to data/features_13d_cache.csv for instant repeated loads.
    """
    cache_path = BASE_DIR / "data" / "features_13d_cache.csv"
    if use_cache and cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        raw_df = pd.read_csv(csv_path).drop_duplicates()
        if len(cached_df) == len(raw_df):
            print(f"Loaded {len(cached_df)} engineered feature vectors from cache ({cache_path.name})")
            return cached_df

    df = pd.read_csv(csv_path).drop_duplicates().copy()
    rows = []
    print(f"Engineering features for {len(df)} resume-JD pairs...")

    for i, row in df.iterrows():
        res_text = str(row["resume_text"])
        jd_text = str(row["job_description"])

        res_skills = extract_skills_exact(res_text)
        jd_skills = extract_skills_exact(jd_text)
        matched = get_matched_skills(res_skills, jd_skills)
        missing = get_missing_skills(res_skills, jd_skills)
        sk_score = skill_match_score(res_skills, jd_skills)

        # Regex experience fallback for training data
        res_exp_match = pd.Series([res_text]).str.extract(r'(\d+)\+?\s*years?')[0]
        res_exp = float(res_exp_match.iloc[0]) if not res_exp_match.isna().iloc[0] else None
        jd_exp = extract_jd_experience(jd_text)

        res_exp_mentioned = int(res_exp is not None)
        jd_exp_mentioned = int(not pd.isna(jd_exp))

        if res_exp_mentioned and jd_exp_mentioned and jd_exp > 0:
            exp_score = round(min(res_exp / jd_exp, 1.0), 2)
            exp_ratio = round(res_exp / jd_exp, 2)
        else:
            exp_score = EXPERIENCE_UNKNOWN_SENTINEL
            exp_ratio = EXPERIENCE_UNKNOWN_SENTINEL

        unstated_exp_senior = int(
            not res_exp_mentioned and not pd.isna(jd_exp) and jd_exp >= 5
        )

        res_domain = identify_domain(res_skills)
        jd_domain = identify_domain(jd_skills)
        domain_match = int(res_domain == jd_domain and res_domain != "Unknown")

        sim_score = compute_semantic_similarity(res_text, jd_text)

        # ── Boundary-aware features ───────────────────────────────────────
        skill_gap_signed = len(res_skills) - len(jd_skills)

        if res_exp_mentioned and jd_exp_mentioned:
            experience_gap_years = round(res_exp - jd_exp, 2)
        else:
            experience_gap_years = EXPERIENCE_UNKNOWN_SENTINEL

        total_jd = len(matched) + len(missing)
        if total_jd > 0:
            skill_balance_ratio = round(
                min(len(matched), len(missing)) / total_jd, 4
            )
        else:
            skill_balance_ratio = 0.0

        rows.append({
            "skill_match_score": sk_score,
            "matched_skill_count": len(matched),
            "missing_skill_count": len(missing),
            "experience_match_score": exp_score,
            "experience_ratio": exp_ratio,
            "resume_experience_mentioned": res_exp_mentioned,
            "jd_experience_mentioned": jd_exp_mentioned,
            "unstated_experience_for_senior_role": unstated_exp_senior,
            "semantic_similarity_score": sim_score,
            "domain_match": domain_match,
            "skill_gap_signed": skill_gap_signed,
            "experience_gap_years": experience_gap_years,
            "skill_balance_ratio": skill_balance_ratio,
            "match_label": row["match_label"]
        })

    result_df = pd.DataFrame(rows)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(cache_path, index=False)
        print(f"Cached {len(result_df)} feature vectors to {cache_path.name}")
    except Exception as e:
        print(f"Warning: could not write feature cache: {e}")
    return result_df


from sklearn.calibration import CalibratedClassifierCV
from app.config import (
    FEATURES, MODELS_DIR, EXPERIENCE_UNKNOWN_SENTINEL,
    CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD, CALIBRATION_METHOD,
    MIN_SKILL_EVIDENCE, OVERQUALIFICATION_RATIO_THRESHOLD
)

# ── Threshold Calibration ─────────────────────────────────────────────────────

def calibrate_stage_a_threshold(X, y_binary, n_splits=5):
    """
    Uses stratified K-fold CV with calibrated classifiers to find the Stage-A
    honest probability threshold that maximises Match F1 while keeping recall >= 88%.

    This ensures true Matches are NOT accidentally sent to Stage B (where they
    would be forced into Partial or No-Match).
    """
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
            estimator=base_rf, method=CALIBRATION_METHOD, cv=5
        )
        fold_model.fit(X.iloc[train_idx], y_binary.iloc[train_idx])
        oof_proba[val_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]

    best_threshold = 0.5
    best_f1 = 0.0

    for t in np.arange(0.35, 0.80, 0.01):
        preds = (oof_proba >= t).astype(int)
        tp = ((preds == 1) & (y_binary == 1)).sum()
        fp = ((preds == 1) & (y_binary == 0)).sum()
        fn = ((preds == 0) & (y_binary == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)

        if recall >= 0.88 and f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"  Calibrated Stage A threshold: {best_threshold:.2f} "
          f"(F1: {best_f1:.3f})")
    return round(best_threshold, 2)


# ── Training & Export ─────────────────────────────────────────────────────────

def train_and_export():
    csv_path = BASE_DIR / "resumeJD2_pairs.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    data = build_dataset_from_csv(str(csv_path))
    X = data[FEATURES]
    y = data["match_label"].str.lower().str.strip()

    print(f"\nClass distribution:")
    for label, count in y.value_counts().items():
        print(f"  {label:>15}: {count}")

    # ── Stage A: Match (1) vs Rest (0) Calibrated ─────────────────────────
    print("\n[Stage A] Training & calibrating Match vs Rest binary classifier...")
    y_stage_a = (y == "match").astype(int)

    rf_a = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=4,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    model_a = CalibratedClassifierCV(
        estimator=rf_a, method=CALIBRATION_METHOD, cv=5
    )
    model_a.fit(X, y_stage_a)

    cascade_threshold = calibrate_stage_a_threshold(X, y_stage_a)

    # ── Stage B: Partial Match vs No Match (non-Match subset only) ────────
    print("\n[Stage B] Training & calibrating Partial Match vs No Match classifier...")
    non_match_mask = y != "match"
    X_stage_b = X[non_match_mask].reset_index(drop=True)
    y_stage_b = y[non_match_mask].reset_index(drop=True)

    print(f"  Stage B training set: {len(X_stage_b)} samples")
    for label, count in y_stage_b.value_counts().items():
        print(f"    {label:>15}: {count}")

    rf_b = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=4,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    model_b = CalibratedClassifierCV(
        estimator=rf_b, method=CALIBRATION_METHOD, cv=5
    )
    model_b.fit(X_stage_b, y_stage_b)

    # ── Legacy single-model fallback (calibrated) ─────────────────────────
    print("\n[Baseline] Training & calibrating 3-way multi-class model...")
    rf_single = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    model_single = CalibratedClassifierCV(
        estimator=rf_single, method=CALIBRATION_METHOD, cv=5
    )
    model_single.fit(X, y)

    # ── Save everything ───────────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model_a, MODELS_DIR / "stage_a_model.joblib")
    joblib.dump(model_b, MODELS_DIR / "stage_b_model.joblib")
    joblib.dump(model_single, MODELS_DIR / "rf_model.joblib")
    joblib.dump({
        "features": FEATURES,
        "cascade_threshold": cascade_threshold,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD,
        "calibration_method": CALIBRATION_METHOD,
        "is_calibrated": True,
        "min_skill_evidence": MIN_SKILL_EVIDENCE,
        "overqualification_ratio_threshold": OVERQUALIFICATION_RATIO_THRESHOLD,
        "model_type": "cascade_v1_calibrated"
    }, MODELS_DIR / "model_config.joblib")

    print(f"\nSuccessfully trained & exported calibrated models:")
    print(f"  Stage A  -> {MODELS_DIR / 'stage_a_model.joblib'}")
    print(f"  Stage B  -> {MODELS_DIR / 'stage_b_model.joblib'}")
    print(f"  Baseline -> {MODELS_DIR / 'rf_model.joblib'}")
    print(f"  Config   -> {MODELS_DIR / 'model_config.joblib'}")
    print(f"  Cascade threshold: {cascade_threshold}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Margin threshold: {MARGIN_THRESHOLD}")
    print(f"  Calibration method: {CALIBRATION_METHOD}")


if __name__ == "__main__":
    train_and_export()

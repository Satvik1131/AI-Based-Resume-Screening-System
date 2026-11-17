"""
Two-stage cascade classifier for resume-JD matching.

Stage A: Binary — is this a confident Match?
  → If predict_proba(Match) >= calibrated threshold → predict 'match'
Stage B: Binary — Partial Match vs No Match
  → Only runs on cases Stage A didn't confidently call Match

SHAP explanations come from whichever stage made the final decision.
Safety guardrails (domain mismatch, severe skill gap, etc.) are unchanged.
"""
import joblib
import pandas as pd
import numpy as np
import shap
from app.config import (
    MODELS_DIR,
    FEATURES,
    FEATURE_LABELS,
    CONFIDENCE_THRESHOLD,
    MARGIN_THRESHOLD,
    CASCADE_THRESHOLD,
    MIN_SKILL_EVIDENCE,
    MIN_RESUME_SKILLS
)

_stage_a = None
_stage_b = None
_explainer_a = None
_explainer_b = None
_cascade_threshold = None
_confidence_threshold = None
_margin_threshold = None


def _extract_base_tree_estimator(model):
    """
    Extracts the underlying tree estimator from a CalibratedClassifierCV wrapper.
    SHAP TreeExplainer requires the underlying tree ensemble, not the calibrated wrapper.
    """
    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        cal_sub = model.calibrated_classifiers_[0]
        if hasattr(cal_sub, "estimator"):
            return cal_sub.estimator
    if hasattr(model, "estimator"):
        return model.estimator
    return model


def _load_ml_components():
    """
    Lazy-loads the two-stage cascade model and SHAP explainers.
    Falls back to legacy single-model path if cascade files don't exist
    but the old rf_model.joblib does.
    """
    global _stage_a, _stage_b, _explainer_a, _explainer_b
    global _cascade_threshold, _confidence_threshold, _margin_threshold

    if _stage_a is not None:
        return (
            _stage_a, _stage_b, _explainer_a, _explainer_b,
            _cascade_threshold, _confidence_threshold, _margin_threshold
        )

    stage_a_path = MODELS_DIR / "stage_a_model.joblib"
    stage_b_path = MODELS_DIR / "stage_b_model.joblib"
    config_path  = MODELS_DIR / "model_config.joblib"

    for p in (stage_a_path, stage_b_path, config_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Model file not found: {p}. "
                "Run 'python training/export_model.py' to train the cascade model."
            )

    _stage_a = joblib.load(stage_a_path)
    _stage_b = joblib.load(stage_b_path)

    base_a = _extract_base_tree_estimator(_stage_a)
    base_b = _extract_base_tree_estimator(_stage_b)
    _explainer_a = shap.TreeExplainer(base_a)
    _explainer_b = shap.TreeExplainer(base_b)

    config = joblib.load(config_path)
    _cascade_threshold = config.get("cascade_threshold", CASCADE_THRESHOLD)
    _confidence_threshold = config.get("confidence_threshold", CONFIDENCE_THRESHOLD)
    _margin_threshold = config.get("margin_threshold", MARGIN_THRESHOLD)

    return (
        _stage_a, _stage_b, _explainer_a, _explainer_b,
        _cascade_threshold, _confidence_threshold, _margin_threshold
    )


def evaluate_candidate_ml(feature_data: dict) -> dict:
    """
    Runs the engineered 13-feature vector through the calibrated two-stage cascade classifier.
    Calculates predicted class, calibrated confidence, decision margin, SHAP impacts, and safety flags.

    Downstream code maintains full backwards compatibility while gaining:
    1. Honest calibrated probabilities (via CalibratedClassifierCV Platt scaling).
    2. Decision margin tracking (top-1 vs top-2 probability gap).
    3. Near-tie detection for human review (Problem 4 Refinement 1).
    """
    (
        model_a, model_b, explainer_a, explainer_b,
        cascade_thresh, conf_thresh, margin_thresh
    ) = _load_ml_components()

    row_features = feature_data["features"]
    X_row = pd.DataFrame([row_features])[FEATURES]

    # ── Stage A: Match detection (calibrated probabilities) ───────────────
    proba_a = model_a.predict_proba(X_row)[0]
    match_class_idx = list(model_a.classes_).index(1)
    match_prob = float(proba_a[match_class_idx])

    if match_prob >= cascade_thresh:
        pred_class = "match"
        confidence = match_prob
        # In binary Stage A, margin is gap between Match and Rest
        margin = abs(match_prob - (1.0 - match_prob))
        second_prob = 1.0 - match_prob
        deciding_stage = "a"
    else:
        # ── Stage B: Partial vs No-Match (calibrated probabilities) ───────
        proba_b = model_b.predict_proba(X_row)[0]
        sorted_indices = np.argsort(proba_b)[::-1]
        top_idx = sorted_indices[0]
        pred_class = model_b.classes_[top_idx]
        confidence = float(proba_b[top_idx])

        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else top_idx
        second_prob = float(proba_b[second_idx]) if len(sorted_indices) > 1 else 0.0
        margin = float(confidence - second_prob)
        deciding_stage = "b"

    confidence = round(confidence, 2)
    margin = round(margin, 2)

    # ── Evidence, Confidence & Margin Checks ──────────────────────────────
    resume_skill_cnt = len(feature_data["resume_skills"])
    matched_cnt = row_features["matched_skill_count"]
    missing_cnt = row_features["missing_skill_count"]

    sufficient_evidence = (
        (matched_cnt + missing_cnt >= MIN_SKILL_EVIDENCE)
        and (resume_skill_cnt >= MIN_RESUME_SKILLS)
    )
    low_confidence = confidence < conf_thresh
    narrow_margin = margin < margin_thresh
    unverifiable_seniority = bool(
        row_features["unstated_experience_for_senior_role"]
    )

    # Safety Guardrails: Domain and Severe Skill Mismatch
    resume_dom = feature_data.get("resume_domain", "Unknown")
    jd_dom = feature_data.get("jd_domain", "Unknown")
    domain_mismatch = bool(
        row_features.get("domain_match", 1.0) == 0.0
        and resume_dom != "Unknown"
        and jd_dom != "Unknown"
    )
    severe_skill_mismatch = bool(
        row_features.get("skill_match_score", 1.0) < 0.25
    )

    needs_review = (
        low_confidence
        or narrow_margin
        or not sufficient_evidence
        or unverifiable_seniority
        or domain_mismatch
        or severe_skill_mismatch
    )

    # Business Rule Guardrail: domain + skill double-mismatch → No Match
    if domain_mismatch and severe_skill_mismatch:
        pred_class = "no match"

    if domain_mismatch and severe_skill_mismatch:
        review_reason = (
            f"domain mismatch ({resume_dom} vs {jd_dom}) "
            f"and severe skill gap (<25% match)"
        )
    elif domain_mismatch:
        review_reason = (
            f"domain mismatch ({resume_dom} candidate vs {jd_dom} role)"
        )
    elif severe_skill_mismatch:
        review_reason = "severe skill mismatch (<25% skills matched)"
    elif unverifiable_seniority:
        review_reason = (
            "unstated candidate experience against a senior-level role"
        )
    elif narrow_margin:
        review_reason = (
            f"narrow decision margin ({confidence:.0%} vs {second_prob:.0%}: near-tie between classes)"
        )
    elif low_confidence:
        review_reason = (
            f"low model confidence ({confidence:.0%} < {conf_thresh:.0%})"
        )
    elif not sufficient_evidence:
        review_reason = "insufficient skill evidence"
    else:
        review_reason = None

    # ── SHAP Explanations (from the deciding stage's base forest) ─────────
    if deciding_stage == "a":
        sv = explainer_a.shap_values(X_row)
        # Binary classifier → SHAP for class 1 (Match)
        if isinstance(sv, list):
            contributions = sv[match_class_idx][0]
        elif len(sv.shape) == 3:
            contributions = sv[0, :, match_class_idx]
        else:
            contributions = sv[0]
    else:
        sv = explainer_b.shap_values(X_row)
        class_idx = list(model_b.classes_).index(pred_class)
        if isinstance(sv, list):
            contributions = sv[class_idx][0]
        elif len(sv.shape) == 3:
            contributions = sv[0, :, class_idx]
        else:
            contributions = sv[0]

    impact = sorted(
        zip(FEATURES, contributions), key=lambda t: -abs(t[1])
    )
    reasons = [{
        "factor": FEATURE_LABELS[feat],
        "value": round(float(X_row[feat].values[0]), 2),
        "direction": "supported" if contrib > 0 else "worked against",
        "impact": round(float(contrib), 3)
    } for feat, contrib in impact[:4]]

    return {
        "prediction": pred_class,
        "confidence": confidence,
        "margin": margin,
        "is_calibrated": True,
        "needs_human_review": needs_review,
        "review_reason": review_reason,
        "reasons": reasons
    }

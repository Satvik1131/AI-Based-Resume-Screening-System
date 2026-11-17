"""
Unit tests for Problem 4: Probability Calibration, ECE measurement,
decision margins, and evidence-based human review.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import shap

from training.calibration_analysis import expected_calibration_error, per_class_ece, compute_brier_scores
from app.services.ml_classifier import _extract_base_tree_estimator, evaluate_candidate_ml
from app.config import CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD, FEATURES


class TestCalibrationMath(unittest.TestCase):

    def test_perfectly_calibrated_ece(self):
        # 10 samples predicted with confidence 1.0, all 10 correct
        # 10 samples predicted with confidence 0.5, exactly 5 correct
        y_true = np.array([1]*10 + [1]*5 + [0]*5)
        # Class 1 prob
        proba_class_1 = np.array([1.0]*10 + [0.5]*10)
        proba_class_0 = 1.0 - proba_class_1
        proba = np.column_stack([proba_class_0, proba_class_1])
        classes = np.array([0, 1])

        ece = expected_calibration_error(y_true, proba, classes, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=5)

    def test_uncalibrated_overconfident_ece(self):
        # Model outputs 0.90 confidence for all 10 samples, but only 2 are correct (accuracy = 0.20)
        y_true = np.array([1, 1] + [0]*8)
        proba_1 = np.array([0.90]*10)
        proba_0 = 1.0 - proba_1
        proba = np.column_stack([proba_0, proba_1])
        classes = np.array([0, 1])

        ece = expected_calibration_error(y_true, proba, classes, n_bins=10)
        # Expected error: |0.20 - 0.90| = 0.70
        self.assertAlmostEqual(ece, 0.70, places=5)

    def test_per_class_ece_keys(self):
        y_true = np.array(["match", "partial match", "no match"] * 10)
        proba = np.random.dirichlet((1, 1, 1), size=len(y_true))
        classes = np.array(["match", "no match", "partial match"])

        pc_ece = per_class_ece(y_true, proba, classes, n_bins=5)
        self.assertIn("match", pc_ece)
        self.assertIn("partial match", pc_ece)
        self.assertIn("no match", pc_ece)
        for k, v in pc_ece.items():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


class TestCalibratedClassifierWrapper(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(60, 4)
        self.y = np.array(["match"]*20 + ["partial match"]*20 + ["no match"]*20)

    def test_calibrated_cv_probability_properties(self):
        base_rf = RandomForestClassifier(n_estimators=10, random_state=42)
        calibrated = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=3)
        calibrated.fit(self.X, self.y)

        proba = calibrated.predict_proba(self.X[:10])
        self.assertEqual(proba.shape, (10, 3))
        # Each probability in [0, 1]
        self.assertTrue((proba >= 0.0).all() and (proba <= 1.0).all())
        # Probabilities sum to 1.0
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(10), atol=1e-5)

    def test_shap_extracts_base_tree_estimator(self):
        base_rf = RandomForestClassifier(n_estimators=10, random_state=42)
        calibrated = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=3)
        calibrated.fit(self.X, self.y)

        extracted_rf = _extract_base_tree_estimator(calibrated)
        self.assertIsInstance(extracted_rf, RandomForestClassifier)

        # Ensure SHAP TreeExplainer runs without error
        explainer = shap.TreeExplainer(extracted_rf)
        sv = explainer.shap_values(self.X[:2])
        self.assertIsNotNone(sv)


class TestDecisionMarginAndHumanReview(unittest.TestCase):

    def test_narrow_margin_flags_near_tie(self):
        # Synthetic feature data where match and partial match are virtually tied
        feature_data = {
            "candidate_name": "Test Candidate",
            "resume_skills": ["python", "django", "git", "sql"],
            "matched_skills": ["python", "sql"],
            "missing_skills": ["docker", "kubernetes"],
            "resume_domain": "Software Development",
            "jd_domain": "Software Development",
            "features": {
                "skill_match_score": 0.50,
                "matched_skill_count": 2,
                "missing_skill_count": 2,
                "experience_match_score": 0.8,
                "experience_ratio": 0.8,
                "resume_experience_mentioned": 1,
                "jd_experience_mentioned": 1,
                "unstated_experience_for_senior_role": 0,
                "semantic_similarity_score": 0.65,
                "domain_match": 1,
                "skill_gap_signed": 0,
                "experience_gap_years": -1.0,
                "skill_balance_ratio": 0.5,
            }
        }

        res = evaluate_candidate_ml(feature_data)
        self.assertIn("confidence", res)
        self.assertIn("margin", res)
        self.assertIn("is_calibrated", res)
        self.assertTrue(res["is_calibrated"])

        # If margin is narrower than threshold, needs_human_review must be True
        if res["margin"] < MARGIN_THRESHOLD:
            self.assertTrue(res["needs_human_review"])
            self.assertIn("margin", res["review_reason"].lower())

    def test_strong_match_passes_review_gates(self):
        # Outstanding candidate matching high-confidence profile
        feature_data = {
            "candidate_name": "Strong Candidate",
            "resume_skills": ["python", "react", "aws", "docker", "postgresql", "fastapi", "git", "ci/cd"],
            "matched_skills": ["python", "react", "aws", "docker", "postgresql", "fastapi", "git"],
            "missing_skills": ["kubernetes"],
            "resume_domain": "Software Development",
            "jd_domain": "Software Development",
            "features": {
                "skill_match_score": 0.50,
                "matched_skill_count": 7,
                "missing_skill_count": 8,
                "experience_match_score": 1.0,
                "experience_ratio": 1.4,
                "resume_experience_mentioned": 1,
                "jd_experience_mentioned": 1,
                "unstated_experience_for_senior_role": 0,
                "semantic_similarity_score": 0.65,
                "domain_match": 1,
                "skill_gap_signed": -3,
                "experience_gap_years": 2.0,
                "skill_balance_ratio": 0.47,
            }
        }

        res = evaluate_candidate_ml(feature_data)
        self.assertEqual(res["prediction"], "match")
        self.assertGreaterEqual(res["confidence"], CONFIDENCE_THRESHOLD)
        self.assertGreaterEqual(res["margin"], MARGIN_THRESHOLD)
        self.assertFalse(res["needs_human_review"])
        self.assertIsNone(res["review_reason"])


if __name__ == "__main__":
    unittest.main()

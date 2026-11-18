import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Pipeline Parameters (Problem 4: Calibrated & Evidence-Backed)
CONFIDENCE_THRESHOLD = 0.65        # Evidence-backed: auto-decides ~69% of cases at ~82% accuracy
MARGIN_THRESHOLD = 0.10            # Evidence-backed: flags near-ties where top 2 probabilities differ by < 10%
CALIBRATION_METHOD = "sigmoid"     # Platt scaling calibrator (CalibratedClassifierCV)
CASCADE_THRESHOLD = 0.65          # Stage-A Match confidence; overridden by calibrated value in model_config
MIN_SKILL_EVIDENCE = 3
MIN_RESUME_SKILLS = 2
OVERQUALIFICATION_RATIO_THRESHOLD = 1.75
HIGH_EXPERIENCE_REQUIREMENT_THRESHOLD = 5
EXPERIENCE_UNKNOWN_SENTINEL = -1.0

# Ollama / LLM Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Feature columns expected by RandomForest model
FEATURES = [
    "skill_match_score",
    "matched_skill_count",
    "missing_skill_count",
    "experience_match_score",
    "experience_ratio",
    "resume_experience_mentioned",
    "jd_experience_mentioned",
    "unstated_experience_for_senior_role",
    "semantic_similarity_score",
    "domain_match",
    # Boundary-aware features (help distinguish Partial Match)
    "skill_gap_signed",
    "experience_gap_years",
    "skill_balance_ratio",
]

FEATURE_LABELS = {
    "skill_match_score": "Skill Match",
    "matched_skill_count": "Number of Matched Skills",
    "missing_skill_count": "Number of Missing Skills",
    "experience_match_score": "Experience Match",
    "experience_ratio": "Experience Ratio (uncapped)",
    "resume_experience_mentioned": "Resume States Experience",
    "jd_experience_mentioned": "JD States Required Experience",
    "unstated_experience_for_senior_role": "Unverified Experience vs. Senior Role",
    "semantic_similarity_score": "Overall Semantic Fit",
    "domain_match": "Domain Match",
    "skill_gap_signed": "Skill Breadth Gap (Resume vs JD)",
    "experience_gap_years": "Experience Gap (Years)",
    "skill_balance_ratio": "Skill Coverage Balance",
}

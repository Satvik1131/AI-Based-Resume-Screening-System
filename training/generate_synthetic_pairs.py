"""
Structured-Spec Synthetic Data Generator for Borderline Resumes (Problem 2: Fix 1).

Generates training samples from explicit, structured specifications so the
ground truth label ('partial match') is controlled BY CONSTRUCTION rather than guessed.

Validation Gate:
  After generation, runs the pipeline's exact skill extractor (extract_skills_exact).
  If the candidate's extracted skills diverge significantly from the spec, the sample
  is rejected.

Usage:
    # 1. Generate specifications and prompt batch:
    py training/generate_synthetic_pairs.py --mode sample-specs --n 50 --out data/synthetic_specs.json

    # 2. Run generation via local Ollama LLM (if Ollama is running):
    py training/generate_synthetic_pairs.py --mode generate-llm --n 10 --out data/synthetic_partial_pairs.csv

    # 3. Offline dry-run / template generator:
    py training/generate_synthetic_pairs.py --mode dry-run
"""
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from app.config import OLLAMA_MODEL, OLLAMA_HOST
from app.services.skill_matcher import extract_skills_exact

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

ROLE_TEMPLATES = {
    "Data Science": [
        ("Machine Learning Engineer", ["python", "pytorch", "docker", "sql", "scikit learn", "git"], 4),
        ("NLP Data Scientist", ["python", "nlp", "transformers", "pytorch", "huggingface", "sql"], 3),
        ("Generative AI Engineer", ["python", "llm", "rag", "langchain", "vector database", "docker"], 3),
        ("Data Analyst", ["python", "sql", "data analysis", "tableau", "excel"], 2),
    ],
    "Software Development": [
        ("Backend Python Developer", ["python", "fastapi", "postgresql", "docker", "redis", "git"], 3),
        ("Full Stack Developer", ["react", "typescript", "nodejs", "postgresql", "rest apis", "docker"], 4),
        ("Java Backend Engineer", ["java", "spring boot", "microservices", "mysql", "docker", "git"], 5),
    ],
    "DevOps": [
        ("DevOps / Cloud Engineer", ["aws", "docker", "kubernetes", "terraform", "linux", "jenkins"], 4),
        ("Site Reliability Engineer", ["linux", "kubernetes", "prometheus", "grafana", "python", "docker"], 4),
    ],
    "Cyber Security": [
        ("Security Analyst", ["network security", "siem", "wireshark", "vulnerability assessment", "incident response"], 3),
    ]
}

RESUME_STYLES = ["chronological", "functional", "modern-tech"]


# ── Step 1: Spec Sampler ──────────────────────────────────────────────────────

def sample_partial_spec(domain: str = None) -> Dict[str, Any]:
    """
    Constructs a borderline Partial Match spec with controlled skill overlap (50-80%)
    or experience shortfall (50-80% of requirement).
    """
    if not domain or domain not in ROLE_TEMPLATES:
        domain = random.choice(list(ROLE_TEMPLATES.keys()))

    role_name, req_skills, req_exp = random.choice(ROLE_TEMPLATES[domain])

    mode = random.choice(["skill_shortfall", "exp_shortfall", "dual_borderline"])

    if mode == "skill_shortfall":
        # Overlap: 50% - 75% of skills
        n_matched = max(2, int(len(req_skills) * random.uniform(0.50, 0.75)))
        cand_skills = random.sample(req_skills, n_matched)
        cand_exp = req_exp
    elif mode == "exp_shortfall":
        # Candidate has almost all skills (85-100%), but experience is 50-70% of requirement
        cand_skills = list(req_skills)
        cand_exp = max(1, int(req_exp * random.uniform(0.50, 0.70)))
    else:  # dual_borderline
        n_matched = max(2, int(len(req_skills) * random.uniform(0.60, 0.80)))
        cand_skills = random.sample(req_skills, n_matched)
        cand_exp = max(1, int(req_exp * 0.75))

    missing_skills = [s for s in req_skills if s not in cand_skills]
    style = random.choice(RESUME_STYLES)

    spec = {
        "domain": domain,
        "role": role_name,
        "required_skills": req_skills,
        "required_experience_years": req_exp,
        "candidate_skills": cand_skills,
        "candidate_missing_skills": missing_skills,
        "candidate_experience_years": cand_exp,
        "borderline_mechanism": mode,
        "style": style,
        "intended_label": "partial match"
    }
    return spec


def build_llm_prompt(spec: Dict[str, Any]) -> str:
    """Creates a targeted prompt for an LLM to generate text matching the spec exactly."""
    prompt = f"""You are a professional HR assistant generating a realistic training resume for machine learning evaluation.
Generate a realistic {spec['style']} resume adhering STRICTLY to the following profile:

- Target Role: {spec['role']}
- Total Experience: exactly {spec['candidate_experience_years']} years (clearly mention "{spec['candidate_experience_years']} years of professional experience" in summary/work history).
- Candidate MUST POSSESS these skills: {', '.join(spec['candidate_skills'])}.
- Candidate MUST NOT possess or mention any of these skills: {', '.join(spec['candidate_missing_skills'])}.

Format requirements:
- Write raw resume text with sections: SUMMARY, WORK EXPERIENCE, SKILLS, EDUCATION.
- Keep tone realistic, concise, and professional (avoid fluff or excessive adjectives).
- Do NOT include any introductory or concluding conversational text. Output ONLY the resume content.
"""
    return prompt


# ── Step 2: Validation Gate ───────────────────────────────────────────────────

def validate_synthetic_resume(resume_text: str, spec: Dict[str, Any]) -> bool:
    """
    Validation Rule:
      Run exact skill extractor on synthetic text.
      The extracted skills must include >= 80% of candidate_skills and
      must NOT contain the explicitly forbidden missing_skills.
    """
    extracted = extract_skills_exact(resume_text)
    
    # Must have matched most of the intended skills
    intended = set(spec["candidate_skills"])
    found_intended = intended.intersection(set(extracted))
    if len(found_intended) / len(intended) < 0.75:
        return False

    # Must NOT hallucinate forbidden missing skills
    forbidden = set(spec["candidate_missing_skills"])
    found_forbidden = forbidden.intersection(set(extracted))
    if len(found_forbidden) > 0:
        return False

    return True


# ── Step 3: Generation Harness ────────────────────────────────────────────────

def generate_with_ollama(spec: Dict[str, Any]) -> str:
    """Queries local Ollama instance if available."""
    import ollama
    prompt = build_llm_prompt(spec)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3}
    )
    return response["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Synthetic Borderline Data Generator.")
    parser.add_argument("--mode", choices=["sample-specs", "generate-llm", "dry-run"], default="dry-run")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="data/synthetic_specs.json", help="Output file path")
    args = parser.parse_args()

    print("=" * 72)
    print("  STRUCTURED-SPEC SYNTHETIC DATA GENERATOR (PROBLEM 2, FIX 1)".center(72))
    print("=" * 72)

    specs = [sample_partial_spec() for _ in range(args.n)]

    if args.mode == "dry-run":
        print(f"Generated {len(specs)} sample specifications:\n")
        for i, s in enumerate(specs, 1):
            print(f"[{i}] Role: {s['role']} ({s['domain']}) | Mechanism: {s['borderline_mechanism']}")
            print(f"    Required Skills: {s['required_skills']} ({s['required_experience_years']} yrs req)")
            print(f"    Candidate Has:   {s['candidate_skills']} ({s['candidate_experience_years']} yrs)")
            print(f"    Missing Skills:  {s['candidate_missing_skills']}")
            print(f"    Intended Label:  {s['intended_label'].upper()}")
            print("-" * 72)
        print("\nPrompt Template for Sample 1:")
        print(build_llm_prompt(specs[0]))

    elif args.mode == "sample-specs":
        out_file = BASE_DIR / args.out
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(specs, f, indent=2)
        print(f"Exported {len(specs)} structured specifications to:\n  {out_file}")

    elif args.mode == "generate-llm":
        print(f"Attempting to generate {len(specs)} synthetic resumes via Ollama ({OLLAMA_MODEL})...")
        valid_records = []
        for i, spec in enumerate(specs, 1):
            try:
                print(f"  Generating sample {i}/{len(specs)}: {spec['role']} ...")
                res_text = generate_with_ollama(spec)
                if validate_synthetic_resume(res_text, spec):
                    print(f"    -> PASSED verification gate.")
                    valid_records.append({
                        "resume_text": res_text,
                        "job_description": f"Role: {spec['role']}. Required skills: {', '.join(spec['required_skills'])}. Experience required: {spec['required_experience_years']} years.",
                        "match_label": "partial match",
                        "is_synthetic": True,
                        "spec": json.dumps(spec)
                    })
                else:
                    print(f"    -> REJECTED by verification gate (diverged from spec).")
            except Exception as e:
                print(f"    -> Error with Ollama: {e}")
                break

        if valid_records:
            out_csv = BASE_DIR / args.out
            pd.DataFrame(valid_records).to_csv(out_csv, index=False)
            print(f"\nSaved {len(valid_records)} verified synthetic pairs to:\n  {out_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()

"""
Pair Recombination Generator (Problem 2: Fix 2).

Leverages existing real resumes and cross-pairs them with different JDs
from resumeJD2_pairs.csv to naturally generate realistic borderline
(Partial Match) pairs without hallucinated text.

Filtering Criteria for Borderline Candidates:
  1. Skill match score between 0.40 and 0.80 (the classic partial match zone)
  2. Matched skill count >= 2 (not an empty no-match)
  3. Missing skill count >= 1 (not a perfect match)
  4. Real human-written text preserved

Usage:
    py training/recombine_pairs.py --max-pairs 150
"""
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
from app.config import (
    EXPERIENCE_UNKNOWN_SENTINEL
)
from app.services.skill_matcher import (
    extract_skills_exact, get_matched_skills, get_missing_skills,
    skill_match_score, identify_domain, compute_semantic_similarity
)
from app.services.feature_builder import extract_jd_experience


def recombine_dataset(csv_path: str, max_pairs: int = 150) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path).drop_duplicates().copy()
    
    # Isolate strong match resumes (real candidates with proven qualifications)
    matches = raw_df[raw_df["match_label"].str.lower().str.strip() == "match"].copy()
    all_jds = raw_df[["job_description"]].drop_duplicates().copy()
    
    print(f"Loaded {len(matches)} Match resumes and {len(all_jds)} distinct JDs.")
    print("Cross-pairing resumes against different JDs to identify borderline Partial Matches...")

    borderline_pairs = []
    seen_hashes = set()

    for r_idx, r_row in matches.iterrows():
        res_text = str(r_row["resume_text"])
        res_skills = extract_skills_exact(res_text)
        
        # Regex experience fallback
        res_exp_match = pd.Series([res_text]).str.extract(r'(\d+)\+?\s*years?')[0]
        res_exp = float(res_exp_match.iloc[0]) if not res_exp_match.isna().iloc[0] else None
        res_exp_mentioned = int(res_exp is not None)
        res_domain = identify_domain(res_skills)

        # Compare against other JDs
        for j_idx, j_row in all_jds.iterrows():
            if len(borderline_pairs) >= max_pairs:
                break

            jd_text = str(j_row["job_description"])
            
            # Avoid pairing against the original JD if identical
            if res_text[:100] in jd_text or jd_text[:100] in res_text:
                continue

            pair_hash = hash((res_text[:80], jd_text[:80]))
            if pair_hash in seen_hashes:
                continue
            seen_hashes.add(pair_hash)

            jd_skills = extract_skills_exact(jd_text)
            matched = get_matched_skills(res_skills, jd_skills)
            missing = get_missing_skills(res_skills, jd_skills)
            sk_score = skill_match_score(res_skills, jd_skills)

            # Targeted Partial Match Filter:
            # 1. Has non-trivial skill overlap (40% to 80%)
            # 2. At least 2 matched skills, and at least 1 missing skill
            if not (0.40 <= sk_score <= 0.80 and len(matched) >= 2 and len(missing) >= 1):
                continue

            jd_exp = extract_jd_experience(jd_text)
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

            jd_domain = identify_domain(jd_skills)
            domain_match = int(res_domain == jd_domain and res_domain != "Unknown")
            sim_score = compute_semantic_similarity(res_text, jd_text)

            # Filter out completely irrelevant pairs (different universes)
            if sim_score < 0.35 and domain_match == 0:
                continue

            # Boundary features
            skill_gap_signed = len(res_skills) - len(jd_skills)
            if res_exp_mentioned and jd_exp_mentioned:
                experience_gap_years = round(res_exp - jd_exp, 1)
            else:
                experience_gap_years = EXPERIENCE_UNKNOWN_SENTINEL

            total_jd = len(matched) + len(missing)
            skill_balance_ratio = round(min(len(matched), len(missing)) / total_jd, 4) if total_jd > 0 else 0.0

            borderline_pairs.append({
                "resume_text": res_text,
                "job_description": jd_text,
                "match_label": "partial match",
                "source": "pair_recombination",
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
            })

        if len(borderline_pairs) >= max_pairs:
            break

    result_df = pd.DataFrame(borderline_pairs)
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Recombine real resumes and JDs to form Partial Matches.")
    parser.add_argument("--max-pairs", type=int, default=100, help="Number of borderline pairs to mine")
    parser.add_argument("--out", type=str, default="data/recombined_partial_pairs.csv", help="Output path")
    args = parser.parse_args()

    csv_path = BASE_DIR / "resumeJD2_pairs.csv"
    out_path = BASE_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PAIR RECOMBINATION GENERATOR (PROBLEM 2, FIX 2)".center(70))
    print("=" * 70)

    recombined_df = recombine_dataset(str(csv_path), max_pairs=args.max_pairs)
    print(f"\nMined {len(recombined_df)} authentic borderline Partial Match pairs.")
    
    if len(recombined_df) > 0:
        print(f"  Avg Skill Match Score:      {recombined_df['skill_match_score'].mean():.3f}")
        print(f"  Avg Matched Skills:         {recombined_df['matched_skill_count'].mean():.1f}")
        print(f"  Avg Missing Skills:         {recombined_df['missing_skill_count'].mean():.1f}")
        print(f"  Avg Semantic Sim Score:     {recombined_df['semantic_similarity_score'].mean():.3f}")
        print(f"  Avg Skill Balance Ratio:    {recombined_df['skill_balance_ratio'].mean():.3f}")

        recombined_df.to_csv(out_path, index=False)
        print(f"\nSaved recombined pairs to:\n  {out_path}")
    else:
        print("No pairs met the filtering criteria.")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Experience Calculator Utility (Step 1).

Parses work experience and education section texts to compute deterministic,
gap-aware professional experience.

Features:
- Multi-range parsing on a single line (e.g., 'Jun–Aug 2025 | Feb–Jul 2026')
- Inherited year parsing for intra-year ranges (e.g., 'Jun–Aug 2025' -> Jun 2025 to Aug 2025)
- Three-tier gap handling:
    * gap <= 2 months: merged as normal transition (notice period, relocation)
    * gap > 2 months overlapping education: excluded, tagged as 'education'
    * gap > 2 months unexplained: flagged in res.gaps for recruiter review
- No double counting for concurrent/overlapping jobs
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple

_MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9, "sept": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_MONTH_PAT = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_YEAR_PAT = r"(?:19|20)\d{2}"
_PRESENT_PAT = r"(?:present|current|now|ongoing|till\s*(?:date|now)|to\s*date)"

# Regex matching date tokens with optional month and year
_TOKEN_PAT = rf"(?:({_MONTH_PAT})\.?\s*({_YEAR_PAT})?|({_YEAR_PAT})|({_PRESENT_PAT}))"

# Matches a single date range, e.g. "Jun–Aug 2025", "Jan 2020 - Dec 2021", "2021 - Present"
_RANGE_REGEX = re.compile(
    rf"(?:({_MONTH_PAT})\.?\s*({_YEAR_PAT})?|({_YEAR_PAT}))"
    rf"\s*(?:-|–|—|/|to)\s*"
    rf"(?:({_PRESENT_PAT})|({_MONTH_PAT})\.?\s*({_YEAR_PAT})|({_YEAR_PAT}))",
    re.IGNORECASE
)


@dataclass
class JobInterval:
    start: datetime
    end: datetime
    raw: str = ""


@dataclass
class ExperienceResult:
    total_years: float
    total_months: int
    jobs: List[Dict] = field(default_factory=list)
    gaps: List[Dict] = field(default_factory=list)


def _parse_month(m_str: Optional[str]) -> Optional[int]:
    if not m_str:
        return None
    return _MONTH_NAMES.get(m_str.lower().strip()[:3])


def _extract_date_ranges(text: str) -> List[Tuple[datetime, datetime, str]]:
    """Extracts all start and end date pairs from a section text."""
    intervals = []
    now = datetime.now()

    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue

        for match in _RANGE_REGEX.finditer(line_clean):
            # Groups from _RANGE_REGEX:
            # 1: start_month, 2: start_year, 3: start_year_only
            # 4: end_present, 5: end_month, 6: end_year, 7: end_year_only
            s_month_str = match.group(1)
            s_year_str  = match.group(2) or match.group(3)

            e_present   = match.group(4)
            e_month_str = match.group(5)
            e_year_str  = match.group(6) or match.group(7)

            if e_present:
                e_year = now.year
                e_month = now.month
            else:
                e_year = int(e_year_str) if e_year_str else now.year
                e_month = _parse_month(e_month_str) or 12

            # Year inheritance: if start has month but no year (e.g. "Jun–Aug 2025"), inherit end year
            if s_year_str:
                s_year = int(s_year_str)
            else:
                s_year = e_year

            s_month = _parse_month(s_month_str) or 1

            try:
                dt_start = datetime(s_year, s_month, 1)
                # End date inclusive: last day of month / next month day 1 for interval math
                dt_end = datetime(e_year, e_month, 28)
                if dt_end >= dt_start:
                    intervals.append((dt_start, dt_end, match.group(0)))
            except (ValueError, OverflowError):
                continue

    return intervals


def _months_inclusive(start: datetime, end: datetime) -> int:
    """Number of calendar months inclusive from start month to end month."""
    months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    return max(1, months)


def _gap_months(prior_end: datetime, next_start: datetime) -> int:
    """Number of gap months between consecutive non-overlapping jobs."""
    if next_start <= prior_end:
        return 0
    return max(0, (next_start.year - prior_end.year) * 12 + (next_start.month - prior_end.month) - 1)


def calculate_experience(
    experience_section_text: str,
    education_section_text: str = "",
    gap_merge_threshold_months: int = 2
) -> ExperienceResult:
    """
    Calculates total experience years and months from experience section text.
    Reconciles gaps against education periods and gap merge threshold.
    """
    if not experience_section_text or not experience_section_text.strip():
        return ExperienceResult(total_years=0.0, total_months=0, jobs=[], gaps=[])

    raw_intervals = _extract_date_ranges(experience_section_text)
    if not raw_intervals:
        return ExperienceResult(total_years=0.0, total_months=0, jobs=[], gaps=[])

    # Extract education intervals for gap reconciliation
    edu_intervals = _extract_date_ranges(education_section_text) if education_section_text else []

    # Sort intervals by start date
    sorted_intervals = sorted(raw_intervals, key=lambda t: t[0])

    # Step 1: Merge overlapping or adjacent intervals
    merged_jobs: List[Tuple[datetime, datetime]] = []
    for s_dt, e_dt, _ in sorted_intervals:
        if not merged_jobs:
            merged_jobs.append((s_dt, e_dt))
        else:
            prev_s, prev_e = merged_jobs[-1]
            if s_dt <= prev_e:
                merged_jobs[-1] = (prev_s, max(prev_e, e_dt))
            else:
                merged_jobs.append((s_dt, e_dt))

    # Step 2: Gap analysis between consecutive merged blocks
    gaps = []
    final_blocks = []
    curr_start, curr_end = merged_jobs[0]

    for next_start, next_end in merged_jobs[1:]:
        gap_len = _gap_months(curr_end, next_start)

        if gap_len <= gap_merge_threshold_months:
            # Short normal transition (notice period / relocation) -> Merge
            gaps.append({
                "start": curr_end.strftime("%b %Y"),
                "end": next_start.strftime("%b %Y"),
                "months": gap_len,
                "tag": "merged",
                "detail": f"Normal transition gap ({gap_len} mo <= {gap_merge_threshold_months} mo threshold)"
            })
            curr_end = max(curr_end, next_end)
        else:
            # Check if gap overlaps education
            edu_overlap = any(
                (edu_s <= next_start and edu_e >= curr_end) or
                (curr_end <= edu_s <= next_start) or
                (curr_end <= edu_e <= next_start)
                for edu_s, edu_e, _ in edu_intervals
            )

            if edu_overlap:
                gaps.append({
                    "start": curr_end.strftime("%b %Y"),
                    "end": next_start.strftime("%b %Y"),
                    "months": gap_len,
                    "tag": "education",
                    "detail": f"Higher education gap ({gap_len} months)"
                })
            else:
                gaps.append({
                    "start": curr_end.strftime("%b %Y"),
                    "end": next_start.strftime("%b %Y"),
                    "months": gap_len,
                    "tag": "unexplained",
                    "detail": f"Unexplained employment gap ({gap_len} months)"
                })

            final_blocks.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end

    final_blocks.append((curr_start, curr_end))

    # Step 3: Sum total months across blocks
    total_months = sum(_months_inclusive(b_s, b_e) for b_s, b_e in final_blocks)
    total_years = round(total_months / 12.0, 2)

    jobs_dicts = [
        {"start_date": b_s.strftime("%b %Y"), "end_date": b_e.strftime("%b %Y"), "months": _months_inclusive(b_s, b_e)}
        for b_s, b_e in final_blocks
    ]

    return ExperienceResult(
        total_years=total_years,
        total_months=total_months,
        jobs=jobs_dicts,
        gaps=gaps
    )

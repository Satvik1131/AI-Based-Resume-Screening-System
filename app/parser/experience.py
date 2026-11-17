"""
Python-level experience calculation from structured job records.

The LLM is NOT involved here. All date parsing and arithmetic is deterministic.

Algorithm (gap-aware):
  1. Parse (start_date, end_date) for every job using the flexible date parser.
  2. Build a list of (start_datetime, end_datetime) intervals.
  3. Sort intervals by start date.
  4. Merge overlapping intervals (handles concurrent jobs correctly).
  5. For each gap between consecutive merged intervals:
       - gap ≤ GAP_MERGE_THRESHOLD_MONTHS  → merge as continuous (normal transition)
       - gap overlaps an education entry    → exclude from experience, tag as "further education"
       - otherwise                          → flag as unexplained gap for human review
  6. Sum total months across final merged blocks (excluding education and unexplained gaps).
  7. Return total years rounded to 1 decimal place + gap analysis metadata.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.utils.dates import parse_date_flexible, months_between


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum gap (in months) between consecutive jobs that is treated as a
# normal transition (notice period, relocation, short job search).
# Gaps ≤ this threshold are merged into continuous experience.
GAP_MERGE_THRESHOLD_MONTHS: int = 2


# ---------------------------------------------------------------------------
# Gap analysis data structures
# ---------------------------------------------------------------------------

class GapInfo:
    """Describes a gap between two consecutive job blocks."""

    __slots__ = ("start", "end", "months", "tag", "detail")

    def __init__(
        self,
        start: datetime,
        end: datetime,
        months: int,
        tag: str,          # "merged" | "education" | "unexplained"
        detail: str = "",
    ):
        self.start = start
        self.end = end
        self.months = months
        self.tag = tag
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "start": self.start.strftime("%Y-%m"),
            "end":   self.end.strftime("%Y-%m"),
            "months": self.months,
            "tag":    self.tag,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_job_interval(job: dict) -> Optional[tuple[datetime, datetime]]:
    """
    Extract a (start, end) datetime pair from a job record.
    Returns None if the interval is invalid or unparseable.
    """
    start_raw = (job.get("start_date") or "").strip()
    end_raw   = (job.get("end_date") or "").strip()

    start_dt = parse_date_flexible(start_raw) if start_raw else None
    end_dt   = parse_date_flexible(end_raw)   if end_raw   else datetime.now()

    # Treat missing end date as current
    if end_dt is None:
        end_dt = datetime.now()

    if start_dt is None:
        return None  # Cannot place this job on the timeline

    if end_dt < start_dt:
        return None  # Invalid interval

    return (start_dt, end_dt)


def _merge_overlapping_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    """
    Classic interval merge: collapse overlapping or touching job spans
    so that concurrent roles don't double-count time.
    """
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda t: t[0])
    merged: list[tuple[datetime, datetime]] = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # Overlapping or touching — extend
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _parse_education_intervals(
    education_list: list[dict],
) -> list[tuple[datetime, datetime]]:
    """
    Extract date intervals from education records.
    Used to check whether a gap between jobs is explained by education.
    """
    intervals: list[tuple[datetime, datetime]] = []
    for edu in education_list:
        start_raw = (edu.get("start_date") or "").strip()
        end_raw   = (edu.get("end_date") or "").strip()

        start_dt = parse_date_flexible(start_raw) if start_raw else None
        end_dt   = parse_date_flexible(end_raw)   if end_raw   else None

        if start_dt and end_dt and end_dt >= start_dt:
            intervals.append((start_dt, end_dt))

    return intervals


def _gap_overlaps_education(
    gap_start: datetime,
    gap_end: datetime,
    edu_intervals: list[tuple[datetime, datetime]],
) -> Optional[tuple[datetime, datetime]]:
    """
    Check if a gap period overlaps with any education interval.
    Returns the overlapping education interval if found, else None.

    We consider a match if the education interval covers at least half
    of the gap, OR if the gap falls entirely within the education period.
    """
    for edu_start, edu_end in edu_intervals:
        # Calculate overlap
        overlap_start = max(gap_start, edu_start)
        overlap_end   = min(gap_end, edu_end)

        if overlap_start < overlap_end:
            # There is actual overlap
            gap_months = months_between(gap_start, gap_end) or 1
            overlap_months = months_between(overlap_start, overlap_end) or 1

            # Accept if overlap covers at least 50% of the gap
            if overlap_months >= gap_months * 0.5:
                return (edu_start, edu_end)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_total_experience(
    jobs: list[dict],
    education: Optional[list[dict]] = None,
    gap_threshold_months: Optional[int] = None,
) -> Optional[float]:
    """
    Calculate total years of professional experience from a list of validated
    job records, using gap-aware merging.

    Parameters
    ----------
    jobs : list[dict]
        Validated job records with start_date / end_date fields.
    education : list[dict] | None
        Education records (used to explain large gaps between jobs).
    gap_threshold_months : int | None
        Override the default gap merge threshold (for tuning).

    Returns
    -------
    float | None
        Total years of experience (rounded to 1 decimal), or None if
        no valid date intervals can be extracted.

    Gap handling:
      - gap ≤ threshold  → merged as continuous experience
      - gap overlaps education → excluded from experience count
      - otherwise → flagged as unexplained, excluded from experience count
    """
    threshold = gap_threshold_months if gap_threshold_months is not None else GAP_MERGE_THRESHOLD_MONTHS

    # --- Step 1: Parse job intervals ---
    intervals: list[tuple[datetime, datetime]] = []
    for job in jobs:
        interval = _parse_job_interval(job)
        if interval is not None:
            intervals.append(interval)

    if not intervals:
        return None

    # --- Step 2: Merge overlapping intervals (concurrent jobs) ---
    merged = _merge_overlapping_intervals(intervals)

    # --- Step 3: Parse education intervals for gap checking ---
    edu_intervals = _parse_education_intervals(education or [])

    # --- Step 4: Walk consecutive merged blocks and classify gaps ---
    # We build the final "experience blocks" by deciding which gaps to bridge.
    final_blocks: list[tuple[datetime, datetime]] = [merged[0]]

    for i in range(1, len(merged)):
        prev_end   = final_blocks[-1][1]
        next_start = merged[i][0]
        next_end   = merged[i][1]

        gap_months = months_between(prev_end, next_start)

        if gap_months <= threshold:
            # Small gap — normal transition, merge into continuous experience
            final_blocks[-1] = (final_blocks[-1][0], max(prev_end, next_end))

        elif _gap_overlaps_education(prev_end, next_start, edu_intervals):
            # Large gap explained by education — don't merge, keep separate blocks
            final_blocks.append((next_start, next_end))

        else:
            # Large unexplained gap — don't merge, keep separate blocks
            final_blocks.append((next_start, next_end))

    # --- Step 5: Sum months across final blocks ---
    total_months = sum(months_between(s, e) for s, e in final_blocks)

    if total_months <= 0:
        return None

    return round(total_months / 12.0, 1)


def calculate_experience_with_gaps(
    jobs: list[dict],
    education: Optional[list[dict]] = None,
    gap_threshold_months: Optional[int] = None,
) -> dict:
    """
    Full gap-aware experience calculation that returns both the total
    experience and a detailed gap analysis.

    Returns
    -------
    dict with keys:
      - total_years: float | None
      - gap_analysis: list[dict]  — one entry per gap between job blocks
      - merged_blocks: int        — number of continuous experience blocks
    """
    threshold = gap_threshold_months if gap_threshold_months is not None else GAP_MERGE_THRESHOLD_MONTHS

    # --- Step 1: Parse job intervals ---
    intervals: list[tuple[datetime, datetime]] = []
    for job in jobs:
        interval = _parse_job_interval(job)
        if interval is not None:
            intervals.append(interval)

    if not intervals:
        return {"total_years": None, "gap_analysis": [], "merged_blocks": 0}

    # --- Step 2: Merge overlapping intervals (concurrent jobs) ---
    merged = _merge_overlapping_intervals(intervals)

    # --- Step 3: Parse education intervals ---
    edu_intervals = _parse_education_intervals(education or [])

    # --- Step 4: Walk gaps and classify ---
    gap_analysis: list[GapInfo] = []
    final_blocks: list[tuple[datetime, datetime]] = [merged[0]]

    for i in range(1, len(merged)):
        prev_end   = final_blocks[-1][1]
        next_start = merged[i][0]
        next_end   = merged[i][1]

        gap_months = months_between(prev_end, next_start)

        if gap_months <= 0:
            # No actual gap (adjacent or overlapping after first merge)
            final_blocks[-1] = (final_blocks[-1][0], max(prev_end, next_end))
            continue

        if gap_months <= threshold:
            # Small gap — merge as continuous experience
            final_blocks[-1] = (final_blocks[-1][0], max(prev_end, next_end))
            gap_analysis.append(GapInfo(
                start=prev_end,
                end=next_start,
                months=gap_months,
                tag="merged",
                detail=f"Small gap ({gap_months}mo) — treated as normal job transition",
            ))

        else:
            edu_match = _gap_overlaps_education(prev_end, next_start, edu_intervals)
            if edu_match:
                # Education-explained gap — keep blocks separate
                final_blocks.append((next_start, next_end))
                gap_analysis.append(GapInfo(
                    start=prev_end,
                    end=next_start,
                    months=gap_months,
                    tag="education",
                    detail=(
                        f"Gap ({gap_months}mo) overlaps education period "
                        f"({edu_match[0].strftime('%Y-%m')} – {edu_match[1].strftime('%Y-%m')})"
                    ),
                ))

            else:
                # Unexplained large gap — keep blocks separate, flag for review
                final_blocks.append((next_start, next_end))
                gap_analysis.append(GapInfo(
                    start=prev_end,
                    end=next_start,
                    months=gap_months,
                    tag="unexplained",
                    detail=(
                        f"Unexplained gap ({gap_months}mo) — flagged for human review "
                        f"(possible sabbatical, layoff, caregiving, or career break)"
                    ),
                ))

    # --- Step 5: Sum months across final blocks ---
    total_months = sum(months_between(s, e) for s, e in final_blocks)
    total_years = round(total_months / 12.0, 1) if total_months > 0 else None

    return {
        "total_years":   total_years,
        "gap_analysis":  [g.to_dict() for g in gap_analysis],
        "merged_blocks": len(final_blocks),
    }


def format_years_and_months(years_float: Optional[float]) -> str:
    """
    Format a floating-point year value into a human-readable 'X years Y months' string.

    Examples:
      1.8  → '1 year 10 months'
      1.4  → '1 year 5 months'
      0.5  → '6 months'
      2.0  → '2 years'
      None → 'Not stated / unverified'
    """
    if years_float is None:
        return "Not stated / unverified"

    try:
        total_months = round(float(years_float) * 12.0)
        if total_months <= 0:
            return "0 months"

        y = total_months // 12
        m = total_months % 12

        parts = []
        if y > 0:
            parts.append(f"{y} {'year' if y == 1 else 'years'}")
        if m > 0:
            parts.append(f"{m} {'month' if m == 1 else 'months'}")

        return " ".join(parts) if parts else "0 months"
    except (ValueError, TypeError):
        return str(years_float)


def separate_internships(jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split job records into (full_time_or_unknown, internships).

    Internship detection uses the employment_type field set by the LLM extractor.
    Jobs with no employment_type are classified as full-time.
    """
    full_time:   list[dict] = []
    internships: list[dict] = []

    for job in jobs:
        emp_type = (job.get("employment_type") or "").lower()
        if "intern" in emp_type:
            internships.append(job)
        else:
            full_time.append(job)

    return full_time, internships

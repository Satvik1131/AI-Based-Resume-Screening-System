"""
Tests for experience date parsing and calculation.
These tests exercise the deterministic Python pipeline — no LLM calls.
"""
import sys
import unittest
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.utils.dates import parse_date_flexible, months_between, is_present
from app.parser.experience import (
    calculate_total_experience,
    calculate_experience_with_gaps,
    separate_internships,
    GAP_MERGE_THRESHOLD_MONTHS,
)


class TestIsPresent(unittest.TestCase):

    def test_present_variants(self):
        for token in ["Present", "present", "Current", "current", "Now", "now",
                      "Till Date", "till date", "Ongoing", "ongoing", "To Date"]:
            with self.subTest(token=token):
                self.assertTrue(is_present(token), f"Expected '{token}' to be present")

    def test_non_present_strings(self):
        for token in ["2022", "Jan 2022", "December 2021", "2020-06", ""]:
            with self.subTest(token=token):
                self.assertFalse(is_present(token), f"Expected '{token}' to NOT be present")


class TestParseDateFlexible(unittest.TestCase):

    def test_present_returns_now(self):
        result = parse_date_flexible("Present")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, datetime.now().year)

    def test_month_year_short(self):
        result = parse_date_flexible("Jan 2020")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2020)
        self.assertEqual(result.month, 1)

    def test_month_year_full(self):
        result = parse_date_flexible("March 2019")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2019)
        self.assertEqual(result.month, 3)

    def test_year_only(self):
        result = parse_date_flexible("2021")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2021)
        self.assertEqual(result.month, 1)

    def test_mm_yyyy(self):
        result = parse_date_flexible("06/2021")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2021)
        self.assertEqual(result.month, 6)

    def test_yyyy_mm(self):
        result = parse_date_flexible("2021-06")
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2021)
        self.assertEqual(result.month, 6)

    def test_empty_string(self):
        self.assertIsNone(parse_date_flexible(""))

    def test_garbage(self):
        self.assertIsNone(parse_date_flexible("not a date"))


class TestMonthsBetween(unittest.TestCase):

    def test_exact_one_year(self):
        d1 = datetime(2020, 1, 1)
        d2 = datetime(2021, 1, 1)
        self.assertEqual(months_between(d1, d2), 12)

    def test_six_months(self):
        d1 = datetime(2020, 1, 1)
        d2 = datetime(2020, 7, 1)
        self.assertEqual(months_between(d1, d2), 6)

    def test_reverse_order(self):
        # Should return positive regardless of order
        d1 = datetime(2021, 1, 1)
        d2 = datetime(2020, 1, 1)
        self.assertEqual(months_between(d1, d2), 12)


# =========================================================================
# Gap-Aware Experience Calculation Tests
# =========================================================================

class TestCalculateTotalExperience(unittest.TestCase):
    """Tests for calculate_total_experience (backward-compatible API)."""

    def test_single_job_two_years(self):
        jobs = [{"start_date": "Jan 2020", "end_date": "Jan 2022"}]
        result = calculate_total_experience(jobs)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 2.0, places=0)

    def test_two_non_overlapping_jobs_small_gap(self):
        """Jobs with a 2-month gap should be merged as continuous experience."""
        jobs = [
            {"start_date": "Jan 2018", "end_date": "Jan 2020"},
            {"start_date": "Mar 2020", "end_date": "Mar 2022"},
        ]
        result = calculate_total_experience(jobs)
        self.assertIsNotNone(result)
        # With 2-month gap merged, total should be ~4.2 years (50 months)
        # Jan 2018 → Mar 2022 = 50 months = 4.2 years
        self.assertGreater(result, 3.9)
        self.assertLess(result, 4.5)

    def test_overlapping_jobs_not_double_counted(self):
        # Two jobs that overlap completely — should count as one period
        jobs = [
            {"start_date": "Jan 2020", "end_date": "Dec 2022"},
            {"start_date": "Jun 2020", "end_date": "Jun 2021"},   # completely within above
        ]
        result = calculate_total_experience(jobs)
        self.assertIsNotNone(result)
        # Should be ~3 years, not ~4.5 years
        self.assertLess(result, 3.5)

    def test_present_end_date(self):
        current_year = datetime.now().year
        jobs = [{"start_date": f"Jan {current_year - 2}", "end_date": "Present"}]
        result = calculate_total_experience(jobs)
        self.assertIsNotNone(result)
        self.assertGreater(result, 1.5)

    def test_no_valid_jobs(self):
        jobs = [{"start_date": "", "end_date": ""}]
        result = calculate_total_experience(jobs)
        self.assertIsNone(result)

    def test_empty_list(self):
        result = calculate_total_experience([])
        self.assertIsNone(result)

    def test_inverted_dates_skipped(self):
        # end_date before start_date — should be skipped
        jobs = [{"start_date": "Jan 2022", "end_date": "Jan 2020"}]
        result = calculate_total_experience(jobs)
        self.assertIsNone(result)


class TestGapAwareExperience(unittest.TestCase):
    """Tests for the gap-aware merging logic — the core new feature."""

    def test_small_gap_merged(self):
        """A 1-month gap (normal notice period) should be merged."""
        jobs = [
            {"start_date": "Jan 2020", "end_date": "Dec 2021"},
            {"start_date": "Jan 2022", "end_date": "Dec 2023"},
        ]
        result = calculate_experience_with_gaps(jobs)
        self.assertIsNotNone(result["total_years"])
        # Jan 2020 → Dec 2023 = 48 months = 4.0 years
        # (The 1-month gap Dec→Jan is within threshold, merged)
        self.assertAlmostEqual(result["total_years"], 4.0, places=0)
        # Gap should be tagged as "merged"
        self.assertEqual(len(result["gap_analysis"]), 1)
        self.assertEqual(result["gap_analysis"][0]["tag"], "merged")

    def test_two_month_gap_merged(self):
        """A 2-month gap (relocation, job search) should be merged."""
        jobs = [
            {"start_date": "Jan 2020", "end_date": "Jun 2021"},
            {"start_date": "Aug 2021", "end_date": "Jun 2023"},
        ]
        result = calculate_experience_with_gaps(jobs)
        self.assertIsNotNone(result["total_years"])
        # Merged: Jan 2020 → Jun 2023 = 42 months = 3.5 years
        self.assertGreater(result["total_years"], 3.0)
        self.assertEqual(result["gap_analysis"][0]["tag"], "merged")

    def test_large_gap_not_merged(self):
        """A 6-month gap should NOT be merged — flagged as unexplained."""
        jobs = [
            {"start_date": "Jan 2018", "end_date": "Dec 2019"},
            {"start_date": "Jun 2020", "end_date": "Jun 2022"},
        ]
        result = calculate_experience_with_gaps(jobs)
        self.assertIsNotNone(result["total_years"])
        # Should be ~4.0 years (24 + 24 months), NOT 4.5 years
        self.assertAlmostEqual(result["total_years"], 4.0, places=0)
        self.assertEqual(len(result["gap_analysis"]), 1)
        self.assertEqual(result["gap_analysis"][0]["tag"], "unexplained")
        self.assertEqual(result["gap_analysis"][0]["months"], 6)

    def test_phd_gap_excluded(self):
        """
        The PhD bug scenario: someone works 3 years, does a 3-year PhD,
        then works 2 more years. Experience should be ~5 years, not ~8.
        """
        jobs = [
            {"start_date": "Jan 2015", "end_date": "Dec 2017"},  # 3 years
            {"start_date": "Jan 2021", "end_date": "Dec 2022"},  # 2 years
        ]
        education = [
            {
                "degree": "PhD in Computer Science",
                "institution": "MIT",
                "start_date": "Jan 2018",
                "end_date": "Dec 2020",
            },
        ]
        result = calculate_experience_with_gaps(jobs, education=education)
        self.assertIsNotNone(result["total_years"])
        # Should be ~5 years (36 + 24 months), NOT merged into 8 years
        self.assertAlmostEqual(result["total_years"], 5.0, places=0)
        self.assertEqual(len(result["gap_analysis"]), 1)
        self.assertEqual(result["gap_analysis"][0]["tag"], "education")

    def test_masters_gap_excluded(self):
        """
        Career break for a Master's degree — should be excluded from experience.
        """
        jobs = [
            {"start_date": "Jun 2016", "end_date": "May 2019"},   # 3 years
            {"start_date": "Sep 2021", "end_date": "Aug 2024"},   # 3 years
        ]
        education = [
            {
                "degree": "MBA",
                "institution": "IIM Ahmedabad",
                "start_date": "Jun 2019",
                "end_date": "Jun 2021",
            },
        ]
        result = calculate_experience_with_gaps(jobs, education=education)
        self.assertIsNotNone(result["total_years"])
        # Should be ~6 years total work experience, not ~8
        self.assertAlmostEqual(result["total_years"], 6.0, places=0)
        self.assertEqual(result["gap_analysis"][0]["tag"], "education")

    def test_normal_career_no_penalty(self):
        """
        A completely normal career with standard 1-month notice periods
        between jobs should show continuous experience, no fragmentation.
        """
        jobs = [
            {"start_date": "Jun 2015", "end_date": "Jul 2017"},
            {"start_date": "Aug 2017", "end_date": "Sep 2019"},
            {"start_date": "Oct 2019", "end_date": "Nov 2021"},
            {"start_date": "Dec 2021", "end_date": "Jan 2024"},
        ]
        result = calculate_experience_with_gaps(jobs)
        self.assertIsNotNone(result["total_years"])
        # Should be ~8.6 years (Jun 2015 → Jan 2024 = 103 months)
        self.assertGreater(result["total_years"], 8.0)
        # All gaps should be "merged"
        for gap in result["gap_analysis"]:
            self.assertEqual(gap["tag"], "merged")
        # Should be 1 merged block (all connected)
        self.assertEqual(result["merged_blocks"], 1)

    def test_sabbatical_flagged(self):
        """
        A 12-month gap with no education overlap should be flagged
        as unexplained for human review — not silently penalized.
        """
        jobs = [
            {"start_date": "Jan 2018", "end_date": "Dec 2019"},
            {"start_date": "Jan 2021", "end_date": "Dec 2023"},
        ]
        result = calculate_experience_with_gaps(jobs, education=[])
        self.assertIsNotNone(result["total_years"])
        # Should be ~5 years (24 + 36 months), gap not counted
        self.assertAlmostEqual(result["total_years"], 5.0, places=0)
        self.assertEqual(result["gap_analysis"][0]["tag"], "unexplained")
        # Dec 2019 → Jan 2021 = 13 months
        self.assertEqual(result["gap_analysis"][0]["months"], 13)

    def test_multiple_gaps_mixed(self):
        """
        Career with both small gaps (merged) and a large unexplained gap.
        """
        jobs = [
            {"start_date": "Jan 2015", "end_date": "Dec 2016"},   # 2 years
            {"start_date": "Feb 2017", "end_date": "Nov 2018"},   # ~2 years (1mo gap → merge)
            # 8-month gap here
            {"start_date": "Jul 2019", "end_date": "Jun 2021"},   # 2 years
        ]
        result = calculate_experience_with_gaps(jobs)

        # 2 gaps: first merged (2mo), second unexplained (8mo)
        self.assertEqual(len(result["gap_analysis"]), 2)
        self.assertEqual(result["gap_analysis"][0]["tag"], "merged")
        self.assertEqual(result["gap_analysis"][1]["tag"], "unexplained")

        # Total: blocks after merge = (Jan 2015 → Nov 2018) + (Jul 2019 → Jun 2021)
        # = 47 months + 24 months = 71 months ≈ 5.9 years
        self.assertGreater(result["total_years"], 5.5)
        self.assertLess(result["total_years"], 6.5)

    def test_custom_threshold(self):
        """Custom gap_threshold_months overrides the default."""
        jobs = [
            {"start_date": "Jan 2020", "end_date": "Dec 2020"},
            {"start_date": "Apr 2021", "end_date": "Dec 2021"},  # 4-month gap (Dec→Apr)
        ]
        # Default threshold (2mo) — should NOT merge
        result_default = calculate_experience_with_gaps(jobs, gap_threshold_months=2)
        self.assertEqual(result_default["gap_analysis"][0]["tag"], "unexplained")

        # Custom threshold (4mo) — SHOULD merge
        result_custom = calculate_experience_with_gaps(jobs, gap_threshold_months=4)
        self.assertEqual(result_custom["gap_analysis"][0]["tag"], "merged")
        # Total with merge: Jan 2020 → Dec 2021 = 24 months = 2.0 years
        self.assertAlmostEqual(result_custom["total_years"], 2.0, places=0)

    def test_education_without_dates_ignored(self):
        """
        Education records without dates should not crash or affect gap analysis.
        """
        jobs = [
            {"start_date": "Jan 2018", "end_date": "Dec 2019"},
            {"start_date": "Jan 2021", "end_date": "Dec 2022"},
        ]
        education = [
            {"degree": "B.Tech", "institution": "IIT", "start_date": "", "end_date": ""},
        ]
        result = calculate_experience_with_gaps(jobs, education=education)
        # Gap should be unexplained since education has no dates
        self.assertEqual(result["gap_analysis"][0]["tag"], "unexplained")

    def test_backward_compat_no_education(self):
        """
        calculate_total_experience without education arg should still work.
        """
        jobs = [
            {"start_date": "Jan 2020", "end_date": "Jan 2022"},
        ]
        result = calculate_total_experience(jobs)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 2.0, places=0)

    def test_same_company_promotion_no_gap(self):
        """
        Back-to-back roles at same company (promotion) — 0 gap, should merge.
        """
        jobs = [
            {"start_date": "Jan 2020", "end_date": "Jun 2021", "company": "Google",
             "designation": "Software Engineer"},
            {"start_date": "Jul 2021", "end_date": "Dec 2023", "company": "Google",
             "designation": "Senior Software Engineer"},
        ]
        result = calculate_experience_with_gaps(jobs)
        self.assertIsNotNone(result["total_years"])
        # Should be 4 years continuous
        self.assertAlmostEqual(result["total_years"], 4.0, places=0)
        self.assertEqual(result["merged_blocks"], 1)

    def test_zero_gap_analysis_for_single_job(self):
        """Single job should produce no gap analysis."""
        jobs = [{"start_date": "Jan 2020", "end_date": "Dec 2023"}]
        result = calculate_experience_with_gaps(jobs)
        self.assertEqual(result["gap_analysis"], [])
        self.assertEqual(result["merged_blocks"], 1)

    def test_gap_analysis_dict_format(self):
        """Verify gap analysis entries have the expected structure."""
        jobs = [
            {"start_date": "Jan 2020", "end_date": "Dec 2020"},
            {"start_date": "Jul 2021", "end_date": "Dec 2021"},
        ]
        result = calculate_experience_with_gaps(jobs)
        gap = result["gap_analysis"][0]
        self.assertIn("start", gap)
        self.assertIn("end", gap)
        self.assertIn("months", gap)
        self.assertIn("tag", gap)
        self.assertIn("detail", gap)
        # Verify date format is YYYY-MM
        self.assertRegex(gap["start"], r"^\d{4}-\d{2}$")
        self.assertRegex(gap["end"], r"^\d{4}-\d{2}$")


class TestSeparateInternships(unittest.TestCase):

    def test_separates_correctly(self):
        jobs = [
            {"company": "A", "employment_type": "Full-time"},
            {"company": "B", "employment_type": "Internship"},
            {"company": "C", "employment_type": ""},
        ]
        full_time, internships = separate_internships(jobs)
        self.assertEqual(len(full_time), 2)
        self.assertEqual(len(internships), 1)
        self.assertEqual(internships[0]["company"], "B")


if __name__ == "__main__":
    unittest.main()

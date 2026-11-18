"""
Tests for Experience Calculator (Step 1).
Includes ENTION test expecting 9 months across two intra-line date ranges.
"""
import unittest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.utils.experience_calculator import calculate_experience


class TestExperienceCalculator(unittest.TestCase):

    def test_ention_line_intra_line_ranges(self):
        """
        VENTION line: 'Full-Stack & Systems Developer Intern | ENTION® | Jun–Aug 2025 | Feb–Jul 2026'
        Jun–Aug 2025 (3 months) + Feb–Jul 2026 (6 months) = exactly 9 months (0.75 years).
        """
        line = "Full-Stack & Systems Developer Intern | ENTION® | Jun–Aug 2025 | Feb–Jul 2026"
        res = calculate_experience(line)
        self.assertEqual(res.total_months, 9)
        self.assertAlmostEqual(res.total_years, 0.75, places=2)
        self.assertEqual(len(res.jobs), 2)
        # Gap between Aug 2025 and Feb 2026 is 5 months
        self.assertEqual(len(res.gaps), 1)
        self.assertEqual(res.gaps[0]["tag"], "unexplained")

    def test_single_job_full_dates(self):
        text = "Software Engineer | Google | Jan 2020 - Dec 2021"
        res = calculate_experience(text)
        self.assertEqual(res.total_months, 24)
        self.assertAlmostEqual(res.total_years, 2.0, places=1)

    def test_short_transition_gap_merged(self):
        # 1-month transition gap between Nov 2021 and Jan 2022 -> merged as continuous
        text = "Job 1 | Jan 2020 - Nov 2021\nJob 2 | Jan 2022 - Dec 2022"
        res = calculate_experience(text)
        # Jan 2020 to Dec 2022 continuous = 36 months
        self.assertEqual(res.total_months, 36)
        self.assertAlmostEqual(res.total_years, 3.0, places=1)
        self.assertEqual(len(res.gaps), 1)
        self.assertEqual(res.gaps[0]["tag"], "merged")

    def test_education_gap_excluded(self):
        exp_text = "Job 1 | Jan 2018 - Dec 2019\nJob 2 | Jan 2022 - Dec 2023"
        edu_text = "M.Tech in CS | IIT Delhi | Jan 2020 - Dec 2021"
        res = calculate_experience(exp_text, edu_text)
        # 2 years Job 1 + 2 years Job 2 = 48 months (education gap excluded)
        self.assertEqual(res.total_months, 48)
        self.assertAlmostEqual(res.total_years, 4.0, places=1)
        self.assertEqual(len(res.gaps), 1)
        self.assertEqual(res.gaps[0]["tag"], "education")


if __name__ == "__main__":
    unittest.main()

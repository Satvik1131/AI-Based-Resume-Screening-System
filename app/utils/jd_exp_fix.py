"""
JD Experience Extraction Fix (Step 3).

Handles:
- Ranges: "3-5 years", "3 to 5 yrs" (returns minimum: 3.0)
- Plus notation: "5+ years", "5+ yrs" (returns 5.0)
- Word numbers: "two years", "three years" (returns 2.0, 3.0)
- Variations: "experience: 3 years", "requires 4+ yrs"
"""
import re
import numpy as np

_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20
}

_NUM_WORDS_OR_DIGITS = r"(?:\d+|" + "|".join(_WORD_TO_NUM.keys()) + r")"

# Pattern 1: Range "3-5 years", "3 to 5 years", "3 - 5 yrs" -> group(1) is min
_RANGE_PAT = re.compile(
    rf"({_NUM_WORDS_OR_DIGITS})\s*(?:-|–|—|to)\s*({_NUM_WORDS_OR_DIGITS})\s*(?:\+)?\s*(?:years?|yrs?)",
    re.IGNORECASE
)

# Pattern 2: Explicit "experience: 5+ years", "5+ years of experience"
_EXP_PLUS_PAT = re.compile(
    rf"(?:experience\s*:?\s*)?({_NUM_WORDS_OR_DIGITS})\s*\+?\s*(?:years?|yrs?)(?:\s*(?:of\s+)?experience)?",
    re.IGNORECASE
)

# Pattern 3: Word numbers or digits followed by year/years/yrs
_GENERAL_PAT = re.compile(
    rf"\b({_NUM_WORDS_OR_DIGITS})\s*\+?\s*(?:years?|yrs?)\b",
    re.IGNORECASE
)


def _to_int(val_str: str) -> int:
    val_clean = val_str.lower().strip()
    if val_clean in _WORD_TO_NUM:
        return _WORD_TO_NUM[val_clean]
    try:
        return int(val_clean)
    except ValueError:
        return 0


def jd_years(jd_text: str) -> float:
    """
    Extracts required years of experience from job description text.
    Returns float (e.g., 3.0) or np.nan if unstated.
    """
    if not jd_text:
        return np.nan

    text = str(jd_text).strip()

    # Check ranges first (e.g. "3-5 years" -> 3.0)
    m_range = _RANGE_PAT.search(text)
    if m_range:
        val = _to_int(m_range.group(1))
        if 0 <= val <= 25:
            return float(val)

    # Check explicit patterns
    for pat in (_EXP_PLUS_PAT, _GENERAL_PAT):
        m = pat.search(text)
        if m:
            val = _to_int(m.group(1))
            if 0 <= val <= 25:
                return float(val)

    return np.nan

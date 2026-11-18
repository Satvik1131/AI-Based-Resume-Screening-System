# Bug Analysis & Fix Plan — Pipeline Edge Cases

## Issue 1 (Root Cause) — Education Level Showing "Unknown"

### Why it happens on Ayush Sharma's resume

The education extraction has a **two-stage chain**:

1. **Rules-based** (`extract_education_rules`): scans for a line that matches `_DEGREE_RE` regex (line 56 in `rules.py`).
2. If rules confidence ≥ 0.70, uses rules. Otherwise, **falls back to Ollama LLM** (`extract_education_section`).

After either path, `best_education_record()` calls `normalize_degree()` which **only checks for these exact degree tokens**:
```
b.tech, b.e, b.sc, b.a, b.com, bba, bca,
m.tech, m.e, m.sc, mba, m.a, mca,
ph.d, doctorate, diploma, bachelor, master, associate
```

**The failure path for Ayush Sharma:**

The degree value that arrives at `normalize_degree()` is a **raw LLM/rule-extracted string**. For example, if Ayush's degree is extracted as:
- `"B.Tech. in Computer Science"` → `b.tech` has a **dot after "Tech"** which may fail the regex `\bb\.?\\s*tech\b`
- `"B.Tech CSE"` → **no dot** — this actually DOES match
- `"BE"` (common format in some resumes) → **NOT in the regex at all** — this is a **missing pattern**
- `"B.E."` with trailing dot → may or may not match `\bb\.?\\s*e\.?\b` depending on the word boundary
- `"Bachelors of Engineering"` → `bachelors` is NOT in the regex (only `bachelor` is)
- `"BTech"` → **no dot, no space** — fails the pattern `b\.?\\s*tech`
- `"B Tech"` (space, no dot) → needs `\\s*` which IS present, so this works

**Also**: if the education **section detection fails** (e.g., section heading is `"Scholastic Record"` or `"Academic Credentials"` — check: `"scholastic details"` is in aliases but NOT `"scholastic record"`), the section text will be empty, rules produce `confidence: 0.0`, LLM falls back and if Ollama is also offline → `education_list = []` → `best_education_record([])` → returns `"Unknown", "Unknown"`.

---

## All Issues Found Across the Pipeline

### 🔴 HIGH PRIORITY (causes wrong/missing output)

---

#### Issue 1 — Education Level = "Unknown"

**Files**: [`education.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/parser/education.py) | [`rules.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/parser/rules.py) | [`sections.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/parser/sections.py)

**Root causes**:
1. **`_DEGREE_RE` in `rules.py` (line 56)** misses common variants:
   - `BE`, `BTech`, `BEng`, `BArch`, `BSc` (no dots/spaces)
   - `Bachelors` (plural form — `bachelor` is matched but `bachelors` in the rules regex is not)
   - `Bachelors of Engineering`, `Bachelors of Technology`
   - `B.Tech.` (trailing dot causes word boundary failure)
   - `12th`, `10th`, `High School`, `HSC`, `SSC`, `Senior Secondary`
   - `LLB`, `LLM`, `MBBS`, `BDS`, `B.Pharmacy`, `Pharm.D`
   - `B.Sc (Hons)`, `M.Sc (Tech)`
2. **`normalize_degree()` in `education.py`** misses:
   - `B.E` with trailing dot → `\bb\.?\s*e\.?\b` — the `e` matches any word starting with `e` which is too broad
   - `BE` with no separator
   - `LLB`, `MBBS`, `BDS` (professional degrees)
   - `Post Graduate Diploma` (PGD is listed but the full form is not)
   - `Integrated M.Tech` (5-year programmes)
   - `Dual Degree` (B.Tech + M.Tech combined)
3. **Section aliases in `sections.py`** miss:
   - `"Scholastic Record"`, `"Scholastic Details"` → `"scholastic details"` is in aliases but `"scholastic record"` is NOT
   - `"Academic Record"` (very common heading)
   - `"Academics"` (popular short heading on student resumes)
   - `"Degrees"` (used in academic CVs)
   - `"Educational Details"`
   - `"Degrees & Certifications"`

**Fix Plan**:

**Step A** — Expand `_DEGREE_RE` in `rules.py` (line 56):
```diff
_DEGREE_RE = re.compile(
-   r"\b(b\.?\s*tech|b\.?\s*e\.?|b\.?\s*sc|b\.?\s*a\.?|b\.?\s*com|bba|bca|"
-   r"m\.?\s*tech|m\.?\s*e\.?|m\.?\s*sc|mba|m\.?\s*a\.?|mca|"
-   r"ph\.?\s*d\.?|doctorate|diploma|b\.?\s*arch|b\.?\s*pharm|"
-   r"bachelor|master|associate)\b",
+   r"\b(b\.?\s*tech\.?|b\.?\s*e\.?|be\b|btech\b|beng\b|"
+   r"b\.?\s*sc\.?|b\.?\s*a\.?|b\.?\s*com\.?|bba|bca|barch\b|b\.?\s*arch\.?|"
+   r"b\.?\s*pharm\.?|bpharm\b|llb\b|mbbs\b|bds\b|"
+   r"m\.?\s*tech\.?|mtech\b|m\.?\s*e\.?|m\.?\s*sc\.?|mba|m\.?\s*a\.?|mca|"
+   r"m\.?\s*eng\.?|m\.?\s*pharm\.?|llm\b|"
+   r"ph\.?\s*d\.?|doctorate|phd\b|"
+   r"diploma|polytechnic|integrated|dual\s+degree|"
+   r"bachelor'?s?|master'?s?|associate'?s?|"
+   r"high\s+school|senior\s+secondary|hsc\b|ssc\b|12th|10th)\b",
    re.IGNORECASE,
)
```

**Step B** — Expand `_PATTERNS` in `normalize_degree()` in `education.py`:
```diff
# Add to Doctorate patterns:
+ r"\bphd\b",  # without dot
# Add to Master patterns:
+ r"\bllm\b", r"\bm\.?\s*pharm\b",
+ r"\bintegrated\s+m\.?\s*tech\b",
# Add to Bachelor patterns:
+ r"\bllb\b", r"\bmbbs\b", r"\bbds\b",
+ r"\bbtec?h?\b",  # BE, BTech without dots
+ r"\bbeng\b",
+ r"\bbachelors?\b",  # explicit plural
+ r"\bb\.?\s*pharm\b",
# Add a new "High School" level below Diploma:
("High School", [
    r"\bhigh\s+school\b", r"\bsenior\s+secondary\b",
    r"\bhsc\b", r"\bssc\b", r"\b12th\b", r"\b10\+2\b",
])
```

**Step C** — Expand section aliases in `sections.py`:
```diff
"education": [
    "education",
+   "academic record",
+   "academics",
+   "scholastic record",   # missing — "scholastic details" is present but not "record"
+   "educational details",
+   "degrees",
+   "degrees & certifications",
    ...
]
```

---

#### Issue 2 — Company Names Extracted as Garbage / Job Descriptions as Designation

**Files**: [`rules.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/parser/rules.py) `extract_experience_rules()` | [`llm.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/parser/llm.py) `extract_experience_section()`

**Root cause**: The rules-based extractor's `_is_role_line()` matches any line containing a role keyword (e.g., "intern", "engineer"). For multi-line resume paragraphs, long bullet points describing duties can contain role keywords and accidentally get treated as a `designation`. Similarly, the LLM sometimes returns multi-sentence descriptions as company names (seen in `extraction_debug.json`: `"company": "conducting lectures and tutorials, mentoring students..."`).

**Fix Plan**:
- **`_is_role_line()`**: Add an explicit length cap — if a line has > 12 words, it should never be treated as a role title even if it contains a role keyword.
- **`extract_experience_rules()`**: When `company` is more than 8 words, it's almost certainly a description line — discard it from the company field.
- **LLM prompt** (`extract_experience_section` in `llm.py`): Add a stricter rule: `"company MUST be a proper organization name, NOT a sentence, NOT a description, NOT more than 6 words"`.
- **`validate_job()` in `validation.py`**: Add a check — if `company` is > 8 words, set to `"Company Not Identified"` instead of silently passing it through.

---

#### Issue 3 — Experience = "Not stated / unverified" Even When Dates Are Present

**Files**: [`experience.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/parser/experience.py) | [`dates.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/utils/dates.py)

**Root cause**: `calculate_total_experience()` calls `parse_date_flexible(start_raw)` — if the start date is in a format not covered by `dates.py` (e.g., `"Jan'21"`, `"Jan-21"`, `"01-2021"`, `"2021-01"`, `"Spring 2021"`, or Indian-style `"01/2021"`), `parse_date_flexible` returns `None` and the job is **skipped from experience calculation**.

**Fix Plan**: Expand `parse_date_flexible()` in `dates.py` to handle:
- `"Jan'21"` → Jan 2021 (apostrophe shorthand)
- `"Jan-21"` → Jan 2021 (hyphen instead of space)
- `"01-2021"` and `"2021-01"` → January 2021
- `"Spring 2021"`, `"Fall 2021"` → March 2021 / September 2021
- Fiscal year references: `"FY2021"`, `"FY 2021"`

---

### 🟡 MEDIUM PRIORITY (causes degraded output quality)

---

#### Issue 4 — Skills Missed From Non-Standard Section Headings

**Files**: [`sections.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/parser/sections.py)

**Root cause**: Section aliases for `"skills"` miss common headings:
- `"Technical Proficiencies"`, `"Competencies"`, `"Programming Languages"`, `"Frameworks & Libraries"`, `"Software & Tools"`, `"Languages & Tools"`, `"Tools & Frameworks"`, `"Relevant Skills"`, `"Soft Skills"`

**Fix Plan**: Add these to `SECTION_ALIASES["skills"]` in `sections.py`.

---

#### Issue 5 — Certifications Section Missed

**Files**: [`sections.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/parser/sections.py)

Missing aliases for certifications:
- `"Achievements & Certifications"`, `"Trainings & Certifications"`, `"Coursework"`, `"Relevant Coursework"`, `"MOOCs"`, `"Online Learning"`

---

#### Issue 6 — Experience Section Aliases Missing Academic Variants

**Files**: [`sections.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/parser/sections.py)

Missing:
- `"Projects & Experience"`, `"Positions of Responsibility"`, `"Volunteer Work"`, `"Extra-Curricular Activities"` (sometimes contains internships listed under this)

---

#### Issue 7 — Name Extraction Fails for Some Resume Formats

**Files**: [`llm.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/parser/llm.py) `_NAME_SKIP_WORDS`

**Root cause**: The `_NAME_SKIP_WORDS` set skips the word `"full"` — so **"Full Name"** in a header is correctly skipped. But it also inadvertently skips any name containing the substring "full" in a word match. More importantly, names like `"Ravi Kumar Singh"` (4 words) pass, but `"Dr. Ayush Sharma"` may fail because `Dr.` contains a dot which triggers the check `if "." in line_clean and not re.fullmatch(r"[A-Za-z\.\s]+", line_clean): continue` — this check is wrong because `"Dr. Ayush Sharma"` has dots and IS purely alphabetic+dot+space, so it should pass. However `"Dr."` by itself matches, but since it also has a dot, it needs to pass the `fullmatch` check, which it does.

The real issue: `"stack"` is in `_NAME_SKIP_WORDS` — if a candidate's name happens to contain the word "stack" as a surname (unlikely but possible), they would get "Unknown".

Also, names with hyphenation like `"Jean-Pierre Dupont"` contain a hyphen, and after `re.sub(r"[^a-zA-Z\s]", "", line_clean)` the hyphen is stripped, becoming `"JeanPierre Dupont"` with incorrect capitalization via `.title()`.

**Fix Plan**:
- Remove overly generic words from `_NAME_SKIP_WORDS` that could be parts of names
- Handle hyphenated names by splitting on `-` before stripping, then recombining

---

#### Issue 8 — Docling `iterate_items()` API Breaking Across Versions

**Files**: [`layout.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/parser/layout.py) lines 73

**Root cause**: `doc.iterate_items()` is a Docling API that changed between versions (returns `(element, level)` in some versions, just `element` in others). The current code assumes `(element, _level)` unpacking. If a newer/older Docling version changes this, the entire parser crashes and falls back to PyMuPDF with a traceback.

**Fix Plan**: Wrap `iterate_items()` in a try-except that handles both tuple and non-tuple return formats, with a version-safe iteration helper.

---

#### Issue 9 — JD Experience Extraction Picks Up Wrong Numbers

**Files**: [`feature_builder.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/app/services/feature_builder.py) `extract_jd_experience()`

**Root cause**: The pattern `r'(\d+)\+?\s*years'` will match phrases like `"minimum 10 years"`, `"up to 5 years"` — but also incorrectly matches `"15 years of industry history"` (irrelevant to required experience) or the number in `"team of 3 years experience"`. There's no check that the matched number is a plausible experience requirement (e.g., 0–20 years).

**Fix Plan**: Add context filtering — only extract numbers in the range 0–20, and prefer matches appearing near `"required"`, `"minimum"`, or `"at least"` keywords.

---

#### Issue 10 — Skill DB Missing Modern Tech Skills

**Files**: [`data/skills_db.json`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/data/skills_db.json)

Likely missing from the current skill DB (based on modern resume content):
- `"langchain"`, `"llamaindex"`, `"crewai"` — popular LLM frameworks
- `"hugging face"`, `"huggingface"` — model hosting
- `"streamlit"`, `"gradio"` — ML app deployment
- `"pyspark"`, `"databricks"` — data engineering
- `"dbt"` — already in the code's flat list but may not be in the JSON DB
- `"pinecone"`, `"chroma"`, `"weaviate"` — vector DBs
- `"supabase"`, `"firebase"` — backend-as-a-service
- `"next.js"`, `"nestjs"` — modern web frameworks
- `"tailwindcss"`, `"shadcn"` — CSS frameworks
- `"agile"`, `"scrum"`, `"jira"` — soft/process skills (seen as missing in the earlier run)
- `"leadership"`, `"communication"`, `"problem solving"` — soft skills

---

### 🟢 LOW PRIORITY (cosmetic/minor improvements)

---

#### Issue 11 — Scikit-learn Version Warning on Model Load

**Files**: [`models/rf_model.joblib`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/models/rf_model.joblib)

The model was pickled with scikit-learn 1.9.0 but the environment has 1.8.0. Fix: run `python training/export_model.py` to retrain.

---

#### Issue 12 — pyproject.toml Pins numpy ≥ 2.5.1 Which Requires Python 3.12

**Files**: [`pyproject.toml`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/pyproject.toml)

`numpy>=2.5.1` only supports Python 3.12+, but the project uses Python 3.11. This causes `uv run` to fail. Fix: remove or relax the numpy constraint in `pyproject.toml` since `requirements.txt` already handles it correctly.

---

#### Issue 13 — Upskilling Recommendations Truncated in Report

**Files**: [`run.py`](file:///c:/Users/Satvik/OneDrive/Desktop/NLP_Integrated_Shortlister/run.py) line 319

The resource string is truncated with `...` at terminal width. When Ollama is offline, the fallback resource text is a generic sentence — not a specific course link. This is acceptable but could be improved to include hardcoded fallback URLs per skill category.

---

## Summary: Prioritized Fix Order

| # | Issue | Severity | Files to Change |
|---|---|---|---|
| 1 | Education level Unknown | 🔴 HIGH | `rules.py`, `education.py`, `sections.py` |
| 2 | Company name is a description | 🔴 HIGH | `rules.py`, `llm.py`, `validation.py` |
| 3 | Experience not calculated (date parse fails) | 🔴 HIGH | `dates.py` |
| 4 | Skills section not detected | 🟡 MEDIUM | `sections.py` |
| 5 | Certifications section missed | 🟡 MEDIUM | `sections.py` |
| 6 | Experience section aliases incomplete | 🟡 MEDIUM | `sections.py` |
| 7 | Name extraction edge cases | 🟡 MEDIUM | `llm.py` |
| 8 | Docling API version fragility | 🟡 MEDIUM | `layout.py` |
| 9 | JD experience extraction picks wrong number | 🟡 MEDIUM | `feature_builder.py` |
| 10 | Skill DB missing modern skills | 🟡 MEDIUM | `data/skills_db.json` |
| 11 | sklearn version warning | 🟢 LOW | retrain model |
| 12 | pyproject.toml numpy conflict | 🟢 LOW | `pyproject.toml` |
| 13 | Upskilling recommendations truncated | 🟢 LOW | `run.py` |

---

## 🚀 Layout-Aware Document Model Architecture (Vision Heavy-Path Prototype)

### Overview
To handle complex, multi-column, table-heavy, and infographic/graphic resumes without slowing down the 70–80% of standard resumes that use single-column layouts, a **hybrid routing architecture** is implemented.

```
                  ┌──────────────────────────────┐
                  │    Input Resume Document     │
                  └──────────────┬───────────────┘
                                 │
                   [ app/parser/router.py ]
                   Lightweight Layout Analyzer
                                 │
               ┌─────────────────┴─────────────────┐
               ▼                                   ▼
    [ Fast Path (70–80%) ]             [ Heavy Vision Path (20%) ]
  • PyMuPDF / Regex / Docling         • Donut / LayoutLMv3
  • Minimal CPU & Memory latency     • Vision-Language spatial parsing
  • Standard 1-column layout          • Multi-column / graphic resumes
```

### Key Components

1. **Lightweight Layout Router (`app/parser/router.py`)**:
   - Analyzes bounding-box spatial coordinates, image density, vector drawings, and column clusters.
   - Classifies layout into `SIMPLE_SINGLE_COLUMN`, `MULTI_COLUMN`, `TABLE_HEAVY`, `GRAPHIC_COMPLEX`, or `SCANNED_IMAGE`.
   - Routes simple resumes to the fast path and complex ones to the heavy vision path.

2. **Vision-Language Parser Prototype (`app/parser/vision_models.py`)**:
   - Implements OCR-free visual document understanding using models like **Donut** (`naver-clova-ix/donut-base-vqa`) or **LayoutLMv3**.
   - Uses **lazy initialization** so optional dependencies (`transformers`, `torch`) do not inflate baseline footprint or break lightweight execution.
   - Converts PDF pages into renderable PIL image frames for visual region/text classification.

3. **Fallback & Resiliency**:
   - If the heavy vision model is unavailable or encounters missing GPU dependencies, the system gracefully falls back to the standard `PyMuPDF` / `Docling` / `pdfplumber` chain without raising runtime exceptions.


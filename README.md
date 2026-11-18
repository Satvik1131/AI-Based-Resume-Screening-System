<div align="center">

# 🤖 NLP-Integrated Resume Shortlister
### An AI-Driven Candidate Screening System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![ML](https://img.shields.io/badge/ML-Random%20Forest-green?logo=scikit-learn)](https://scikit-learn.org)
[![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama%203.2-orange)](https://ollama.com)
[![License](https://img.shields.io/badge/License-Educational-purple)]()

> **End-to-end AI pipeline** that parses resumes in any format, extracts structured information using a hybrid LLM + rule-based approach, and scores candidates against a Job Description using a calibrated two-stage Random Forest classifier — with full SHAP explainability. **100% offline. No cloud APIs.**

</div>

---

## 📌 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [How It Works — System Overview](#-how-it-works--system-overview)
3. [Architecture Deep Dive](#-architecture-deep-dive)
4. [13-Feature Engineering](#-13-feature-engineering)
5. [Two-Stage Cascade Classifier](#-two-stage-cascade-classifier)
6. [Project Structure](#-project-structure)
7. [Tech Stack](#-tech-stack)
8. [Setup & Installation](#-setup--installation)
9. [Running the Screener](#-running-the-screener)
10. [Sample Output](#-sample-output)
11. [Supported File Formats](#-supported-file-formats)
12. [Configuration & Environment Variables](#-configuration--environment-variables)
13. [Running Tests](#-running-tests)
14. [Key Design Decisions & Novelty](#-key-design-decisions--novelty)
15. [Expected Performance](#-expected-performance)
16. [Troubleshooting](#-troubleshooting)
17. [Team](#-team)

---

## 🎯 Problem Statement

Manual resume screening is one of the **most time-consuming bottlenecks** in the recruitment pipeline:

- Recruiters spend only **6–8 seconds** per resume on average
- Existing ATS systems rely on brittle **keyword matching** — they miss semantic equivalences (e.g., *"ML"* vs *"Machine Learning"*)
- Resume formats vary wildly — multi-column PDFs, scanned images, infographics — breaking most parsers
- Decisions are **opaque** — rejected candidates never know *why*

This project builds an **intelligent, explainable, and format-agnostic** screening system that understands resumes the way a human recruiter does — but at scale, in seconds, and with full transparency.

---

## 🔍 How It Works — System Overview

The pipeline runs in **4 stages:**

```
📄 Resume (PDF/DOCX/TXT/Image)
        │
        ▼
┌─────────────────────────────┐
│  STAGE 1: Document Parsing  │  ← Layout router → PyMuPDF / Docling / OCR
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  STAGE 2: NLP Extraction    │  ← Section detection → Rules → LLM fallback
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  STAGE 3: ML Classification │  ← 13-feature vector → Cascade RF → SHAP
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  STAGE 4: Report Generation │  ← Decision + Confidence + Insights
└─────────────────────────────┘
```

---

## 🏗️ Architecture Deep Dive

### Stage 1 — Layout-Aware Document Parsing

A **Layout Complexity Router** first analyzes each resume's spatial geometry (columns, graphics, images) and routes it to the most appropriate parser:

| Resume Type | Path | Parsers Used |
|---|---|---|
| Simple / single-column | ⚡ Fast Path | Docling → PyMuPDF → pdfplumber → Tesseract OCR |
| Multi-column / graphic-heavy / scanned | 🧠 Heavy Vision Path | Donut (OCR-free) / LayoutLMv3 |

> Every stage has a **fallback** — if Docling fails, PyMuPDF takes over; if that fails, pdfplumber; then OCR. **The system never hard-fails.**

---

### Stage 2 — Hybrid NLP Extraction

After parsing, the raw text is split into **semantic sections** (Experience, Education, Skills, Certifications, etc.) using header detection.

Each section is processed by:
1. **Deterministic rule-based extractors** (regex, pattern matching, dictionary scan) — **primary method**
2. **Locally-hosted LLM (Llama 3.2 via Ollama)** — invoked only when rules return low-confidence results

> ⚠️ **Experience duration is always calculated in Python** (not by the LLM) to avoid hallucinated date arithmetic.

---

### Stage 3 — Feature Engineering & ML Classification

A **13-dimensional feature vector** is built from:
- Skill match scores (fuzzy + semantic)
- Experience ratios
- Semantic similarity (SBERT embeddings)
- Domain alignment
- 3 novel boundary-aware features

This is passed to a **Two-Stage Cascade Random Forest** with calibrated probabilities.

---

### Stage 4 — Explainable Report

The final output includes:
- ✅ Match decision + confidence %
- 📊 Top 4 SHAP feature attributions (why the decision was made)
- 🧠 LLM-generated qualitative insights (strengths, gaps, career trajectory)
- 📚 Upskilling recommendations for missing skills
- 🚨 Human review flag for borderline cases

---

## 🧮 13-Feature Engineering

The model uses exactly **13 features** — 10 core + 3 boundary-aware:

| # | Feature | Description |
|---|---|---|
| 1 | `skill_match_score` | % of JD-required skills found in resume |
| 2 | `matched_skill_count` | Absolute count of matched skills |
| 3 | `missing_skill_count` | Absolute count of missing skills |
| 4 | `experience_match_score` | `min(resume_exp / jd_exp, 1.0)` — capped at 1 |
| 5 | `experience_ratio` | Raw `resume_exp / jd_exp` (uncapped, detects overqualification) |
| 6 | `resume_experience_mentioned` | Binary: does resume state years of experience? |
| 7 | `jd_experience_mentioned` | Binary: does JD state required years? |
| 8 | `unstated_experience_for_senior_role` | Flag: resume has no experience but JD requires 5+ yrs |
| 9 | `semantic_similarity_score` | SBERT cosine similarity between resume and JD text |
| 10 | `domain_match` | Binary: same domain (e.g., both "AI/ML")? |
| 11 | `skill_gap_signed` | `len(resume_skills) − len(jd_skills)` — signed surplus/deficit |
| 12 | `experience_gap_years` | `resume_exp − jd_exp` in raw years |
| 13 | `skill_balance_ratio` | `min(matched, missing) / total_jd_skills` — peaks at 0.5 for Partial Match |

> Features 11–13 are novel **boundary-aware** features that help the classifier distinguish *Partial Match* from clear Match/No-Match cases.

---

## 🌲 Two-Stage Cascade Classifier

```
                    ┌──────────────────┐
Input Features ───▶ │   Stage A        │
                    │  Match Detector  │
                    │  (Calibrated RF) │
                    └──────────────────┘
                           │
              ┌────────────┴──────────────┐
         P(Match) ≥ 0.65            P(Match) < 0.65
              │                          │
              ▼                          ▼
         "MATCH" ✅            ┌─────────────────────┐
                               │      Stage B         │
                               │  Partial vs No-Match │
                               │  (Calibrated RF)     │
                               └─────────────────────┘
                                          │
                               ┌──────────┴──────────┐
                          "PARTIAL MATCH"       "NO MATCH" ❌
```

**Safety Guardrails** (override the classifier):
- 🔴 **Domain mismatch + severe skill gap (<25%)** → forced "No Match"
- 🟡 **Low confidence (<65%)** → flagged for Human Review
- 🟡 **Narrow margin (<10% between top-2 classes)** → flagged for Human Review
- 🟡 **Unverifiable seniority** → flagged for Human Review

**Calibration:** Both stages use **Platt scaling** (`CalibratedClassifierCV`) for honest, interpretable probability scores (not raw overconfident probabilities).

---

## 📁 Project Structure

```
NLP_Integrated_Shortlister/
│
├── run.py                          ← 🚀 Main entry point
├── requirements.txt                ← All Python dependencies
├── pyproject.toml                  ← Project metadata
├── Project_Synopsis.md             ← Full academic project synopsis
├── implementation_plan.md          ← Development roadmap
│
├── app/                            ← Core application code
│   ├── config.py                   ← Thresholds, paths, feature names, Ollama config
│   ├── pipeline.py                 ← Orchestrates the full 4-stage pipeline
│   │
│   ├── parser/                     ← Resume parsing & NLP extraction
│   │   ├── layout.py               ← Layout router + fallback chain (Docling→PyMuPDF→OCR)
│   │   ├── router.py               ← Complexity classifier (simple vs complex layout)
│   │   ├── sections.py             ← Semantic section detector (Experience/Skills/Edu...)
│   │   ├── llm.py                  ← Ollama LLM per-section extractors
│   │   ├── rules.py                ← Rule-based extractors (regex, pattern matching)
│   │   ├── experience.py           ← Date math & experience year calculation
│   │   ├── education.py            ← Degree normalization & level classification
│   │   ├── skills.py               ← Skill merge, dedup, normalization
│   │   ├── vision_models.py        ← Donut / LayoutLMv3 for complex layouts
│   │   └── orchestrator.py         ← Coordinates parser → sections → extraction
│   │
│   ├── services/                   ← ML, feature engineering & reporting
│   │   ├── feature_builder.py      ← Builds the 13-feature vector
│   │   ├── ml_classifier.py        ← Two-stage cascade RF + SHAP explanations
│   │   ├── skill_matcher.py        ← Fuzzy + SBERT semantic skill matching
│   │   ├── report_generator.py     ← Assembles final report dict
│   │   ├── text_extraction.py      ← Safe file-to-text extraction
│   │   └── llm_extractor.py        ← Full-resume LLM extraction (legacy path)
│   │
│   └── utils/                      ← Shared utilities
│       ├── dates.py                ← Date parsing helpers
│       ├── schema.py               ← Canonical ResumeSchema dataclass
│       ├── validation.py           ← Pre-ML data validation
│       ├── experience_calculator.py← Experience gap analysis
│       └── jd_exp_fix.py           ← JD experience extraction edge cases
│
├── models/                         ← Pre-trained ML models (included)
│   ├── stage_a_model.joblib        ← Stage A: Match Detector (Calibrated RF)
│   ├── stage_b_model.joblib        ← Stage B: Partial vs No-Match (Calibrated RF)
│   ├── model_config.joblib         ← Thresholds used at training time
│   └── rf_model.joblib             ← Legacy single-stage model (fallback)
│
├── training/                       ← Model training scripts
│   ├── generate_synthetic_pairs.py ← Synthetic resume-JD pair generator
│   ├── recombine_pairs.py          ← Dataset recombination utilities
│   ├── calibration_analysis.py     ← Platt scaling analysis
│   ├── learning_curve.py           ← Model learning curve evaluation
│   ├── evaluate_model.py           ← Evaluation metrics & confusion matrix
│   └── export_model.py             ← Exports trained models to /models/
│
├── data/                           ← Training dataset
│   └── resumeJD_pairs.csv          ← Labeled resume-JD pairs
│
├── ipdocs/                         ← 📂 Sample files (ready to test!)
│   ├── Resume.pdf                  ← Sample candidate resume
│   ├── Celebal_JD.pdf              ← Sample job description
│   └── ...                         ← More sample resumes & JDs
│
├── tests/                          ← 72 unit tests
│   ├── test_pipeline.py
│   ├── test_calibration.py
│   ├── test_experience_calculator.py
│   └── test_parser/
│       ├── test_experience.py
│       ├── test_sections.py
│       ├── test_schema.py
│       ├── test_router.py
│       └── test_validation.py
│
└── reports/                        ← Generated assets
    └── calibration_curve.png       ← Model calibration visualization
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Document Parsing** | [Docling](https://github.com/DS4SD/docling) | Primary layout-aware PDF/DOCX parser |
| | [PyMuPDF](https://pymupdf.readthedocs.io) | Fallback 1 — two-column aware PDF |
| | [pdfplumber](https://github.com/jsvine/pdfplumber) | Fallback 2 — plain PDF text |
| | [python-docx](https://python-docx.readthedocs.io) | DOCX parsing |
| **OCR** | [pytesseract](https://github.com/madmaze/pytesseract) + [pdf2image](https://github.com/Belval/pdf2image) | Scanned / image resumes |
| **Vision Models** | Donut, LayoutLMv3 | Complex / infographic layouts |
| **LLM** | [Ollama](https://ollama.com) + Llama 3.2 | Local LLM extraction (fully offline) |
| **NLP / Embeddings** | [sentence-transformers](https://www.sbert.net) (SBERT) | Semantic similarity scoring |
| **Skill Matching** | [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) | Fuzzy string skill matching |
| **ML Model** | [scikit-learn](https://scikit-learn.org) Random Forest + CalibratedClassifierCV | Two-stage cascade classifier |
| **Explainability** | [SHAP](https://shap.readthedocs.io) TreeExplainer | Feature attribution for decisions |
| **Data Processing** | pandas, numpy | Feature engineering & data handling |
| **Language** | Python 3.11+ | — |

---

## ⚙️ Setup & Installation

### Prerequisites

- **Python 3.11+** ([download](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Git** ([download](https://git-scm.com/downloads))
- *(Optional but recommended)* **Ollama** — for LLM-powered extraction

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NLP_Integrated_Shortlister.git
cd NLP_Integrated_Shortlister
```

---

### Step 2 — (Recommended) Create a virtual environment

```bash
# Create venv
python -m venv .venv

# Activate — Windows PowerShell:
.venv\Scripts\activate

# Activate — Mac / Linux:
source .venv/bin/activate
```

You'll see `(.venv)` in your terminal when it's active.

---

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> ⏳ **First install takes 3–8 minutes** — `torch`, `sentence-transformers`, and `docling` are large packages.

> 💡 `docling` is the most accurate parser. If it fails to install, the pipeline **automatically falls back to PyMuPDF** — no action needed.

---

### Step 4 — (Optional) Install Ollama for LLM extraction

Ollama gives significantly richer extraction using a local LLM. **If skipped, rule-based extraction is used automatically.**

**Windows:**
1. Download from **https://ollama.com/download** → Windows
2. Install and run it (starts a background service on port `11434`)
3. Pull the model:
   ```bash
   ollama pull llama3.2
   ```
4. Verify:
   ```bash
   ollama list
   # → Should show: llama3.2
   ```

**Mac/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

---

## 🚀 Running the Screener

```bash
python run.py
```

You'll be prompted:

```
Enter path to resume file (press Enter for default sample):
Enter path to job description file (press Enter for default sample):
```

| Input | Action |
|---|---|
| Press **Enter** (blank) | Uses sample files from `ipdocs/` folder |
| Type a file path | Uses your own resume / JD file |

**Supported file formats:** `.pdf`, `.docx`, `.txt`, `.jpg`, `.png`

**Example with custom files:**
```
Enter path to resume file: C:\Users\You\Documents\my_resume.pdf
Enter path to job description file: C:\Users\You\Documents\job_desc.pdf
```

**Debug mode** (dumps full JSON report):
```bash
python run.py --debug
```

---

## 📊 Sample Output

```
══════════════════════════════════════════════════════════════════════════════
                        CANDIDATE EVALUATION REPORT
══════════════════════════════════════════════════════════════════════════════
  Candidate Name       : NAMAN GOYAL
  Model Confidence     : 62.0%  (Decision Threshold: 55%)
  Final Decision       : UNDER HUMAN REVIEW
  Human Review Needed  : YES — low model confidence, please verify manually
──────────────────────────────────────────────────────────────────────────────
  Extraction Engine    : Modular Pipeline (rule_based)
  Experience Source    : Layout-aware sections via pymupdf
──────────────────────────────────────────────────────────────────────────────
  SKILLS ANALYSIS
  Overall Skill Match  : 13/16 required skills matched (81.2%)
  Matched Skills       : Artificial Intelligence, Express, Javascript, Llm,
                         Mongodb, Nodejs, Pinecone, Problem Solving, Rag,
                         React, Rest Api, Sql, Leadership
  Missing Skills       : Agile, Ai Agents, Databricks
  Bonus Skills         : Docker, FastAPI  (not required, but a plus)
──────────────────────────────────────────────────────────────────────────────
  EXPERIENCE & EDUCATION
  Candidate Experience : 0.8 yrs (9 months 18 days)
  JD Requires          : 2.0 yrs
  Education Level      : Bachelor
  Education Details    : B.Tech Computer Science
──────────────────────────────────────────────────────────────────────────────
  DOMAIN & SEMANTIC FIT
  Candidate Domain     : AI/ML
  JD Domain            : AI/ML
  Domain Match         : YES
  Semantic Similarity  : 61.5%
──────────────────────────────────────────────────────────────────────────────
  TOP FACTORS DRIVING THIS DECISION
  - Experience Match decreased match confidence (value: -1.2, impact: -0.142)
  - Skill Match increased match confidence (value: 0.81, impact: +0.118)
  - Semantic Similarity increased match confidence (value: 0.615, impact: +0.094)
  - Experience Gap decreased match confidence (value: -1.2, impact: -0.071)
──────────────────────────────────────────────────────────────────────────────
  SUGGESTED UPSKILLING (for missing JD skills)
  - Agile               -> Agile & Scrum Fundamentals (Coursera)
  - Ai Agents           -> Building AI Agents with LangChain (DeepLearning.AI)
  - Databricks          -> Databricks Lakehouse Fundamentals (Free — databricks.com)
══════════════════════════════════════════════════════════════════════════════
```

---

## 📂 Supported File Formats

| Format | Support | Parser Chain |
|--------|---------|-------------|
| `.pdf` | ✅ Full | Docling → PyMuPDF → pdfplumber → Tesseract OCR |
| `.docx` | ✅ Full | Docling → python-docx |
| `.txt` | ✅ Plain text | Direct read |
| `.jpg` / `.png` | ✅ OCR | pytesseract + pdf2image |

---

## 🔧 Configuration & Environment Variables

All tuneable parameters live in [`app/config.py`](app/config.py):

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.65` | Min confidence to auto-decide without human review |
| `MARGIN_THRESHOLD` | `0.10` | If top-2 class probabilities differ by < 10%, flag for review |
| `CASCADE_THRESHOLD` | `0.65` | Stage A: min confidence to call "Match" |
| `OVERQUALIFICATION_RATIO_THRESHOLD` | `1.75` | experience_ratio above this = overqualified |

**Environment variables** (set before running):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Ollama model to use for LLM extraction |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

**Windows PowerShell:**
```powershell
$env:OLLAMA_MODEL = "llama3.2"
python run.py
```

**Mac / Linux:**
```bash
export OLLAMA_MODEL=llama3.2
python run.py
```

---

## 🧪 Running Tests

All **72 tests** are fully deterministic — no Ollama, no internet, no LLM needed:

```bash
# Run all parser tests
python -m pytest tests/test_parser/ -v

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=term-missing
```

---

## 💡 Key Design Decisions & Novelty

### 1. Privacy-First, Fully Offline
The LLM (Llama 3.2) runs **entirely on your local machine** via Ollama. No resume data is sent to OpenAI, Anthropic, or any external API. Your data never leaves your computer.

### 2. Graceful Degradation at Every Layer
```
Docling fails? → PyMuPDF
PyMuPDF fails? → pdfplumber
pdfplumber fails? → Tesseract OCR
Ollama offline? → Rule-based extraction
Any parser fails? → Never crashes — always returns a report
```

### 3. Calibrated Probabilities (Not Raw Scores)
Most classifiers output overconfident raw probabilities. We use **Platt scaling** (`CalibratedClassifierCV`) so that a 70% confidence score actually means the model is right ~70% of the time.

### 4. Human-in-the-Loop for Borderline Cases
Instead of making overconfident wrong decisions, the system **flags near-ties and low-confidence predictions** for human review. This is explicitly better than a black-box auto-reject.

### 5. 3 Novel Boundary-Aware Features
Standard skill match scores don't distinguish between *"3 matched out of 4 required"* (great!) vs *"3 matched out of 20 required"* (terrible!). Our three additional features capture these absolute magnitudes, significantly improving partial-match discrimination.

### 6. Dual-Path Layout Routing
Simple resumes take a fast text-extraction path. Complex, graphic-heavy, or infographic-style resumes are handled by **Donut** (an OCR-free visual document understanding model) for accurate parsing even without text layers.

---

## 📈 Expected Performance

| Metric | Value |
|---|---|
| Accuracy (at 65% confidence threshold) | ~82% |
| Cases auto-decided (no human review) | ~69% |
| Recruiter time reduction (estimated) | 70–80% |
| Test suite | 72 tests, 100% deterministic |
| Supported formats | PDF, DOCX, TXT, JPG, PNG |

---

## ❓ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'shap'` | Run `pip install -r requirements.txt` again |
| `Ollama server not reachable` | Install Ollama from https://ollama.com/download; pipeline still works without it |
| `InconsistentVersionWarning` for joblib | Harmless — minor scikit-learn patch version mismatch between training & runtime |
| Resume parsed with no text | File may be scanned/image-only; ensure pytesseract + Tesseract are installed |
| OCR returns empty text | Install Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki |
| `git index.lock: Permission denied` | Run `Remove-Item .git\index.lock -Force` then retry git commands |
| Slow first run | `sentence-transformers` downloads SBERT model weights on first use (~90 MB) |

---

## 👨‍💻 Team

| Name | Roll Number | GitHub |
|---|---|---|
| Vivaan Mishra | 2024Btech226 | [@mishravivaan178-tech](https://github.com/mishravivaan178-tech) |
| Satvik Agarwal | 2024Btech190 | [@Satvik1131](https://github.com/Satvik1131) |
| Kartik Sharma | 2024Btech092 | [@KK-code001](https://github.com/KK-code001) |
| Mohit Khurana | 2024Btech155 | — |

**Institution:** JK Lakshmipat University, Jaipur
**Academic Year:** 2024–25
**Domain:** Natural Language Processing · Machine Learning · Intelligent Automation

---

## 🙌 Acknowledgements

Built with ❤️ using:
- [Ollama](https://ollama.com) — local LLM inference
- [Docling](https://github.com/DS4SD/docling) — layout-aware document parsing
- [scikit-learn](https://scikit-learn.org) — Random Forest + calibration
- [sentence-transformers](https://www.sbert.net) — semantic similarity
- [SHAP](https://shap.readthedocs.io) — model explainability
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) — fuzzy skill matching

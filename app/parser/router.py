"""
Lightweight Layout Router & Heuristic Complexity Analyzer.

Analyzes document spatial layout (columns, tables, graphic elements)
to determine whether a resume should use the fast path (PyMuPDF / regex / Docling)
or be routed to the vision heavy path (LayoutLMv3 / Donut).
"""
from __future__ import annotations

import os
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class LayoutComplexity(Enum):
    SIMPLE_SINGLE_COLUMN = "simple_single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    GRAPHIC_COMPLEX = "graphic_complex"
    SCANNED_IMAGE = "scanned_image"


@dataclass
class LayoutAnalysisResult:
    complexity: LayoutComplexity
    recommended_path: str   # "fast" | "heavy"
    column_count: int = 1
    image_count: int = 0
    table_detected: bool = False
    confidence: float = 1.0
    reasons: list[str] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


def analyze_document_layout(file_path: str) -> LayoutAnalysisResult:
    """
    Analyzes document spatial geometry, bounding box clusters, image density,
    and table presence to classify layout complexity.
    """
    ext = os.path.splitext(file_path)[1].lower()
    reasons: list[str] = []

    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        reasons.append("Image file input")
        return LayoutAnalysisResult(
            complexity=LayoutComplexity.GRAPHIC_COMPLEX,
            recommended_path="heavy",
            image_count=1,
            reasons=reasons,
        )

    if ext != ".pdf":
        reasons.append("Non-PDF document format (DOCX/TXT)")
        return LayoutAnalysisResult(
            complexity=LayoutComplexity.SIMPLE_SINGLE_COLUMN,
            recommended_path="fast",
            reasons=reasons,
        )

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        if len(doc) == 0:
            return LayoutAnalysisResult(
                complexity=LayoutComplexity.SIMPLE_SINGLE_COLUMN,
                recommended_path="fast",
            )

        page = doc[0]
        rect = page.rect
        page_width = rect.width
        raw_blocks = page.get_text("dict")["blocks"]

        text_blocks = [b for b in raw_blocks if b.get("type") == 0]
        image_blocks = [b for b in raw_blocks if b.get("type") == 1]
        drawings = page.get_drawings()

        image_count = len(image_blocks) + len(page.get_images())
        drawing_count = len(drawings)

        # 1. Scanned Image PDF check
        if not text_blocks and image_count > 0:
            reasons.append("Zero extractable text blocks + embedded images detected (scanned PDF)")
            return LayoutAnalysisResult(
                complexity=LayoutComplexity.SCANNED_IMAGE,
                recommended_path="heavy",
                image_count=image_count,
                reasons=reasons,
            )

        # 2. Multi-column heuristic check
        left_side_blocks = [b for b in text_blocks if b["bbox"][0] < page_width * 0.42 and b["bbox"][1] > 100]
        right_side_blocks = [b for b in text_blocks if b["bbox"][0] >= page_width * 0.42 and b["bbox"][1] > 100]
        
        is_multi_column = len(left_side_blocks) >= 2 and len(right_side_blocks) >= 2

        # 3. Graphic / Infographic check (vector drawings or rating bars)
        is_graphic_heavy = drawing_count > 25 or (image_count > 2 and is_multi_column)

        if is_graphic_heavy:
            reasons.append(f"High vector/drawing count ({drawing_count}) or graphics + columns")
            return LayoutAnalysisResult(
                complexity=LayoutComplexity.GRAPHIC_COMPLEX,
                recommended_path="heavy",
                column_count=2 if is_multi_column else 1,
                image_count=image_count,
                reasons=reasons,
            )

        if is_multi_column:
            reasons.append("Parallel text bounding-box clusters detected across page width")
            return LayoutAnalysisResult(
                complexity=LayoutComplexity.MULTI_COLUMN,
                recommended_path="heavy",
                column_count=2,
                image_count=image_count,
                reasons=reasons,
            )

        # Default: Simple single column
        reasons.append("Standard single-column layout")
        return LayoutAnalysisResult(
            complexity=LayoutComplexity.SIMPLE_SINGLE_COLUMN,
            recommended_path="fast",
            column_count=1,
            image_count=image_count,
            reasons=reasons,
        )

    except Exception as exc:
        logger.warning("[LayoutRouter] Layout analysis failed: %s — falling back to fast path", exc)
        return LayoutAnalysisResult(
            complexity=LayoutComplexity.SIMPLE_SINGLE_COLUMN,
            recommended_path="fast",
            reasons=[f"Analysis exception: {exc}"],
        )

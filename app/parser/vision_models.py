"""
Vision-based Document Understanding Prototype (LayoutLMv3 & Donut).

Provides heavy-path visual layout extraction for complex, multi-column, 
table-heavy, or infographic-style resumes using Vision-Language / Layout Models.

This module uses lazy loading for optional deep learning dependencies 
(transformers, torch, torchvision) so the core fast pipeline remains 
lightweight and unaffected when these packages are not installed.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, List, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)

# Data structure matching app.parser.layout.TextBlock
try:
    from app.parser.layout import TextBlock, ResumeDocument
except ImportError:
    from dataclasses import dataclass, field
    
    @dataclass
    class TextBlock:
        text: str
        block_type: str
        page: int = 0
        bbox: Optional[tuple] = None
        is_bold: bool = False
        font_size: float = 0.0

    @dataclass
    class ResumeDocument:
        blocks: list[TextBlock] = field(default_factory=list)
        raw_text: str = ""
        source_path: str = ""
        parser_used: str = "unknown"
        parser_warnings: list[str] = field(default_factory=list)


def render_pdf_to_images(file_path: str, dpi: int = 150) -> List[Image.Image]:
    """
    Render PDF pages to PIL Images using PyMuPDF (fitz) or pdf2image.
    """
    images: List[Image.Image] = []
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        try:
            return [Image.open(file_path).convert("RGB")]
        except Exception as exc:
            logger.warning("[VisionParser] Failed to open image file: %s", exc)
            return []

    # Try PyMuPDF first (fastest, zero external binaries)
    try:
        import fitz
        doc = fitz.open(file_path)
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        if images:
            return images
    except Exception as exc:
        logger.debug("[VisionParser] PyMuPDF rendering fallback: %s", exc)

    # Fallback: pdf2image
    try:
        from pdf2image import convert_from_path
        return convert_from_path(file_path, dpi=dpi)
    except Exception as exc:
        logger.warning("[VisionParser] pdf2image rendering failed: %s", exc)
        return []


class VisionLayoutParser:
    """
    Vision-Language and Layout Transformer Parser.
    Supports Donut (OCR-free visual document parsing) and LayoutLMv3.
    """

    def __init__(self, model_name: str = "naver-clova-ix/donut-base-vqa"):
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._initialized = False

    def _lazy_init(self) -> bool:
        if self._initialized:
            return self._model is not None

        self._initialized = True
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            logger.info("[VisionParser] Initializing Donut model: %s", self.model_name)
            self._processor = DonutProcessor.from_pretrained(self.model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self._model.eval()
            return True
        except ImportError:
            logger.info("[VisionParser] 'transformers' or 'torch' not installed. Heavy vision path unavailable.")
            return False
        except Exception as exc:
            logger.warning("[VisionParser] Failed to load vision model '%s': %s", self.model_name, exc)
            return False

    def parse(self, file_path: str) -> Optional[ResumeDocument]:
        """
        Run Donut / LayoutLMv3 visual parsing on document pages.
        Returns a structured ResumeDocument or None if unavailable.
        """
        if not self._lazy_init():
            return None

        images = render_pdf_to_images(file_path)
        if not images:
            return None

        blocks: List[TextBlock] = []
        warnings: List[str] = []

        try:
            import torch
            for page_idx, img in enumerate(images, start=1):
                # Process image through Donut
                prompt = "<s_docvqa><s_question>Extract resume content with section headings, work experience, skills, and education.</s_question><s_answer>"
                inputs = self._processor(img, prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self._model.generate(
                        inputs.pixel_values,
                        decoder_input_ids=inputs.decoder_input_ids,
                        max_length=512,
                        early_stopping=True,
                        pad_token_id=self._processor.tokenizer.pad_token_id,
                        eos_token_id=self._processor.tokenizer.eos_token_id,
                    )

                seq = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
                cleaned_seq = seq.replace(prompt, "").strip()

                # Parse text blocks from output
                for line in cleaned_seq.splitlines():
                    line_clean = line.strip()
                    if not line_clean:
                        continue
                    
                    b_type = "paragraph"
                    if line_clean.isupper() or len(line_clean.split()) <= 3:
                        b_type = "heading"
                    
                    blocks.append(TextBlock(
                        text=line_clean,
                        block_type=b_type,
                        page=page_idx,
                    ))

            if not blocks:
                return None

            raw_text = "\n".join(b.text for b in blocks)
            return ResumeDocument(
                blocks=blocks,
                raw_text=raw_text,
                source_path=file_path,
                parser_used=f"donut_vision ({self.model_name})",
                parser_warnings=warnings,
            )

        except Exception as exc:
            logger.warning("[VisionParser] Donut vision parsing failed: %s", exc)
            return None


def parse_with_vision_model(file_path: str) -> Optional[ResumeDocument]:
    """
    Convenience entrypoint to execute heavy vision model parsing on a file.
    """
    parser = VisionLayoutParser()
    return parser.parse(file_path)

"""
Unit tests for Layout Router and Vision Models prototype.
"""
import unittest
from app.parser.router import analyze_document_layout, LayoutComplexity
from app.parser.vision_models import VisionLayoutParser, parse_with_vision_model


class TestLayoutRouterAndVision(unittest.TestCase):

    def test_router_non_pdf_fallback(self):
        result = analyze_document_layout("sample_resume.txt")
        self.assertEqual(result.recommended_path, "fast")
        self.assertEqual(result.complexity, LayoutComplexity.SIMPLE_SINGLE_COLUMN)

    def test_router_image_file(self):
        result = analyze_document_layout("sample_resume.png")
        self.assertEqual(result.recommended_path, "heavy")
        self.assertEqual(result.complexity, LayoutComplexity.GRAPHIC_COMPLEX)

    def test_vision_parser_lazy_init_graceful_fallback(self):
        from unittest.mock import patch
        with patch.object(VisionLayoutParser, "_lazy_init", return_value=False):
            parser = VisionLayoutParser(model_name="non_existent_model_test")
            res = parser.parse("non_existent_file.pdf")
            self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()

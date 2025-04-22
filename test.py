import unittest
import os
from utils.exporter import export_markdown_report, export_json_report


class TestVideoAssistantUtilities(unittest.TestCase):
    def setUp(self):
        self.mock_result = {
            "title": "Quarterly Product Strategy Review",
            "summary": "- Overview of Q3 deliverables\n- Key feature roadmap discussion",
            "action_items": "1. Shubham to refactor audio processing pipeline.",
            "key_decisions": "1. Adopt Chroma DB for vector search.",
            "open_questions": "1. Timeline for multi-language STT evaluation.",
            "transcript": "Welcome everyone to our Q3 roadmap sync. Today we discuss AI assistant features...",
        }

    def test_export_markdown_report(self):
        md = export_markdown_report(self.mock_result)
        self.assertIn("# Quarterly Product Strategy Review", md)
        self.assertIn("Shubham Sura", md)
        self.assertIn("## 📋 Executive Summary", md)
        self.assertIn("Adopt Chroma DB", md)

    def test_export_json_report(self):
        json_str = export_json_report(self.mock_result)
        self.assertIn('"title": "Quarterly Product Strategy Review"', json_str)
        self.assertIn('"generated_by": "AI Video Assistant - Shubham Sura"', json_str)


if __name__ == "__main__":
    unittest.main()

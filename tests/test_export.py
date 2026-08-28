"""§4.7 Export — format-agnostic content, per-format adapters. Scoped to
AutobioEntry only (AutobioSummary is deferred, see export.py's docstring).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photo_grouping import export  # noqa: E402


def _content(body="First paragraph.\n\nSecond paragraph.") -> export.ExportContent:
    return export.ExportContent(title="Autobio — 2026-08-14", date_label="2026-08-14", body_text=body)


class ContentForAutobioEntryTests(unittest.TestCase):
    def test_builds_content_from_an_entry_dict(self):
        entry = {"date": "2026-08-14", "final_text": "A nice day.", "draft_text": "irrelevant"}

        content = export.content_for_autobio_entry(entry)

        self.assertEqual(content.date_label, "2026-08-14")
        self.assertIn("2026-08-14", content.title)
        self.assertEqual(content.body_text, "A nice day.")

    def test_uses_final_text_not_draft_text(self):
        # final_text is always the text to show/export (see repository's
        # autobio functions) — draft_text is the as-generated original.
        entry = {"date": "2026-08-14", "final_text": "Edited version.", "draft_text": "Original draft."}

        content = export.content_for_autobio_entry(entry)

        self.assertEqual(content.body_text, "Edited version.")


class MarkdownAdapterTests(unittest.TestCase):
    def test_includes_title_date_and_body(self):
        md = export.markdown_adapter(_content())
        self.assertIn("# Autobio — 2026-08-14", md)
        self.assertIn("2026-08-14", md)
        self.assertIn("First paragraph.", md)
        self.assertIn("Second paragraph.", md)


class TextAdapterTests(unittest.TestCase):
    def test_includes_title_date_and_body_with_no_markdown_syntax(self):
        txt = export.text_adapter(_content())
        self.assertIn("Autobio — 2026-08-14", txt)
        self.assertIn("First paragraph.", txt)
        self.assertNotIn("#", txt)


class DocxAdapterTests(unittest.TestCase):
    def test_produces_a_readable_docx(self):
        import io

        from docx import Document

        data = export.docx_adapter(_content())

        doc = Document(io.BytesIO(data))
        full_text = "\n".join(p.text for p in doc.paragraphs)
        self.assertIn("Autobio — 2026-08-14", full_text)
        self.assertIn("2026-08-14", full_text)
        self.assertIn("First paragraph.", full_text)
        self.assertIn("Second paragraph.", full_text)

    def test_returns_bytes(self):
        data = export.docx_adapter(_content())
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)


class ExportDispatchTests(unittest.TestCase):
    def test_md_returns_markdown_mimetype_and_bytes(self):
        mimetype, data = export.export(_content(), "md")
        self.assertEqual(mimetype, "text/markdown")
        self.assertIsInstance(data, bytes)
        self.assertIn(b"# Autobio", data)

    def test_txt_returns_plain_mimetype(self):
        mimetype, data = export.export(_content(), "txt")
        self.assertEqual(mimetype, "text/plain")

    def test_docx_returns_docx_mimetype_and_bytes(self):
        mimetype, data = export.export(_content(), "docx")
        self.assertEqual(
            mimetype, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        self.assertIsInstance(data, bytes)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            export.export(_content(), "pdf")


if __name__ == "__main__":
    unittest.main()

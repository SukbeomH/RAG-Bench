"""Tests for markdown normalization module."""

from __future__ import annotations

from autorag_pdf_eval.normalize import (
    collapse_whitespace,
    normalize_markdown,
    strip_blockquote_markers,
    strip_bold_in_headers,
    strip_code_block_wrapper,
    strip_vlm_location_tokens,
    unify_bullet_markers,
)


class TestStripCodeBlockWrapper:
    def test_removes_markdown_fence(self):
        text = "```markdown\n# Title\ntext\n```"
        result, count = strip_code_block_wrapper(text)
        assert "```" not in result
        assert "# Title" in result
        assert count == 1

    def test_removes_md_fence(self):
        text = "```md\ntext\n```"
        result, count = strip_code_block_wrapper(text)
        assert count == 1

    def test_no_fence(self):
        text = "plain text"
        result, count = strip_code_block_wrapper(text)
        assert result == text
        assert count == 0

    def test_multiple_fences(self):
        text = "```markdown\npage1\n```\n\n```markdown\npage2\n```"
        result, count = strip_code_block_wrapper(text)
        assert count == 2
        assert "page1" in result
        assert "page2" in result


class TestUnifyBulletMarkers:
    def test_asterisk(self):
        result, count = unify_bullet_markers("* item")
        assert result == "- item"
        assert count == 1

    def test_bullet_char(self):
        result, count = unify_bullet_markers("\u2022 item")
        assert result == "- item"

    def test_plus(self):
        result, count = unify_bullet_markers("+ item")
        assert result == "- item"

    def test_hyphen_unchanged(self):
        result, count = unify_bullet_markers("- item")
        assert result == "- item"
        assert count == 0

    def test_indented(self):
        result, count = unify_bullet_markers("  * nested")
        assert result == "  - nested"


class TestStripBoldInHeaders:
    def test_bold_header(self):
        result, count = strip_bold_in_headers("# **Title**")
        assert result == "# Title"
        assert count == 1

    def test_h3_bold(self):
        result, count = strip_bold_in_headers("### **Sub**")
        assert result == "### Sub"

    def test_no_bold(self):
        result, count = strip_bold_in_headers("# Title")
        assert result == "# Title"
        assert count == 0


class TestStripBlockquoteMarkers:
    def test_blockquote(self):
        result, count = strip_blockquote_markers("> quoted text")
        assert result == "quoted text"
        assert count == 1

    def test_no_blockquote(self):
        result, count = strip_blockquote_markers("normal text")
        assert result == "normal text"
        assert count == 0


class TestStripVlmLocationTokens:
    def test_loc_tokens(self):
        text = "hello<|LOC_35|><|LOC_39|> world"
        result, count = strip_vlm_location_tokens(text)
        assert result == "hello world"
        assert count == 2

    def test_sep_token(self):
        text = "a<|SEP|>b"
        result, count = strip_vlm_location_tokens(text)
        assert result == "ab"
        assert count == 1

    def test_no_tokens(self):
        text = "normal text"
        result, count = strip_vlm_location_tokens(text)
        assert result == text
        assert count == 0


class TestCollapseWhitespace:
    def test_multiple_spaces(self):
        result, count = collapse_whitespace("hello   world")
        assert "   " not in result

    def test_trailing_spaces(self):
        result, _ = collapse_whitespace("hello   \nworld")
        assert not any(line.endswith(" ") for line in result.splitlines())

    def test_many_blank_lines(self):
        result, _ = collapse_whitespace("a\n\n\n\n\nb")
        assert result.count("\n") <= 3


class TestNormalizeMarkdown:
    def test_combined(self):
        text = "```markdown\n# **Title**\n* item\n> quote\nhello<|LOC_1|>\n```"
        result, log = normalize_markdown(text)
        assert "```" not in result
        assert "# Title" in result
        assert "- item" in result
        assert ">" not in result
        assert "<|LOC" not in result
        assert log.total_changes > 0

    def test_empty(self):
        result, log = normalize_markdown("")
        assert result == ""
        assert log.total_changes == 0

    def test_log_summary(self):
        text = "* a\n* b\n* c"
        _, log = normalize_markdown(text)
        assert "bullet_markers" in log.summary()

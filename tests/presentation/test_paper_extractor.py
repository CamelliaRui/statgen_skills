"""Tests for paper extractor."""

import pytest
from pathlib import Path


def test_extract_text_returns_string():
    """Test that extract_text returns markdown string."""
    from scripts.presentation.paper_extractor import extract_text

    # Create a simple test - will use markitdown which handles missing files
    result = extract_text(Path("/nonexistent/file.pdf"))

    # Should return empty string or error message, not crash
    assert isinstance(result, str)


def test_parse_sections_identifies_imrad():
    """Test that parse_sections identifies IMRAD sections."""
    from scripts.presentation.paper_extractor import parse_sections

    markdown_text = """
# Introduction

This is the introduction section with background information.

# Methods

This describes the methodology used.

## Data Collection

We collected data from multiple sources.

# Results

Here are the main findings.

## Statistical Analysis

The p-value was significant.

# Discussion

We discuss the implications here.

# Conclusion

In conclusion, this study shows...
"""

    sections = parse_sections(markdown_text)

    assert "introduction" in sections
    assert "methods" in sections
    assert "results" in sections
    assert "discussion" in sections or "conclusion" in sections


def test_paper_content_dataclass():
    """Test PaperContent dataclass creation."""
    from scripts.presentation.paper_extractor import PaperContent

    content = PaperContent(
        title="Test Paper",
        authors=["Author One", "Author Two"],
        abstract="This is the abstract.",
        sections={"introduction": "Intro text"},
        figures=[],
        tables=[],
        references=[],
    )

    assert content.title == "Test Paper"
    assert len(content.authors) == 2

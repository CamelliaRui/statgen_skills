"""Tests for PPTX builder."""

import pytest
from pathlib import Path
import tempfile


def test_create_presentation_returns_presentation():
    """Test that create_presentation returns a Presentation object."""
    from scripts.presentation.pptx_builder import create_presentation

    prs = create_presentation()

    # Check it has expected Presentation attributes
    assert hasattr(prs, 'slides')
    assert hasattr(prs, 'slide_width')
    assert hasattr(prs, 'slide_height')


def test_add_title_slide():
    """Test adding a title slide."""
    from scripts.presentation.pptx_builder import create_presentation, add_title_slide

    prs = create_presentation()
    add_title_slide(prs, "Test Title", "Test Subtitle")

    assert len(prs.slides) == 1


def test_add_content_slide():
    """Test adding a content slide with bullets."""
    from scripts.presentation.pptx_builder import create_presentation, add_content_slide

    prs = create_presentation()
    bullets = ["First point", "Second point", "Third point"]
    add_content_slide(prs, "Section Title", bullets)

    assert len(prs.slides) == 1


def test_save_presentation():
    """Test saving presentation to file."""
    from scripts.presentation.pptx_builder import create_presentation, add_title_slide, save_presentation

    prs = create_presentation()
    add_title_slide(prs, "Test", "Subtitle")

    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        output_path = Path(f.name)

    save_presentation(prs, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Cleanup
    output_path.unlink()


def test_add_section_header_slide():
    """Test adding a section header slide."""
    from scripts.presentation.pptx_builder import create_presentation, add_section_header_slide

    prs = create_presentation()
    add_section_header_slide(prs, "Methods")

    assert len(prs.slides) == 1

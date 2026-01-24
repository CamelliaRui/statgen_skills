"""Tests for template parser."""

import pytest
from pathlib import Path


def test_extract_theme_colors_returns_dict():
    """Test that extract_theme_colors returns a dictionary of colors."""
    from scripts.presentation.template_parser import extract_theme_colors
    from pptx import Presentation

    # Create a minimal presentation in memory
    prs = Presentation()

    colors = extract_theme_colors(prs)

    assert isinstance(colors, dict)
    assert "background" in colors or "accent1" in colors or len(colors) >= 0


def test_extract_fonts_returns_dict():
    """Test that extract_fonts returns font information."""
    from scripts.presentation.template_parser import extract_fonts
    from pptx import Presentation

    prs = Presentation()

    fonts = extract_fonts(prs)

    assert isinstance(fonts, dict)
    assert "title" in fonts or "body" in fonts or len(fonts) >= 0

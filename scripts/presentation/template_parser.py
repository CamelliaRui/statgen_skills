"""
Template parser for extracting styling from PPTX files.

Extracts colors, fonts, and layout information from a template
presentation to apply to generated slides.
"""

from pathlib import Path
from typing import Dict, Optional, Any
from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR


def extract_theme_colors(prs: Presentation) -> Dict[str, str]:
    """
    Extract theme colors from a presentation.

    Args:
        prs: python-pptx Presentation object

    Returns:
        Dictionary mapping color names to hex color strings
    """
    colors = {}

    # Try to get colors from theme
    try:
        theme = prs.slide_master.theme
        if theme is not None:
            # Theme colors are typically: dk1, lt1, dk2, lt2, accent1-6
            color_scheme = theme.color_scheme
            if color_scheme is not None:
                for name in ["dk1", "lt1", "dk2", "lt2",
                            "accent1", "accent2", "accent3",
                            "accent4", "accent5", "accent6"]:
                    try:
                        color = getattr(color_scheme, name, None)
                        if color is not None:
                            colors[name] = str(color)
                    except Exception:
                        pass
    except Exception:
        pass

    # Fallback: scan slides for commonly used colors
    if not colors:
        colors = _extract_colors_from_slides(prs)

    return colors


def _extract_colors_from_slides(prs: Presentation) -> Dict[str, str]:
    """Extract colors by scanning slide content."""
    colors = {}

    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        if run.font.color.rgb is not None:
                            rgb = run.font.color.rgb
                            hex_color = f"#{rgb}"
                            if "text" not in colors:
                                colors["text"] = hex_color

            # Check fill colors
            if hasattr(shape, "fill") and shape.fill.type is not None:
                try:
                    if shape.fill.fore_color.rgb is not None:
                        rgb = shape.fill.fore_color.rgb
                        hex_color = f"#{rgb}"
                        if "fill" not in colors:
                            colors["fill"] = hex_color
                except Exception:
                    pass

    return colors


def extract_fonts(prs: Presentation) -> Dict[str, Dict[str, Any]]:
    """
    Extract font information from a presentation.

    Args:
        prs: python-pptx Presentation object

    Returns:
        Dictionary with 'title' and 'body' font specifications
    """
    fonts = {
        "title": {"name": "Calibri", "size": 44, "bold": True},
        "body": {"name": "Calibri", "size": 18, "bold": False},
    }

    # Try to extract from slide master
    try:
        master = prs.slide_master
        for shape in master.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.runs:
                        run = para.runs[0]
                        font_info = {
                            "name": run.font.name or "Calibri",
                            "size": run.font.size.pt if run.font.size else 18,
                            "bold": run.font.bold or False,
                        }
                        # Heuristic: larger fonts are titles
                        if font_info["size"] and font_info["size"] > 30:
                            fonts["title"] = font_info
                        else:
                            fonts["body"] = font_info
                        break
    except Exception:
        pass

    return fonts

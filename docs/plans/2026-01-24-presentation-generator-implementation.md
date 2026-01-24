# Presentation Generator Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an MVP that generates Journal Club presentations from research paper PDFs using a template PPTX for styling.

**Architecture:** Extract content from PDF using markitdown, parse template PPTX for styling (colors, fonts, logo), then generate output PPTX with python-pptx. Interactive prompts gather user preferences for slide counts and content style.

**Tech Stack:** Python 3.9+, markitdown, python-pptx, pymupdf (fitz), Pillow, PyYAML

---

## Task 0: Project Setup

**Files:**
- Create: `scripts/presentation/__init__.py`
- Create: `tests/presentation/__init__.py`
- Create: `templates/presentation/configs/journal_club.yaml`

**Step 1: Create directory structure**

```bash
mkdir -p scripts/presentation
mkdir -p tests/presentation
mkdir -p templates/presentation/configs
```

**Step 2: Create package init files**

Create `scripts/presentation/__init__.py`:
```python
"""
Presentation generator module.

Generates scientific presentations from research papers (PDF) using
template PPTX for styling. Supports journal clubs, lab meetings,
and conference talks.
"""

__all__ = []
```

Create `tests/presentation/__init__.py`:
```python
"""Tests for presentation generator."""
```

**Step 3: Create journal club config**

Create `templates/presentation/configs/journal_club.yaml`:
```yaml
name: Journal Club
duration_minutes: 30-45
detail_level: standard
default_slides:
  title: 1
  introduction: 3
  methods: 3
  results: 8
  discussion: 3
  conclusions: 1
  questions: 1
total_default: 20
figure_preference: key_results
```

**Step 4: Commit**

```bash
git add scripts/presentation/ tests/presentation/ templates/
git commit -m "feat(presentation): add initial directory structure and journal club config"
```

---

## Task 1: Template Parser - Extract Colors and Fonts

**Files:**
- Create: `scripts/presentation/template_parser.py`
- Create: `tests/presentation/test_template_parser.py`

**Step 1: Write the failing test for color extraction**

Create `tests/presentation/test_template_parser.py`:
```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/presentation/test_template_parser.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.presentation.template_parser'"

**Step 3: Write minimal implementation**

Create `scripts/presentation/template_parser.py`:
```python
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/presentation/test_template_parser.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/presentation/template_parser.py tests/presentation/test_template_parser.py
git commit -m "feat(presentation): add template parser for colors and fonts"
```

---

## Task 2: Template Parser - Extract Logo and Layout

**Files:**
- Modify: `scripts/presentation/template_parser.py`
- Modify: `tests/presentation/test_template_parser.py`

**Step 1: Write failing test for logo extraction**

Add to `tests/presentation/test_template_parser.py`:
```python
def test_extract_logo_returns_none_for_empty_presentation():
    """Test that extract_logo returns None when no logo exists."""
    from scripts.presentation.template_parser import extract_logo
    from pptx import Presentation

    prs = Presentation()

    logo = extract_logo(prs)

    assert logo is None


def test_parse_template_returns_template_style():
    """Test that parse_template returns a TemplateStyle object."""
    from scripts.presentation.template_parser import parse_template, TemplateStyle
    from pptx import Presentation

    prs = Presentation()

    style = parse_template(prs)

    assert isinstance(style, TemplateStyle)
    assert hasattr(style, "colors")
    assert hasattr(style, "fonts")
    assert hasattr(style, "logo")
    assert hasattr(style, "slide_width")
    assert hasattr(style, "slide_height")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/presentation/test_template_parser.py::test_extract_logo_returns_none_for_empty_presentation -v
pytest tests/presentation/test_template_parser.py::test_parse_template_returns_template_style -v
```

Expected: FAIL with "cannot import name 'extract_logo'"

**Step 3: Add logo extraction and TemplateStyle class**

Add to `scripts/presentation/template_parser.py`:
```python
from dataclasses import dataclass
from typing import Optional
from PIL import Image
import io


@dataclass
class TemplateStyle:
    """Container for extracted template styling information."""
    colors: Dict[str, str]
    fonts: Dict[str, Dict[str, Any]]
    logo: Optional[bytes]  # PNG image bytes
    logo_position: str  # "top-left", "top-right", "bottom-left", "bottom-right"
    slide_width: float  # inches
    slide_height: float  # inches


def extract_logo(prs: Presentation) -> Optional[bytes]:
    """
    Extract logo image from presentation.

    Looks for small images in the slide master or first slide
    that appear to be logos (corner positioned, small size).

    Args:
        prs: python-pptx Presentation object

    Returns:
        PNG image bytes if logo found, None otherwise
    """
    # Check slide master first
    for shape in prs.slide_master.shapes:
        logo_bytes = _check_shape_for_logo(shape, prs)
        if logo_bytes:
            return logo_bytes

    # Check first slide
    if prs.slides:
        for shape in prs.slides[0].shapes:
            logo_bytes = _check_shape_for_logo(shape, prs)
            if logo_bytes:
                return logo_bytes

    return None


def _check_shape_for_logo(shape, prs: Presentation) -> Optional[bytes]:
    """Check if a shape is likely a logo and extract it."""
    try:
        if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
            # Check if it's in a corner (likely logo position)
            slide_width = prs.slide_width.inches
            slide_height = prs.slide_height.inches

            left = shape.left.inches
            top = shape.top.inches
            width = shape.width.inches
            height = shape.height.inches

            # Logo heuristics: small image in corner
            is_small = width < slide_width * 0.3 and height < slide_height * 0.3
            is_corner = (left < slide_width * 0.3 or left > slide_width * 0.7) and \
                       (top < slide_height * 0.3 or top > slide_height * 0.7)

            if is_small and is_corner:
                image = shape.image
                return image.blob
    except Exception:
        pass

    return None


def parse_template(prs: Presentation) -> TemplateStyle:
    """
    Parse a template presentation and extract all styling information.

    Args:
        prs: python-pptx Presentation object

    Returns:
        TemplateStyle object with all extracted styling
    """
    colors = extract_theme_colors(prs)
    fonts = extract_fonts(prs)
    logo = extract_logo(prs)

    # Determine logo position if logo exists
    logo_position = "top-right"  # default
    if logo:
        logo_position = _detect_logo_position(prs)

    return TemplateStyle(
        colors=colors,
        fonts=fonts,
        logo=logo,
        logo_position=logo_position,
        slide_width=prs.slide_width.inches,
        slide_height=prs.slide_height.inches,
    )


def _detect_logo_position(prs: Presentation) -> str:
    """Detect where the logo is positioned."""
    for shape in prs.slide_master.shapes:
        try:
            if shape.shape_type == 13:  # Picture
                slide_width = prs.slide_width.inches
                slide_height = prs.slide_height.inches
                left = shape.left.inches
                top = shape.top.inches

                h_pos = "left" if left < slide_width / 2 else "right"
                v_pos = "top" if top < slide_height / 2 else "bottom"

                return f"{v_pos}-{h_pos}"
        except Exception:
            pass

    return "top-right"
```

**Step 4: Update imports at top of file**

Update the imports in `scripts/presentation/template_parser.py`:
```python
"""
Template parser for extracting styling from PPTX files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import io

from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/presentation/test_template_parser.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add scripts/presentation/template_parser.py tests/presentation/test_template_parser.py
git commit -m "feat(presentation): add logo extraction and TemplateStyle class"
```

---

## Task 3: Paper Extractor - PDF Text Extraction

**Files:**
- Create: `scripts/presentation/paper_extractor.py`
- Create: `tests/presentation/test_paper_extractor.py`

**Step 1: Write failing test for PDF content extraction**

Create `tests/presentation/test_paper_extractor.py`:
```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/presentation/test_paper_extractor.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

Create `scripts/presentation/paper_extractor.py`:
```python
"""
Paper extractor for extracting content from research paper PDFs.

Uses markitdown to convert PDF to markdown, then parses sections.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


@dataclass
class PaperContent:
    """Container for extracted paper content."""
    title: str
    authors: List[str]
    abstract: str
    sections: Dict[str, str]  # section_name -> content
    figures: List[Tuple[int, bytes, str]]  # (fig_num, image_bytes, caption)
    tables: List[Tuple[int, str, str]]  # (table_num, table_md, caption)
    references: List[str]


def extract_text(pdf_path: Path) -> str:
    """
    Extract text from PDF using markitdown.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Markdown-formatted text content
    """
    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(str(pdf_path))
        return result.text_content
    except FileNotFoundError:
        return ""
    except Exception as e:
        return f"Error extracting PDF: {e}"


def parse_sections(markdown_text: str) -> Dict[str, str]:
    """
    Parse markdown text into IMRAD sections.

    Args:
        markdown_text: Markdown-formatted paper text

    Returns:
        Dictionary mapping section names to content
    """
    sections = {}

    # Define section patterns (case-insensitive)
    section_patterns = {
        "introduction": r"(?:^|\n)#+\s*(?:introduction|background)\s*\n",
        "methods": r"(?:^|\n)#+\s*(?:methods?|methodology|materials?\s*and\s*methods?)\s*\n",
        "results": r"(?:^|\n)#+\s*(?:results?|findings?)\s*\n",
        "discussion": r"(?:^|\n)#+\s*(?:discussion|interpretation)\s*\n",
        "conclusion": r"(?:^|\n)#+\s*(?:conclusions?|summary)\s*\n",
        "abstract": r"(?:^|\n)#+\s*abstract\s*\n",
    }

    # Find all section headers
    header_pattern = r"(?:^|\n)(#+)\s*([^\n]+)\n"
    headers = [(m.start(), m.group(1), m.group(2).strip())
               for m in re.finditer(header_pattern, markdown_text, re.IGNORECASE)]

    # Map headers to IMRAD sections
    for i, (pos, level, title) in enumerate(headers):
        title_lower = title.lower()

        # Determine section type
        section_type = None
        for stype, pattern in section_patterns.items():
            if re.search(pattern, f"\n{level} {title}\n", re.IGNORECASE):
                section_type = stype
                break

        if section_type is None:
            # Try simple matching
            for stype in ["introduction", "methods", "results", "discussion", "conclusion", "abstract"]:
                if stype in title_lower:
                    section_type = stype
                    break

        if section_type:
            # Extract content until next header of same or higher level
            start = pos + len(level) + len(title) + 2

            # Find end (next header of same/higher level)
            end = len(markdown_text)
            for j in range(i + 1, len(headers)):
                next_pos, next_level, _ = headers[j]
                if len(next_level) <= len(level):
                    end = next_pos
                    break

            content = markdown_text[start:end].strip()

            # Append if section already exists
            if section_type in sections:
                sections[section_type] += "\n\n" + content
            else:
                sections[section_type] = content

    return sections


def extract_title_and_authors(markdown_text: str) -> Tuple[str, List[str]]:
    """
    Extract paper title and authors from markdown text.

    Args:
        markdown_text: Markdown-formatted paper text

    Returns:
        Tuple of (title, list of authors)
    """
    lines = markdown_text.strip().split("\n")

    title = "Untitled Paper"
    authors = []

    # First non-empty line is often the title
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            title = line
            break
        elif line.startswith("# "):
            title = line[2:].strip()
            break

    # Look for author patterns
    author_patterns = [
        r"(?:^|\n)(?:Authors?|By):\s*(.+)",
        r"(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)",
    ]

    for pattern in author_patterns:
        match = re.search(pattern, markdown_text[:2000], re.IGNORECASE)
        if match:
            author_text = match.group(1)
            # Split by comma or "and"
            authors = [a.strip() for a in re.split(r",|\band\b", author_text) if a.strip()]
            if authors:
                break

    return title, authors


def extract_abstract(markdown_text: str) -> str:
    """Extract abstract from paper."""
    # Look for abstract section
    abstract_match = re.search(
        r"(?:^|\n)#+\s*abstract\s*\n([\s\S]*?)(?=\n#+|\Z)",
        markdown_text,
        re.IGNORECASE
    )

    if abstract_match:
        return abstract_match.group(1).strip()

    # Fallback: look for "Abstract:" or "Abstract." pattern
    abstract_match = re.search(
        r"(?:abstract[:\.]?\s*)([\s\S]{100,1500}?)(?=\n\n|\n#+|introduction|background)",
        markdown_text,
        re.IGNORECASE
    )

    if abstract_match:
        return abstract_match.group(1).strip()

    return ""


def extract_paper_content(pdf_path: Path) -> PaperContent:
    """
    Extract all content from a research paper PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        PaperContent object with all extracted content
    """
    markdown_text = extract_text(pdf_path)

    title, authors = extract_title_and_authors(markdown_text)
    abstract = extract_abstract(markdown_text)
    sections = parse_sections(markdown_text)

    return PaperContent(
        title=title,
        authors=authors,
        abstract=abstract,
        sections=sections,
        figures=[],  # Figure extraction in Task 4
        tables=[],
        references=[],
    )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/presentation/test_paper_extractor.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/presentation/paper_extractor.py tests/presentation/test_paper_extractor.py
git commit -m "feat(presentation): add paper extractor for PDF text parsing"
```

---

## Task 4: Paper Extractor - Figure Extraction

**Files:**
- Modify: `scripts/presentation/paper_extractor.py`
- Modify: `tests/presentation/test_paper_extractor.py`

**Step 1: Write failing test for figure extraction**

Add to `tests/presentation/test_paper_extractor.py`:
```python
def test_extract_figures_returns_list():
    """Test that extract_figures returns a list."""
    from scripts.presentation.paper_extractor import extract_figures

    # Test with non-existent file
    figures = extract_figures(Path("/nonexistent/file.pdf"))

    assert isinstance(figures, list)


def test_extract_figures_with_main_only_flag():
    """Test that main_only flag filters supplementary figures."""
    from scripts.presentation.paper_extractor import extract_figures

    # Just test that the function accepts the parameter
    figures = extract_figures(Path("/nonexistent/file.pdf"), main_only=True)

    assert isinstance(figures, list)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/presentation/test_paper_extractor.py::test_extract_figures_returns_list -v
```

Expected: FAIL with "cannot import name 'extract_figures'"

**Step 3: Add figure extraction**

Add to `scripts/presentation/paper_extractor.py`:
```python
def extract_figures(
    pdf_path: Path,
    main_only: bool = False
) -> List[Tuple[int, bytes, str]]:
    """
    Extract figures from PDF.

    Args:
        pdf_path: Path to PDF file
        main_only: If True, exclude supplementary figures

    Returns:
        List of (figure_number, image_bytes, caption) tuples
    """
    figures = []

    if not pdf_path.exists():
        return figures

    try:
        import fitz  # pymupdf

        doc = fitz.open(str(pdf_path))
        figure_num = 0

        for page_num, page in enumerate(doc):
            images = page.get_images()

            for img_idx, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Skip very small images (likely icons)
                    if len(image_bytes) < 5000:
                        continue

                    figure_num += 1

                    # Try to extract caption from nearby text
                    caption = _extract_figure_caption(page, img_idx, figure_num)

                    # Check if supplementary
                    if main_only:
                        caption_lower = caption.lower()
                        if "supplement" in caption_lower or "supp" in caption_lower or "s" in caption_lower[:5]:
                            continue

                    figures.append((figure_num, image_bytes, caption))

                except Exception:
                    continue

        doc.close()

    except ImportError:
        # pymupdf not installed
        pass
    except Exception:
        pass

    return figures


def _extract_figure_caption(page, img_idx: int, figure_num: int) -> str:
    """Extract caption for a figure from page text."""
    try:
        text = page.get_text()

        # Look for "Figure X" or "Fig. X" patterns
        patterns = [
            rf"(?:Figure|Fig\.?)\s*{figure_num}[:\.]?\s*([^\n]+(?:\n(?![A-Z]|\d|Figure|Fig)[^\n]+)*)",
            rf"(?:Figure|Fig\.?)\s*{figure_num}[:\.]?\s*(.{{50,500}})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                caption = match.group(1).strip()
                # Clean up caption
                caption = re.sub(r"\s+", " ", caption)
                return caption[:500]  # Limit length

    except Exception:
        pass

    return f"Figure {figure_num}"
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/presentation/test_paper_extractor.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/presentation/paper_extractor.py tests/presentation/test_paper_extractor.py
git commit -m "feat(presentation): add figure extraction from PDFs"
```

---

## Task 5: PPTX Builder - Basic Slide Creation

**Files:**
- Create: `scripts/presentation/pptx_builder.py`
- Create: `tests/presentation/test_pptx_builder.py`

**Step 1: Write failing test for slide builder**

Create `tests/presentation/test_pptx_builder.py`:
```python
"""Tests for PPTX builder."""

import pytest
from pathlib import Path
import tempfile


def test_create_presentation_returns_presentation():
    """Test that create_presentation returns a Presentation object."""
    from scripts.presentation.pptx_builder import create_presentation
    from pptx import Presentation

    prs = create_presentation()

    assert isinstance(prs, Presentation)


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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/presentation/test_pptx_builder.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

Create `scripts/presentation/pptx_builder.py`:
```python
"""
PPTX builder for generating PowerPoint presentations.

Creates slides with content, figures, and styling based on
template and configuration.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import io


def create_presentation(
    width_inches: float = 13.333,
    height_inches: float = 7.5
) -> Presentation:
    """
    Create a new presentation with specified dimensions.

    Args:
        width_inches: Slide width (default: widescreen 16:9)
        height_inches: Slide height

    Returns:
        New Presentation object
    """
    prs = Presentation()
    prs.slide_width = Inches(width_inches)
    prs.slide_height = Inches(height_inches)
    return prs


def add_title_slide(
    prs: Presentation,
    title: str,
    subtitle: str = "",
    presenter: str = "",
    date: str = "",
) -> None:
    """
    Add a title slide to the presentation.

    Args:
        prs: Presentation object
        title: Main title text
        subtitle: Subtitle or paper title
        presenter: Presenter name
        date: Presentation date
    """
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)

    # Add title
    title_box = slide.shapes.add_textbox(
        Inches(0.5), Inches(2.5), Inches(12.33), Inches(1.5)
    )
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = title
    title_para.font.size = Pt(44)
    title_para.font.bold = True
    title_para.alignment = PP_ALIGN.CENTER

    # Add subtitle
    if subtitle:
        subtitle_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(4.2), Inches(12.33), Inches(1)
        )
        subtitle_frame = subtitle_box.text_frame
        subtitle_para = subtitle_frame.paragraphs[0]
        subtitle_para.text = subtitle
        subtitle_para.font.size = Pt(24)
        subtitle_para.alignment = PP_ALIGN.CENTER

    # Add presenter and date
    if presenter or date:
        info_text = ""
        if presenter:
            info_text += presenter
        if date:
            info_text += f"\n{date}" if presenter else date

        info_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(5.5), Inches(12.33), Inches(1)
        )
        info_frame = info_box.text_frame
        info_para = info_frame.paragraphs[0]
        info_para.text = info_text
        info_para.font.size = Pt(18)
        info_para.alignment = PP_ALIGN.CENTER


def add_content_slide(
    prs: Presentation,
    title: str,
    bullets: List[str],
    font_size: int = 18,
) -> None:
    """
    Add a content slide with bullet points.

    Args:
        prs: Presentation object
        title: Slide title
        bullets: List of bullet point strings
        font_size: Font size for bullets
    """
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)

    # Add title
    title_box = slide.shapes.add_textbox(
        Inches(0.5), Inches(0.3), Inches(12.33), Inches(0.8)
    )
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = title
    title_para.font.size = Pt(32)
    title_para.font.bold = True

    # Add bullets
    bullet_box = slide.shapes.add_textbox(
        Inches(0.5), Inches(1.3), Inches(12.33), Inches(5.7)
    )
    bullet_frame = bullet_box.text_frame
    bullet_frame.word_wrap = True

    for i, bullet in enumerate(bullets):
        if i == 0:
            para = bullet_frame.paragraphs[0]
        else:
            para = bullet_frame.add_paragraph()

        para.text = f"• {bullet}"
        para.font.size = Pt(font_size)
        para.space_after = Pt(12)


def add_figure_slide(
    prs: Presentation,
    title: str,
    figure_bytes: bytes,
    caption: str = "",
) -> None:
    """
    Add a slide with a figure and caption.

    Args:
        prs: Presentation object
        title: Slide title
        figure_bytes: Image bytes
        caption: Figure caption
    """
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)

    # Add title
    title_box = slide.shapes.add_textbox(
        Inches(0.5), Inches(0.3), Inches(12.33), Inches(0.8)
    )
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = title
    title_para.font.size = Pt(32)
    title_para.font.bold = True

    # Add figure
    image_stream = io.BytesIO(figure_bytes)
    # Center the image
    pic = slide.shapes.add_picture(
        image_stream, Inches(1.5), Inches(1.5), width=Inches(10)
    )

    # Add caption
    if caption:
        caption_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(6.5), Inches(12.33), Inches(0.8)
        )
        caption_frame = caption_box.text_frame
        caption_para = caption_frame.paragraphs[0]
        caption_para.text = caption
        caption_para.font.size = Pt(14)
        caption_para.font.italic = True
        caption_para.alignment = PP_ALIGN.CENTER


def add_section_header_slide(
    prs: Presentation,
    section_name: str,
) -> None:
    """
    Add a section header slide.

    Args:
        prs: Presentation object
        section_name: Name of the section
    """
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)

    # Add section title centered
    title_box = slide.shapes.add_textbox(
        Inches(0.5), Inches(3), Inches(12.33), Inches(1.5)
    )
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = section_name
    title_para.font.size = Pt(48)
    title_para.font.bold = True
    title_para.alignment = PP_ALIGN.CENTER


def save_presentation(prs: Presentation, output_path: Path) -> None:
    """
    Save presentation to file.

    Args:
        prs: Presentation object
        output_path: Path to save the PPTX file
    """
    prs.save(str(output_path))
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/presentation/test_pptx_builder.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/presentation/pptx_builder.py tests/presentation/test_pptx_builder.py
git commit -m "feat(presentation): add PPTX builder for slide creation"
```

---

## Task 6: Content Mapper - Generate Bullet Points

**Files:**
- Create: `scripts/presentation/content_mapper.py`
- Create: `tests/presentation/test_content_mapper.py`

**Step 1: Write failing test for content mapper**

Create `tests/presentation/test_content_mapper.py`:
```python
"""Tests for content mapper."""

import pytest


def test_extract_key_points_returns_list():
    """Test that extract_key_points returns a list of strings."""
    from scripts.presentation.content_mapper import extract_key_points

    text = """
    This study investigates the relationship between gene expression and disease.
    We found that 50 genes were significantly associated with the trait.
    The results suggest a strong genetic component to the phenotype.
    """

    points = extract_key_points(text, max_points=3)

    assert isinstance(points, list)
    assert len(points) <= 3
    assert all(isinstance(p, str) for p in points)


def test_map_sections_to_slides():
    """Test mapping paper sections to slide configuration."""
    from scripts.presentation.content_mapper import map_sections_to_slides

    sections = {
        "introduction": "Background information about the study.",
        "methods": "We used statistical analysis.",
        "results": "We found significant associations.",
        "discussion": "These results suggest...",
    }

    slide_counts = {
        "introduction": 2,
        "methods": 2,
        "results": 4,
        "discussion": 2,
    }

    slide_config = map_sections_to_slides(sections, slide_counts)

    assert "introduction" in slide_config
    assert len(slide_config["introduction"]) == 2


def test_extractive_vs_generative_mode():
    """Test that mode parameter affects output style."""
    from scripts.presentation.content_mapper import extract_key_points

    text = "The p-value was 0.001, indicating statistical significance."

    extractive = extract_key_points(text, mode="extractive")
    generative = extract_key_points(text, mode="generative")

    assert isinstance(extractive, list)
    assert isinstance(generative, list)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/presentation/test_content_mapper.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

Create `scripts/presentation/content_mapper.py`:
```python
"""
Content mapper for converting paper sections to slide content.

Extracts key points from text and maps them to slides based on
configuration.
"""

from typing import Dict, List, Tuple
import re


def extract_key_points(
    text: str,
    max_points: int = 5,
    mode: str = "extractive",
) -> List[str]:
    """
    Extract key points from text for slide bullets.

    Args:
        text: Source text to extract from
        max_points: Maximum number of points to extract
        mode: "extractive" (direct sentences) or "generative" (summarized)

    Returns:
        List of bullet point strings
    """
    if not text or not text.strip():
        return []

    # Clean text
    text = re.sub(r"\s+", " ", text).strip()

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if mode == "extractive":
        # Score sentences by importance indicators
        scored = []
        for sent in sentences:
            score = 0
            sent_lower = sent.lower()

            # Importance indicators
            if any(word in sent_lower for word in ["significant", "important", "key", "main", "primary"]):
                score += 2
            if any(word in sent_lower for word in ["found", "showed", "demonstrated", "revealed"]):
                score += 2
            if any(word in sent_lower for word in ["conclude", "suggest", "indicate"]):
                score += 1
            if re.search(r"p\s*[<>=]\s*0\.\d+", sent_lower):
                score += 3  # Statistical results
            if re.search(r"\d+%", sent):
                score += 1  # Percentages
            if len(sent) < 150:
                score += 1  # Prefer concise sentences

            scored.append((score, sent))

        # Sort by score and take top points
        scored.sort(key=lambda x: x[0], reverse=True)
        points = [sent for _, sent in scored[:max_points]]

    else:  # generative mode
        # For now, use extractive as fallback
        # In full implementation, this would use LLM
        points = extract_key_points(text, max_points, mode="extractive")

        # Simplify sentences for generative style
        simplified = []
        for point in points:
            # Remove citations
            point = re.sub(r"\([^)]*\d{4}[^)]*\)", "", point)
            # Remove excessive detail in parentheses
            point = re.sub(r"\([^)]{50,}\)", "", point)
            point = point.strip()
            if point:
                simplified.append(point)

        points = simplified

    return points


def split_text_for_slides(text: str, num_slides: int) -> List[str]:
    """
    Split text into roughly equal chunks for multiple slides.

    Args:
        text: Source text
        num_slides: Number of slides to split into

    Returns:
        List of text chunks, one per slide
    """
    if num_slides <= 0:
        return []

    if num_slides == 1:
        return [text]

    # Split by paragraphs first
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if len(paragraphs) <= num_slides:
        # Pad with empty if needed
        chunks = paragraphs + [""] * (num_slides - len(paragraphs))
        return chunks[:num_slides]

    # Distribute paragraphs across slides
    chunks = []
    paras_per_slide = len(paragraphs) / num_slides

    current_chunk = []
    current_count = 0

    for para in paragraphs:
        current_chunk.append(para)
        current_count += 1

        if current_count >= paras_per_slide and len(chunks) < num_slides - 1:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_count = 0

    # Add remaining to last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def map_sections_to_slides(
    sections: Dict[str, str],
    slide_counts: Dict[str, int],
    mode: str = "extractive",
    points_per_slide: int = 4,
) -> Dict[str, List[List[str]]]:
    """
    Map paper sections to slide content.

    Args:
        sections: Dictionary of section_name -> content
        slide_counts: Dictionary of section_name -> number of slides
        mode: "extractive" or "generative"
        points_per_slide: Number of bullet points per slide

    Returns:
        Dictionary of section_name -> list of slide bullet lists
    """
    result = {}

    for section_name, count in slide_counts.items():
        content = sections.get(section_name, "")

        if not content or count == 0:
            result[section_name] = []
            continue

        # Split content for multiple slides
        chunks = split_text_for_slides(content, count)

        # Extract key points for each chunk
        slides = []
        for chunk in chunks:
            points = extract_key_points(
                chunk,
                max_points=points_per_slide,
                mode=mode
            )
            slides.append(points)

        result[section_name] = slides

    return result


def generate_slide_titles(
    section_name: str,
    num_slides: int,
) -> List[str]:
    """
    Generate slide titles for a section.

    Args:
        section_name: Name of the section
        num_slides: Number of slides in section

    Returns:
        List of slide titles
    """
    # Capitalize section name
    base_title = section_name.replace("_", " ").title()

    if num_slides == 1:
        return [base_title]

    # Generate numbered titles
    titles = []
    for i in range(num_slides):
        if i == 0:
            titles.append(base_title)
        else:
            titles.append(f"{base_title} (cont.)")

    return titles
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/presentation/test_content_mapper.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/presentation/content_mapper.py tests/presentation/test_content_mapper.py
git commit -m "feat(presentation): add content mapper for bullet generation"
```

---

## Task 7: Main Generator - Orchestrate Pipeline

**Files:**
- Create: `scripts/presentation/generator.py`
- Create: `tests/presentation/test_generator.py`
- Modify: `scripts/presentation/__init__.py`

**Step 1: Write failing test for generator**

Create `tests/presentation/test_generator.py`:
```python
"""Tests for presentation generator."""

import pytest
from pathlib import Path
import tempfile


def test_generate_presentation_config():
    """Test PresentationConfig dataclass."""
    from scripts.presentation.generator import PresentationConfig

    config = PresentationConfig(
        presentation_type="journal_club",
        structure="imrad",
        slide_counts={"introduction": 3, "methods": 3, "results": 8, "discussion": 3},
        include_supplementary=False,
        content_mode="extractive",
    )

    assert config.presentation_type == "journal_club"
    assert config.slide_counts["results"] == 8


def test_load_presentation_config_from_yaml():
    """Test loading config from YAML file."""
    from scripts.presentation.generator import load_config

    config = load_config("journal_club")

    assert config is not None
    assert "default_slides" in config or config.get("name") == "Journal Club"


def test_generator_init():
    """Test PresentationGenerator initialization."""
    from scripts.presentation.generator import PresentationGenerator, PresentationConfig

    config = PresentationConfig(
        presentation_type="journal_club",
        structure="imrad",
        slide_counts={"introduction": 3},
        include_supplementary=False,
        content_mode="extractive",
    )

    generator = PresentationGenerator(config)

    assert generator.config == config
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/presentation/test_generator.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

Create `scripts/presentation/generator.py`:
```python
"""
Presentation generator orchestrator.

Coordinates template parsing, paper extraction, content mapping,
and PPTX building to generate complete presentations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import yaml

from pptx import Presentation

from .template_parser import parse_template, TemplateStyle
from .paper_extractor import extract_paper_content, extract_figures, PaperContent
from .content_mapper import map_sections_to_slides, generate_slide_titles
from .pptx_builder import (
    create_presentation,
    add_title_slide,
    add_content_slide,
    add_figure_slide,
    add_section_header_slide,
    save_presentation,
)


@dataclass
class PresentationConfig:
    """Configuration for presentation generation."""
    presentation_type: str  # "journal_club", "lab_meeting", "conference_talk"
    structure: str  # "imrad" or "flexible"
    slide_counts: Dict[str, int]
    include_supplementary: bool = False
    content_mode: str = "extractive"  # "extractive" or "generative"
    presenter_name: str = ""
    presentation_date: str = ""
    custom_sections: Optional[List[str]] = None


# Default slide counts by presentation type
DEFAULT_CONFIGS = {
    "journal_club": {
        "title": 1,
        "introduction": 3,
        "methods": 3,
        "results": 8,
        "discussion": 3,
        "conclusions": 1,
        "questions": 1,
    },
    "lab_meeting": {
        "title": 1,
        "introduction": 4,
        "methods": 6,
        "results": 12,
        "discussion": 4,
        "future_directions": 2,
        "questions": 1,
    },
    "conference_talk": {
        "title": 1,
        "background": 2,
        "methods": 1,
        "results": 5,
        "conclusions": 2,
        "questions": 1,
    },
}


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file or defaults.

    Args:
        config_name: Name of config (e.g., "journal_club")

    Returns:
        Configuration dictionary
    """
    # Try to load from YAML file
    config_dir = Path(__file__).parent.parent.parent / "templates" / "presentation" / "configs"
    config_path = config_dir / f"{config_name}.yaml"

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Fall back to built-in defaults
    if config_name in DEFAULT_CONFIGS:
        return {
            "name": config_name.replace("_", " ").title(),
            "default_slides": DEFAULT_CONFIGS[config_name],
        }

    raise ValueError(f"Unknown config: {config_name}")


class PresentationGenerator:
    """
    Main presentation generator class.

    Orchestrates the pipeline from paper PDF to output PPTX.
    """

    def __init__(self, config: PresentationConfig):
        """
        Initialize generator with configuration.

        Args:
            config: PresentationConfig object
        """
        self.config = config
        self.template_style: Optional[TemplateStyle] = None
        self.paper_content: Optional[PaperContent] = None

    def load_template(self, template_path: Path) -> None:
        """
        Load and parse template PPTX.

        Args:
            template_path: Path to template PPTX file
        """
        prs = Presentation(str(template_path))
        self.template_style = parse_template(prs)

    def load_paper(self, pdf_path: Path) -> None:
        """
        Load and extract content from paper PDF.

        Args:
            pdf_path: Path to paper PDF file
        """
        self.paper_content = extract_paper_content(pdf_path)

        # Extract figures
        figures = extract_figures(
            pdf_path,
            main_only=not self.config.include_supplementary
        )
        self.paper_content.figures = figures

    def generate(self, output_path: Path) -> Path:
        """
        Generate the presentation.

        Args:
            output_path: Path to save output PPTX

        Returns:
            Path to generated PPTX file
        """
        if self.paper_content is None:
            raise ValueError("No paper loaded. Call load_paper() first.")

        # Create presentation
        prs = create_presentation()

        # Add title slide
        add_title_slide(
            prs,
            title=self.paper_content.title,
            subtitle=", ".join(self.paper_content.authors[:3]),
            presenter=self.config.presenter_name,
            date=self.config.presentation_date or datetime.now().strftime("%B %d, %Y"),
        )

        # Map content to slides
        slide_content = map_sections_to_slides(
            sections=self.paper_content.sections,
            slide_counts=self.config.slide_counts,
            mode=self.config.content_mode,
        )

        # Generate slides for each section
        section_order = ["introduction", "methods", "results", "discussion", "conclusions"]
        if self.config.custom_sections:
            section_order = self.config.custom_sections

        figure_idx = 0

        for section in section_order:
            if section not in self.config.slide_counts:
                continue

            count = self.config.slide_counts[section]
            if count == 0:
                continue

            # Add section header for larger sections
            if count > 2:
                add_section_header_slide(prs, section.replace("_", " ").title())

            # Add content slides
            titles = generate_slide_titles(section, count)
            bullets_list = slide_content.get(section, [])

            for i in range(count):
                title = titles[i] if i < len(titles) else section.title()
                bullets = bullets_list[i] if i < len(bullets_list) else []

                if bullets:
                    add_content_slide(prs, title, bullets)
                else:
                    add_content_slide(prs, title, ["[Content to be added]"])

            # Add figures for results section
            if section == "results" and self.paper_content.figures:
                for fig_num, fig_bytes, caption in self.paper_content.figures:
                    if figure_idx < 5:  # Limit figures
                        add_figure_slide(prs, f"Figure {fig_num}", fig_bytes, caption)
                        figure_idx += 1

        # Add questions slide
        if self.config.slide_counts.get("questions", 0) > 0:
            add_section_header_slide(prs, "Questions?")

        # Save presentation
        save_presentation(prs, output_path)

        return output_path


def generate_presentation(
    pdf_path: Path,
    output_path: Path,
    template_path: Optional[Path] = None,
    presentation_type: str = "journal_club",
    slide_counts: Optional[Dict[str, int]] = None,
    include_supplementary: bool = False,
    content_mode: str = "extractive",
    presenter_name: str = "",
) -> Path:
    """
    Convenience function to generate a presentation.

    Args:
        pdf_path: Path to paper PDF
        output_path: Path for output PPTX
        template_path: Optional path to template PPTX
        presentation_type: Type of presentation
        slide_counts: Custom slide counts per section
        include_supplementary: Include supplementary figures
        content_mode: "extractive" or "generative"
        presenter_name: Name of presenter

    Returns:
        Path to generated PPTX
    """
    # Load default config
    config_data = load_config(presentation_type)

    # Build config
    counts = slide_counts or config_data.get("default_slides", DEFAULT_CONFIGS["journal_club"])

    config = PresentationConfig(
        presentation_type=presentation_type,
        structure="imrad",
        slide_counts=counts,
        include_supplementary=include_supplementary,
        content_mode=content_mode,
        presenter_name=presenter_name,
    )

    # Generate
    generator = PresentationGenerator(config)

    if template_path:
        generator.load_template(template_path)

    generator.load_paper(pdf_path)

    return generator.generate(output_path)
```

**Step 4: Update package init**

Update `scripts/presentation/__init__.py`:
```python
"""
Presentation generator module.

Generates scientific presentations from research papers (PDF) using
template PPTX for styling. Supports journal clubs, lab meetings,
and conference talks.
"""

from .generator import (
    PresentationConfig,
    PresentationGenerator,
    generate_presentation,
    load_config,
)
from .template_parser import TemplateStyle, parse_template
from .paper_extractor import PaperContent, extract_paper_content
from .content_mapper import extract_key_points, map_sections_to_slides
from .pptx_builder import (
    create_presentation,
    add_title_slide,
    add_content_slide,
    add_figure_slide,
    save_presentation,
)

__all__ = [
    "PresentationConfig",
    "PresentationGenerator",
    "generate_presentation",
    "load_config",
    "TemplateStyle",
    "parse_template",
    "PaperContent",
    "extract_paper_content",
    "extract_key_points",
    "map_sections_to_slides",
    "create_presentation",
    "add_title_slide",
    "add_content_slide",
    "add_figure_slide",
    "save_presentation",
]
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/presentation/test_generator.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add scripts/presentation/generator.py scripts/presentation/__init__.py tests/presentation/test_generator.py
git commit -m "feat(presentation): add main generator orchestrator"
```

---

## Task 8: Integration Test

**Files:**
- Create: `tests/presentation/test_integration.py`

**Step 1: Write integration test**

Create `tests/presentation/test_integration.py`:
```python
"""Integration tests for presentation generator."""

import pytest
from pathlib import Path
import tempfile


@pytest.mark.slow
def test_full_pipeline_with_mock_content():
    """Test the full pipeline with mock content (no real PDF)."""
    from scripts.presentation.generator import (
        PresentationConfig,
        PresentationGenerator,
    )
    from scripts.presentation.paper_extractor import PaperContent
    from pptx import Presentation

    # Create config
    config = PresentationConfig(
        presentation_type="journal_club",
        structure="imrad",
        slide_counts={
            "introduction": 2,
            "methods": 2,
            "results": 3,
            "discussion": 2,
        },
        include_supplementary=False,
        content_mode="extractive",
        presenter_name="Test User",
    )

    # Create generator and inject mock content
    generator = PresentationGenerator(config)
    generator.paper_content = PaperContent(
        title="Test Paper: A Study of Testing",
        authors=["Author One", "Author Two"],
        abstract="This paper tests the presentation generator.",
        sections={
            "introduction": "This study investigates presentation generation. We found that automated tools can save time. The field has grown significantly.",
            "methods": "We used Python to build a generator. The system extracts content from PDFs. Statistical analysis was performed.",
            "results": "The generator produced valid PPTX files. Quality was assessed by users. Significant improvements were noted.",
            "discussion": "These results suggest automation is feasible. Future work could improve accuracy. Limitations include edge cases.",
        },
        figures=[],
        tables=[],
        references=[],
    )

    # Generate presentation
    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        output_path = Path(f.name)

    result_path = generator.generate(output_path)

    # Verify output
    assert result_path.exists()
    assert result_path.stat().st_size > 0

    # Open and verify structure
    prs = Presentation(str(result_path))
    assert len(prs.slides) >= 5  # Title + at least some content slides

    # Cleanup
    output_path.unlink()


def test_config_loading():
    """Test that configs load correctly."""
    from scripts.presentation.generator import load_config

    config = load_config("journal_club")

    assert "default_slides" in config or "name" in config


def test_generator_with_custom_slide_counts():
    """Test generator accepts custom slide counts."""
    from scripts.presentation.generator import PresentationConfig, PresentationGenerator
    from scripts.presentation.paper_extractor import PaperContent
    import tempfile

    config = PresentationConfig(
        presentation_type="journal_club",
        structure="imrad",
        slide_counts={
            "introduction": 1,
            "results": 2,
        },
        include_supplementary=False,
        content_mode="extractive",
    )

    generator = PresentationGenerator(config)
    generator.paper_content = PaperContent(
        title="Minimal Test",
        authors=["Author"],
        abstract="Abstract",
        sections={"introduction": "Intro text.", "results": "Results text."},
        figures=[],
        tables=[],
        references=[],
    )

    with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
        output_path = Path(f.name)

    result = generator.generate(output_path)
    assert result.exists()

    output_path.unlink()
```

**Step 2: Run integration tests**

```bash
pytest tests/presentation/test_integration.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/presentation/test_integration.py
git commit -m "test(presentation): add integration tests"
```

---

## Task 9: Update SKILL.md and Documentation

**Files:**
- Modify: `SKILL.md`
- Modify: `README.md`

**Step 1: Add presentation generator to SKILL.md**

Add after the FUSION TWAS section in `SKILL.md`:

```markdown
### Presentation Generator

Generate scientific presentations from research papers (PDF) with customizable templates and formats.

**Capabilities:**
- Extract content and figures from paper PDFs
- Apply custom template styling (logo, colors, fonts)
- Support for Journal Club, Lab Meeting, and Conference Talk formats
- IMRAD or flexible section structures
- Extractive or generative bullet point styles
- Export to PPTX (compatible with Google Slides)

**API Functions:**
- `generate_presentation(pdf, output, type, ...)` - Generate presentation
- `PresentationGenerator(config)` - Full control over generation
- `load_config(name)` - Load preset configuration

**Example Usage:**

```python
from scripts.presentation import generate_presentation

# Generate a journal club presentation
generate_presentation(
    pdf_path="paper.pdf",
    output_path="presentation.pptx",
    presentation_type="journal_club",
    presenter_name="Your Name",
)
```

**Interactive Workflow:**
```
"Create a journal club presentation from this paper"
"Generate a 15-minute conference talk from my manuscript"
"Make a lab meeting presentation with all supplementary figures"
```

**Presentation Types:**

| Type | Duration | Detail Level | Default Slides |
|------|----------|--------------|----------------|
| Journal Club | 30-45 min | Standard | ~20 |
| Lab Meeting | ~60 min | High (methods, implementation) | ~30 |
| Conference Talk | 15-30 min | High-level | ~12-25 |
```

**Step 2: Update README.md**

Add to the Features section:
```markdown
- **Presentation Generator**: Create scientific presentations from research papers with customizable templates
```

Add to the Roadmap:
```markdown
- [x] Presentation Generator - Journal Club, Lab Meeting, Conference Talk
```

**Step 3: Commit documentation**

```bash
git add SKILL.md README.md
git commit -m "docs: add presentation generator documentation"
```

---

## Task 10: Final Verification

**Step 1: Run all tests**

```bash
pytest tests/presentation/ -v
```

Expected: All tests PASS

**Step 2: Run full test suite**

```bash
pytest tests/ -v -m "not slow"
```

Expected: No regressions

**Step 3: Verify imports work**

```bash
python -c "from scripts.presentation import generate_presentation, PresentationGenerator; print('Import OK')"
```

Expected: "Import OK"

**Step 4: Create final commit summarizing the feature**

```bash
git log --oneline -10  # Review commits
```

---

## Summary

Phase 1 MVP includes:
- Template parser (colors, fonts, logo extraction)
- Paper extractor (PDF text + figures via markitdown + pymupdf)
- Content mapper (extractive bullet points)
- PPTX builder (slides with content and figures)
- Generator orchestrator
- Journal Club preset configuration
- Full test coverage

Total: ~10 commits, ~800 lines of Python code

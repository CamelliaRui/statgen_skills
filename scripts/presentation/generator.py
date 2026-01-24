"""
Main generator orchestrator for presentation generation.

Coordinates template parsing, paper extraction, content mapping,
and PPTX building to create complete presentations from research papers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .template_parser import TemplateStyle, parse_template
from .paper_extractor import PaperContent, extract_paper_content
from .content_mapper import map_sections_to_slides, generate_slide_titles
from .pptx_builder import (
    create_presentation,
    add_title_slide,
    add_content_slide,
    add_section_header_slide,
    add_figure_slide,
    save_presentation,
)


# Slide dimension constants (16:9 widescreen)
DEFAULT_SLIDE_WIDTH = 13.333  # inches
DEFAULT_SLIDE_HEIGHT = 7.5    # inches

# Author display constants
MAX_AUTHORS_DISPLAYED = 3

# Section order for IMRAD structure
IMRAD_SECTION_ORDER = ["introduction", "methods", "results", "discussion", "conclusions"]


# Default configuration directory
CONFIG_DIR = Path(__file__).parent.parent.parent / "templates" / "presentation" / "configs"


@dataclass
class PresentationConfig:
    """Configuration for presentation generation."""

    presentation_type: str  # "journal_club", "lab_meeting", "conference_talk"
    structure: str  # "imrad", "custom"
    slide_counts: Dict[str, int]  # section -> number of slides
    include_supplementary: bool
    content_mode: str  # "extractive" or "generative"
    presenter_name: str = ""
    presentation_date: str = ""
    custom_sections: Optional[List[str]] = None


# Default configurations for common presentation types
DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "journal_club": {
        "name": "Journal Club",
        "duration_minutes": "30-45",
        "detail_level": "standard",
        "default_slides": {
            "title": 1,
            "introduction": 3,
            "methods": 3,
            "results": 8,
            "discussion": 3,
            "conclusions": 1,
            "questions": 1,
        },
        "total_default": 20,
        "figure_preference": "key_results",
    },
    "lab_meeting": {
        "name": "Lab Meeting",
        "duration_minutes": "45-60",
        "detail_level": "detailed",
        "default_slides": {
            "title": 1,
            "introduction": 4,
            "methods": 5,
            "results": 10,
            "discussion": 4,
            "conclusions": 1,
            "questions": 1,
        },
        "total_default": 26,
        "figure_preference": "all",
    },
    "conference_talk": {
        "name": "Conference Talk",
        "duration_minutes": "15-20",
        "detail_level": "concise",
        "default_slides": {
            "title": 1,
            "introduction": 2,
            "methods": 2,
            "results": 5,
            "discussion": 2,
            "conclusions": 1,
            "questions": 1,
        },
        "total_default": 14,
        "figure_preference": "key_results",
    },
}


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load presentation configuration from YAML file or defaults.

    Args:
        config_name: Name of the configuration (e.g., "journal_club")

    Returns:
        Dictionary containing configuration settings
    """
    # Try to load from YAML file first
    yaml_path = CONFIG_DIR / f"{config_name}.yaml"

    if yaml_path.exists():
        try:
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            # Fall through to defaults on YAML parse error
            pass

    # Return default config if available
    if config_name in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[config_name]

    # Return a minimal default for unknown types
    return {
        "name": config_name.replace("_", " ").title(),
        "default_slides": {
            "title": 1,
            "introduction": 2,
            "methods": 2,
            "results": 4,
            "discussion": 2,
            "conclusions": 1,
        },
        "total_default": 12,
    }


class PresentationGenerator:
    """
    Main orchestrator for generating presentations from research papers.

    Coordinates the pipeline:
    1. Load and parse template PPTX for styling
    2. Extract content from research paper PDF
    3. Map content to slides based on configuration
    4. Build PPTX with content and styling
    """

    def __init__(self, config: PresentationConfig) -> None:
        """
        Initialize the presentation generator.

        Args:
            config: PresentationConfig with generation settings
        """
        self.config = config
        self.template_style: Optional[TemplateStyle] = None
        self.paper_content: Optional[PaperContent] = None

    def load_template(self, template_path: Path) -> None:
        """
        Load and parse a template PPTX for styling.

        Args:
            template_path: Path to template PPTX file
        """
        if not template_path.exists():
            self.template_style = None
            return

        try:
            from pptx import Presentation as PptxPresentation

            prs = PptxPresentation(str(template_path))
            self.template_style = parse_template(prs)
        except Exception:
            # Template parsing failed - proceed without template styling
            self.template_style = None

    def load_paper(self, pdf_path: Path) -> None:
        """
        Load and extract content from a research paper PDF.

        Args:
            pdf_path: Path to PDF file
        """
        self.paper_content = extract_paper_content(pdf_path)

    def generate(self, output_path: Path) -> Path:
        """
        Generate the presentation and save to output path.

        Args:
            output_path: Path to save the generated PPTX

        Returns:
            Path to the generated presentation
        """
        # Determine slide dimensions from template or use defaults
        width = DEFAULT_SLIDE_WIDTH
        height = DEFAULT_SLIDE_HEIGHT

        if self.template_style:
            width = self.template_style.slide_width
            height = self.template_style.slide_height

        # Create new presentation
        prs = create_presentation(width, height)

        # Add title slide
        title = "Research Paper Presentation"
        subtitle = ""
        if self.paper_content:
            title = self.paper_content.title
            if self.paper_content.authors:
                subtitle = ", ".join(self.paper_content.authors[:MAX_AUTHORS_DISPLAYED])
                if len(self.paper_content.authors) > MAX_AUTHORS_DISPLAYED:
                    subtitle += " et al."

        add_title_slide(prs, title, subtitle)

        # Get sections from paper content
        sections = {}
        if self.paper_content:
            sections = self.paper_content.sections

        # Map sections to slide content
        slide_content = map_sections_to_slides(
            sections,
            self.config.slide_counts,
            mode=self.config.content_mode,
        )

        # Build slides for each section
        section_order = IMRAD_SECTION_ORDER

        for section_name in section_order:
            if section_name not in self.config.slide_counts:
                continue

            num_slides = self.config.slide_counts[section_name]
            if num_slides == 0:
                continue

            # Add section header if there are slides
            add_section_header_slide(prs, section_name.title())

            # Get slide titles
            titles = generate_slide_titles(section_name, num_slides)

            # Get slide content
            section_slides = slide_content.get(section_name, [])

            # Add content slides
            for i in range(num_slides):
                slide_title = titles[i] if i < len(titles) else f"{section_name.title()}"

                # Get bullets for this slide
                bullets = []
                if i < len(section_slides):
                    bullets = section_slides[i]

                if not bullets:
                    bullets = [f"Content for {section_name} slide {i + 1}"]

                add_content_slide(prs, slide_title, bullets)

        # Add figures if available and configured
        if self.paper_content and self.paper_content.figures and self.config.include_supplementary:
            for fig_num, fig_bytes, caption in self.paper_content.figures:
                add_figure_slide(prs, f"Figure {fig_num}", fig_bytes, caption)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save presentation
        save_presentation(prs, output_path)

        return output_path


def generate_presentation(
    output_path: Path,
    pdf_path: Optional[Path] = None,
    template_path: Optional[Path] = None,
    presentation_type: str = "journal_club",
    content_mode: str = "extractive",
    include_supplementary: bool = False,
) -> Path:
    """
    Convenience function to generate a presentation.

    Args:
        output_path: Path to save the generated PPTX
        pdf_path: Path to research paper PDF (optional)
        template_path: Path to template PPTX for styling (optional)
        presentation_type: Type of presentation ("journal_club", "lab_meeting", "conference_talk")
        content_mode: Content extraction mode ("extractive" or "generative")
        include_supplementary: Whether to include supplementary figures

    Returns:
        Path to the generated presentation
    """
    # Load configuration for presentation type
    config_dict = load_config(presentation_type)

    # Build slide counts from config
    slide_counts = config_dict.get("default_slides", {})
    # Remove non-content slides from counts
    content_slide_counts = {
        k: v for k, v in slide_counts.items()
        if k not in ["title", "questions"]
    }

    # Create configuration
    config = PresentationConfig(
        presentation_type=presentation_type,
        structure="imrad",
        slide_counts=content_slide_counts,
        include_supplementary=include_supplementary,
        content_mode=content_mode,
    )

    # Create generator
    generator = PresentationGenerator(config)

    # Load template if provided
    if template_path:
        generator.load_template(template_path)

    # Load paper if provided
    if pdf_path:
        generator.load_paper(pdf_path)

    # Generate and return
    return generator.generate(output_path)

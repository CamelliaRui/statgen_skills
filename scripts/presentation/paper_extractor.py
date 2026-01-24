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
    figures: List[Tuple[int, bytes, str]] = field(default_factory=list)  # (fig_num, image_bytes, caption)
    tables: List[Tuple[int, str, str]] = field(default_factory=list)  # (table_num, table_md, caption)
    references: List[str] = field(default_factory=list)


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
            # Find the actual end of the header line (after newline/whitespace prefix)
            header_end = markdown_text.find("\n", pos + 1)
            start = header_end + 1 if header_end != -1 else pos + len(level) + len(title) + 2

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
            # Split by comma or " and " (with spaces to avoid splitting names like "Anderson")
            authors = [a.strip() for a in re.split(r",\s*|\s+and\s+", author_text) if a.strip()]
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

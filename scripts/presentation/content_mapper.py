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

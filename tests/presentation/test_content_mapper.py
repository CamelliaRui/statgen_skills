"""Tests for content mapper."""

import pytest


def test_extract_key_points_returns_list():
    """Test that extract_key_points returns a list of strings."""
    from scripts.presentation.content_mapper import extract_key_points

    text = """
    This study investigates the relationship between gene expression and disease outcomes in a large cohort.
    We found that 50 genes were significantly associated with the trait using genome-wide analysis.
    The results suggest a strong genetic component to the phenotype under investigation.
    """

    points = extract_key_points(text, max_points=3)

    assert isinstance(points, list)
    assert len(points) <= 3
    assert all(isinstance(p, str) for p in points)


def test_map_sections_to_slides():
    """Test mapping paper sections to slide configuration."""
    from scripts.presentation.content_mapper import map_sections_to_slides

    sections = {
        "introduction": "Background information about the study and its importance to the field of genetics.",
        "methods": "We used statistical analysis methods including linear regression and mixed models.",
        "results": "We found significant associations between genetic variants and phenotypes of interest.",
        "discussion": "These results suggest important implications for understanding disease mechanisms.",
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

    # Use text with citations to verify generative mode removes them
    # Note: Use citation format without periods (e.g., "Smith 2020") to avoid
    # sentence splitting on "et al." which would break the citation across sentences
    text = """
    The p-value was 0.001, indicating statistical significance in the association analysis.
    Previous studies (Smith 2020) have shown similar results in European populations.
    Our findings demonstrate that genetic variants contribute to disease risk substantially.
    """

    extractive = extract_key_points(text, mode="extractive")
    generative = extract_key_points(text, mode="generative")

    assert isinstance(extractive, list)
    assert isinstance(generative, list)

    # Verify generative mode removes citations
    # Check that at least one sentence had citation removed
    extractive_with_citation = [p for p in extractive if "2020" in p]
    generative_with_citation = [p for p in generative if "2020" in p]

    # Extractive should keep citations, generative should remove them
    assert len(extractive_with_citation) > 0, "Extractive should have citation in text"
    assert len(generative_with_citation) == 0, "Generative mode should remove citations"


def test_split_text_for_slides():
    """Test the split function with multiple paragraphs."""
    from scripts.presentation.content_mapper import split_text_for_slides

    text = """First paragraph with important introduction content.

Second paragraph discussing methods and approaches.

Third paragraph presenting key results.

Fourth paragraph with discussion points."""

    # Test splitting into 2 slides
    chunks = split_text_for_slides(text, 2)
    assert len(chunks) == 2
    assert all(isinstance(c, str) for c in chunks)

    # Test splitting into 1 slide
    chunks_single = split_text_for_slides(text, 1)
    assert len(chunks_single) == 1
    assert chunks_single[0] == text

    # Test splitting into 0 slides
    chunks_zero = split_text_for_slides(text, 0)
    assert chunks_zero == []

    # Test with more slides than paragraphs
    chunks_many = split_text_for_slides(text, 10)
    assert len(chunks_many) == 10


def test_generate_slide_titles():
    """Test title generation for slides."""
    from scripts.presentation.content_mapper import generate_slide_titles

    # Test single slide
    titles_single = generate_slide_titles("introduction", 1)
    assert titles_single == ["Introduction"]

    # Test multiple slides
    titles_multi = generate_slide_titles("results", 3)
    assert len(titles_multi) == 3
    assert titles_multi[0] == "Results"
    assert titles_multi[1] == "Results (cont.)"
    assert titles_multi[2] == "Results (cont.)"

    # Test with underscore in name
    titles_underscore = generate_slide_titles("key_findings", 2)
    assert titles_underscore[0] == "Key Findings"
    assert titles_underscore[1] == "Key Findings (cont.)"


def test_extract_key_points_empty_input():
    """Test edge case for empty input."""
    from scripts.presentation.content_mapper import extract_key_points

    # Test empty string
    assert extract_key_points("") == []

    # Test whitespace only
    assert extract_key_points("   ") == []

    # Test None-like empty
    assert extract_key_points("  \n\t  ") == []


def test_extract_key_points_scoring():
    """Test that scoring prioritizes important sentences."""
    from scripts.presentation.content_mapper import extract_key_points

    text = """
    This is a simple sentence without any important keywords in it at all.
    We found that the p-value was less than 0.001 showing significant results.
    The weather was nice that day when we conducted the experiment outdoors.
    """

    points = extract_key_points(text, max_points=1, mode="extractive")

    # The sentence with p-value and "found" should score highest
    assert len(points) == 1
    assert "p-value" in points[0] or "found" in points[0]


def test_map_sections_handles_missing_sections():
    """Test that map_sections_to_slides handles missing sections gracefully."""
    from scripts.presentation.content_mapper import map_sections_to_slides

    sections = {
        "introduction": "This is the introduction section with substantial content for testing.",
    }

    slide_counts = {
        "introduction": 1,
        "methods": 2,  # This section doesn't exist
    }

    result = map_sections_to_slides(sections, slide_counts)

    assert "introduction" in result
    assert "methods" in result
    assert result["methods"] == []  # Missing section should return empty list

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

# tests/fusion/test_reference_data.py
"""Tests for FUSION reference data management."""

import pytest
from pathlib import Path


def test_list_available_tissues():
    """Test that list_available_tissues returns a list with expected tissues."""
    from scripts.fusion.reference_data import list_available_tissues

    tissues = list_available_tissues()
    assert isinstance(tissues, list)
    assert "Whole_Blood" in tissues
    assert "Brain_Cortex" in tissues


def test_get_weights_dir():
    """Test that get_weights_dir returns a Path with 'fusion_weights'."""
    from scripts.fusion.reference_data import get_weights_dir

    weights_dir = get_weights_dir()
    assert isinstance(weights_dir, Path)
    assert "fusion_weights" in str(weights_dir)


def test_get_ld_reference_dir():
    """Test that get_ld_reference_dir returns a Path with 'ld_reference'."""
    from scripts.fusion.reference_data import get_ld_reference_dir

    ld_dir = get_ld_reference_dir()
    assert isinstance(ld_dir, Path)
    assert "ld_reference" in str(ld_dir)


def test_validate_tissue_name():
    """Test that validate_tissue_name raises ValueError for invalid tissue."""
    from scripts.fusion.reference_data import validate_tissue_name

    # Valid tissue should not raise
    validate_tissue_name("Whole_Blood")

    # Invalid tissue should raise ValueError
    with pytest.raises(ValueError):
        validate_tissue_name("Invalid_Tissue_Name")


def test_validate_population():
    """Test that validate_population raises ValueError for invalid population."""
    from scripts.fusion.reference_data import validate_population

    # Valid populations should not raise
    validate_population("EUR")
    validate_population("EAS")
    validate_population("AFR")

    # Invalid population should raise ValueError
    with pytest.raises(ValueError):
        validate_population("INVALID")

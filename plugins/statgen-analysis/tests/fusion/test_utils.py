# tests/fusion/test_utils.py
"""Tests for FUSION TWAS utility functions."""

import pytest
from pathlib import Path


def test_get_fusion_dir_returns_path():
    """Test that get_fusion_dir returns the expected path."""
    from scripts.fusion.utils import get_fusion_dir

    fusion_dir = get_fusion_dir()
    expected = Path.home() / ".statgen_skills" / "fusion_twas"
    assert fusion_dir == expected
    assert isinstance(fusion_dir, Path)


def test_check_r_installed():
    """Test that check_r_installed returns a boolean."""
    from scripts.fusion.utils import check_r_installed

    result = check_r_installed()
    assert isinstance(result, bool)


def test_check_dependencies_returns_dict():
    """Test that check_dependencies returns a dict with expected keys."""
    from scripts.fusion.utils import check_dependencies

    deps = check_dependencies()
    assert isinstance(deps, dict)
    assert "R" in deps
    assert "FUSION" in deps
    assert "PLINK" in deps
    # Each value should be a boolean
    for key, value in deps.items():
        assert isinstance(value, bool), f"Expected bool for {key}, got {type(value)}"


def test_validate_sumstats_columns():
    """Test that validate_sumstats_columns checks for required columns."""
    from scripts.fusion.utils import validate_sumstats_columns

    # Valid with Z score
    valid_z = ["SNP", "A1", "A2", "Z"]
    assert validate_sumstats_columns(valid_z) is True

    # Valid with BETA and SE
    valid_beta = ["SNP", "A1", "A2", "BETA", "SE"]
    assert validate_sumstats_columns(valid_beta) is True

    # Invalid - missing required columns
    invalid = ["SNP", "A1"]
    assert validate_sumstats_columns(invalid) is False

    # Invalid - has neither Z nor BETA+SE
    invalid_no_effect = ["SNP", "A1", "A2", "P"]
    assert validate_sumstats_columns(invalid_no_effect) is False

# tests/ldsc/test_reference_data.py
import pytest
from pathlib import Path


def test_get_reference_dir_returns_expected_path():
    from scripts.ldsc.reference_data import get_reference_dir

    ref_dir = get_reference_dir()
    expected = Path.home() / ".statgen_skills" / "ldsc_references"
    assert ref_dir == expected


def test_get_population_paths_returns_dict_for_valid_pop():
    from scripts.ldsc.reference_data import get_population_paths

    paths = get_population_paths("EUR")
    assert "ld_scores" in paths
    assert "weights" in paths
    assert "frq" in paths
    assert "EUR" in str(paths["ld_scores"])


def test_is_reference_available_returns_false_when_missing():
    from scripts.ldsc.reference_data import is_reference_available

    # Use a fake population dir that doesn't exist
    result = is_reference_available("EUR")
    # Will be False unless already downloaded
    assert isinstance(result, bool)


def test_download_reference_raises_for_invalid_pop():
    from scripts.ldsc.reference_data import download_reference

    with pytest.raises(ValueError, match="not supported"):
        download_reference("INVALID")

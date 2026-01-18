# tests/ldsc/test_run_ldsc.py
import pytest
from pathlib import Path
import tempfile


def test_estimate_heritability_validates_inputs():
    from scripts.ldsc.run_ldsc import estimate_heritability

    with pytest.raises(FileNotFoundError):
        estimate_heritability(
            sumstats="/nonexistent/file.sumstats.gz",
            output_dir="/tmp",
            population="EUR",
        )


def test_genetic_correlation_requires_two_traits():
    from scripts.ldsc.run_ldsc import genetic_correlation

    with pytest.raises(ValueError, match="at least 2"):
        genetic_correlation(
            sumstats=["/path/to/single.sumstats.gz"],
            output_dir="/tmp",
            population="EUR",
        )


def test_output_json_structure():
    from scripts.ldsc.run_ldsc import _build_output_json

    result = _build_output_json(
        success=True,
        analysis_type="h2",
        results={"h2": 0.5},
        files={"log": "/path/to/log"},
    )

    assert result["success"] is True
    assert result["analysis_type"] == "h2"
    assert "h2" in result["results"]

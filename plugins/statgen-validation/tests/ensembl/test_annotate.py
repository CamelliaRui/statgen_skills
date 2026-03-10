"""Tests for Ensembl VEP annotation."""

import pytest


@pytest.mark.slow
def test_annotate_by_rsid_returns_consequences():
    from scripts.ensembl.annotate import annotate_by_rsid

    results = annotate_by_rsid(["rs4420638"])
    assert len(results) == 1
    assert results[0]["most_severe_consequence"] is not None


@pytest.mark.slow
def test_get_variant_info_returns_details():
    from scripts.ensembl.annotate import get_variant_info

    info = get_variant_info("rs4420638")
    assert info["rsid"] == "rs4420638"
    assert info["alleles"] is not None
    assert info["location"] is not None


@pytest.mark.slow
def test_annotate_variants_region_format():
    from scripts.ensembl.annotate import annotate_variants

    results = annotate_variants(["21 26960070 rs116645811 G A"])
    assert len(results) == 1
    assert "most_severe_consequence" in results[0]

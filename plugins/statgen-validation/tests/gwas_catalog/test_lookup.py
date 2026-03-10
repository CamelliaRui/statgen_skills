"""Tests for GWAS Catalog API lookups."""

import pytest


@pytest.mark.slow
def test_lookup_variant_found():
    from scripts.gwas_catalog.lookup import lookup_variant

    result = lookup_variant("rs4420638")
    assert result["found"] is True
    assert result["n_associations"] > 0
    assert len(result["traits"]) > 0


@pytest.mark.slow
def test_lookup_variant_not_found():
    from scripts.gwas_catalog.lookup import lookup_variant

    result = lookup_variant("rs999999999")
    assert result["found"] is False


@pytest.mark.slow
def test_lookup_trait_returns_efo():
    from scripts.gwas_catalog.lookup import lookup_trait

    result = lookup_trait("LDL cholesterol")
    assert result["found"] is True
    assert result["n_traits"] > 0


@pytest.mark.slow
def test_bulk_lookup_variants():
    from scripts.gwas_catalog.lookup import bulk_lookup_variants

    results = bulk_lookup_variants(["rs4420638", "rs429358"])
    assert len(results) == 2
    assert all("rsid" in r for r in results)

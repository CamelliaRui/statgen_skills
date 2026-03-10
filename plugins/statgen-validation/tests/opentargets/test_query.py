"""Tests for OpenTargets API queries."""

import pytest


@pytest.mark.slow
def test_query_target_returns_gene_info():
    from scripts.opentargets.query import query_target

    result = query_target("ENSG00000169174")  # PCSK9
    assert result["found"] is True
    assert result["symbol"] == "PCSK9"
    assert result["disease_count"] > 0


@pytest.mark.slow
def test_query_target_not_found():
    from scripts.opentargets.query import query_target

    result = query_target("ENSG00000000000")
    assert result["found"] is False


@pytest.mark.slow
def test_query_disease_associations_returns_list():
    from scripts.opentargets.query import query_disease_associations

    associations = query_disease_associations("ENSG00000169174", max_results=5)
    assert isinstance(associations, list)
    assert len(associations) > 0
    assert "disease_name" in associations[0]
    assert "score" in associations[0]


@pytest.mark.slow
def test_search_returns_results():
    from scripts.opentargets.query import search

    results = search("LDL cholesterol", entity_types=["disease"])
    assert isinstance(results, list)
    assert len(results) > 0
    assert "name" in results[0]

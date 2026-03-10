# tests/fusion/test_parsers.py
"""Tests for FUSION output parsing."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path


@pytest.fixture
def mock_fusion_output():
    """Create mock FUSION output file."""
    content = """PANEL	FILE	ID	CHR	P0	P1	HSQ	BEST.GWAS.ID	BEST.GWAS.Z	EQTL.ID	EQTL.R2	EQTL.Z	EQTL.GWAS.Z	NSNP	NWGT	MODEL	MODELCV.R2	MODELCV.PV	TWAS.Z	TWAS.P
GTEx.Whole_Blood	ENSG00000123.wgt.RDat	GENE1	1	1000	2000	0.15	rs123	3.5	rs456	0.12	4.2	3.1	100	50	lasso	0.08	0.001	4.5	6.8e-6
GTEx.Whole_Blood	ENSG00000456.wgt.RDat	GENE2	1	3000	4000	0.08	rs789	2.1	rs012	0.05	2.8	1.9	80	30	enet	0.05	0.01	-2.1	0.036
GTEx.Whole_Blood	ENSG00000789.wgt.RDat	GENE3	2	5000	6000	0.22	rs345	5.2	rs678	0.18	5.5	4.8	120	75	blup	0.12	0.0001	6.2	5.3e-10
"""
    return content


def test_parse_twas_results(mock_fusion_output):
    """Test parsing FUSION output file."""
    from scripts.fusion.parsers import parse_twas_results

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(mock_fusion_output)
        f.flush()

        results = parse_twas_results(f.name)

    assert len(results) == 3
    assert results[0].gene == "GENE1"
    assert results[0].chromosome == 1
    assert results[0].twas_z == pytest.approx(4.5)
    assert results[0].twas_p == pytest.approx(6.8e-6)
    assert results[0].best_model == "lasso"

    Path(f.name).unlink()


def test_parse_twas_results_to_dataframe(mock_fusion_output):
    """Test converting results to DataFrame."""
    from scripts.fusion.parsers import parse_twas_results, results_to_dataframe

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(mock_fusion_output)
        f.flush()

        results = parse_twas_results(f.name)
        df = results_to_dataframe(results)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "gene" in df.columns
    assert "twas_z" in df.columns
    assert "twas_p" in df.columns

    Path(f.name).unlink()


def test_get_significant_results(mock_fusion_output):
    """Test filtering significant results."""
    from scripts.fusion.parsers import parse_twas_results, get_significant_results

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(mock_fusion_output)
        f.flush()

        results = parse_twas_results(f.name)
        # Use threshold 1e-5 so GENE1 (6.8e-6) and GENE3 (5.3e-10) pass
        significant = get_significant_results(results, threshold=1e-5)

    # Only GENE1 and GENE3 pass threshold (GENE2 has p=0.036)
    assert len(significant) == 2
    assert significant[0].gene == "GENE1"
    assert significant[1].gene == "GENE3"

    Path(f.name).unlink()

# tests/fusion/test_run_fusion.py
"""Tests for main FUSION TWAS runner."""

import pytest
from pathlib import Path


def test_build_fusion_command():
    """Test that build_fusion_command creates correct Rscript command."""
    from scripts.fusion.run_fusion import build_fusion_command

    cmd = build_fusion_command(
        sumstats=Path("/path/to/sumstats.txt"),
        weights_pos=Path("/path/to/weights.pos"),
        weights_dir=Path("/path/to/weights"),
        ld_ref=Path("/path/to/ldref/1000G.EUR"),
        chromosome=1,
        output=Path("/path/to/output.dat"),
        gwas_n=100000,
        coloc=False,
    )

    # Verify command is a list
    assert isinstance(cmd, list)

    # Verify command contains expected components
    assert "Rscript" in cmd[0]
    assert any("FUSION.assoc_test.R" in str(c) for c in cmd)
    assert "--sumstats" in cmd
    assert "--weights" in cmd
    assert "--chr" in cmd
    assert "1" in cmd

    # Verify paths are in command
    assert str(Path("/path/to/sumstats.txt")) in cmd
    assert str(Path("/path/to/weights.pos")) in cmd


def test_twas_results_container():
    """Test that TWASResults dataclass works properly."""
    from scripts.fusion.run_fusion import TWASResults
    from scripts.fusion.parsers import TWASResult

    # Create some mock TWASResult objects
    mock_result = TWASResult(
        gene="GENE1",
        chromosome=1,
        start=1000,
        end=2000,
        hsq=0.15,
        best_model="lasso",
        twas_z=4.5,
        twas_p=6.8e-6,
        n_snps=100,
        n_weights=50,
        cv_r2=0.08,
        cv_pvalue=0.001,
        best_gwas_snp="rs123",
        best_gwas_z=3.5,
        eqtl_snp="rs456",
        eqtl_r2=0.12,
        panel="GTEx.Whole_Blood",
        weight_file="ENSG00000123.wgt.RDat",
    )

    # Create TWASResults container
    results_container = TWASResults(
        all_results=[mock_result],
        significant=[mock_result],
        output_dir=Path("/path/to/output"),
        tissue="Whole_Blood",
        n_genes_tested=1,
    )

    # Verify container has expected attributes
    assert len(results_container.all_results) == 1
    assert len(results_container.significant) == 1
    assert results_container.output_dir == Path("/path/to/output")
    assert results_container.tissue == "Whole_Blood"
    assert results_container.n_genes_tested == 1


def test_validate_inputs():
    """Test that _validate_inputs raises ValueError for invalid tissue."""
    from scripts.fusion.run_fusion import _validate_inputs
    from pathlib import Path
    import tempfile

    # Create a temporary sumstats file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("SNP\tA1\tA2\tZ\n")
        f.write("rs1\tA\tG\t1.5\n")
        sumstats_path = Path(f.name)

    try:
        # Test invalid tissue raises ValueError
        with pytest.raises(ValueError, match="Invalid tissue name"):
            _validate_inputs(
                sumstats=sumstats_path,
                tissue="InvalidTissue",
                population="EUR",
            )

        # Test invalid population raises ValueError
        with pytest.raises(ValueError, match="Invalid population"):
            _validate_inputs(
                sumstats=sumstats_path,
                tissue="Whole_Blood",
                population="INVALID",
            )

        # Test non-existent sumstats raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            _validate_inputs(
                sumstats=Path("/nonexistent/path/sumstats.txt"),
                tissue="Whole_Blood",
                population="EUR",
            )
    finally:
        sumstats_path.unlink()

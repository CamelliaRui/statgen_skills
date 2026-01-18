# tests/ldsc/test_integration.py
"""
Integration tests for LDSC pipeline.

These tests verify the full workflow from munging to analysis.
Note: Some tests require reference data and may be slow.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_sumstats():
    """Generate sample summary statistics."""
    np.random.seed(42)
    n_snps = 50

    return pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(1, n_snps + 1)],
        "CHR": [1] * n_snps,
        "BP": range(1000000, 1000000 + n_snps * 1000, 1000),
        "A1": ["A"] * n_snps,
        "A2": ["G"] * n_snps,
        "N": [50000] * n_snps,
        "BETA": np.random.normal(0, 0.05, n_snps),
        "SE": np.abs(np.random.normal(0.01, 0.002, n_snps)),
        "P": np.random.uniform(0, 1, n_snps),
    })


def test_munge_produces_valid_output(sample_sumstats):
    """Test that munging produces valid LDSC format."""
    from scripts.ldsc.munge import munge_sumstats

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.csv"
        sample_sumstats.to_csv(input_path, index=False)

        result = munge_sumstats(
            input_path=str(input_path),
            output_prefix=str(Path(tmpdir) / "munged"),
            verbose=False,
        )

        assert "output_path" in result
        assert Path(result["output_path"]).exists()

        # Verify output format
        munged = pd.read_csv(result["output_path"], sep="\t")
        assert "SNP" in munged.columns
        assert "Z" in munged.columns
        assert "N" in munged.columns


def test_full_pipeline_structure(sample_sumstats):
    """Test that all components integrate correctly."""
    from scripts.ldsc.munge import munge_sumstats
    from scripts.ldsc.parsers import parse_h2_log
    from visualization.ldsc_plots import create_h2_barplot
    import matplotlib.pyplot as plt

    # This tests the structure, not actual LDSC execution
    # (which requires reference data)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Munge
        input_path = Path(tmpdir) / "input.csv"
        sample_sumstats.to_csv(input_path, index=False)

        munge_result = munge_sumstats(
            input_path=str(input_path),
            output_prefix=str(Path(tmpdir) / "munged"),
            verbose=False,
        )

        assert Path(munge_result["output_path"]).exists()

        # Step 2: Parse a mock log
        mock_log = """
Total Observed scale h2: 0.25 (0.05)
Lambda GC: 1.02
Mean Chi^2: 1.05
Intercept: 1.001 (0.005)
"""
        parsed = parse_h2_log(mock_log)
        assert parsed["h2"] == pytest.approx(0.25)

        # Step 3: Visualize
        h2_df = pd.DataFrame({
            "trait": ["Test"],
            "h2": [parsed["h2"]],
            "h2_se": [parsed["h2_se"]],
        })
        fig = create_h2_barplot(h2_df)
        assert fig is not None
        plt.close(fig)


def test_munge_handles_missing_columns():
    """Test that munge raises appropriate errors for missing columns."""
    from scripts.ldsc.munge import munge_sumstats

    # Missing required columns
    df = pd.DataFrame({
        "SNP": ["rs1"],
        "A1": ["A"],
        # Missing A2
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.csv"
        df.to_csv(input_path, index=False)

        with pytest.raises(ValueError, match="Missing required"):
            munge_sumstats(
                input_path=str(input_path),
                output_prefix=str(Path(tmpdir) / "munged"),
            )


def test_visualization_pipeline():
    """Test full visualization pipeline."""
    from visualization.ldsc_plots import (
        create_h2_barplot,
        create_rg_heatmap,
        create_enrichment_plot,
    )
    import matplotlib.pyplot as plt

    # Test h2 barplot
    h2_df = pd.DataFrame({
        "trait": ["Height", "BMI"],
        "h2": [0.5, 0.3],
        "h2_se": [0.05, 0.03],
    })
    fig1 = create_h2_barplot(h2_df)
    assert fig1 is not None
    plt.close(fig1)

    # Test rg heatmap
    rg_matrix = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=["T1", "T2"],
        columns=["T1", "T2"],
    )
    fig2 = create_rg_heatmap(rg_matrix)
    assert fig2 is not None
    plt.close(fig2)

    # Test enrichment plot
    enrichment_df = pd.DataFrame({
        "category": ["Cat1", "Cat2"],
        "enrichment": [2.0, 0.5],
        "enrichment_se": [0.2, 0.1],
        "enrichment_p": [0.001, 0.1],
    })
    fig3 = create_enrichment_plot(enrichment_df)
    assert fig3 is not None
    plt.close(fig3)


def test_reference_data_paths():
    """Test reference data path generation."""
    from scripts.ldsc.reference_data import (
        get_reference_dir,
        get_population_paths,
        SUPPORTED_POPULATIONS,
    )

    # Test all supported populations
    for pop in SUPPORTED_POPULATIONS:
        paths = get_population_paths(pop)
        assert "ld_scores" in paths
        assert "weights" in paths
        assert "frq" in paths
        assert pop in str(paths["ld_scores"])


def test_parser_handles_edge_cases():
    """Test parsers handle edge cases gracefully."""
    from scripts.ldsc.parsers import parse_h2_log, parse_rg_log

    # Empty log
    result = parse_h2_log("")
    assert result["h2"] is None

    # Malformed log
    result = parse_h2_log("This is not a valid LDSC log")
    assert result["h2"] is None

    # Empty rg log
    result = parse_rg_log("")
    assert result["correlations"] == []


@pytest.mark.slow
def test_reference_data_download():
    """Test reference data download (slow, requires network)."""
    from scripts.ldsc.reference_data import download_reference, is_reference_available

    # This test actually downloads data - mark as slow
    # Skip if already downloaded to save time
    if not is_reference_available("EUR"):
        result = download_reference("EUR", verbose=True)
        assert result.exists()

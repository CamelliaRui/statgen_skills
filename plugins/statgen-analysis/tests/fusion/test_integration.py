# tests/fusion/test_integration.py
"""Integration tests for FUSION TWAS module."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path


def test_full_module_imports():
    """Test that all public API functions are importable."""
    from scripts.fusion import (
        run_twas_association,
        TWASResults,
        TWASResult,
        list_available_tissues,
        download_weights,
        download_ld_reference,
        check_dependencies,
        download_fusion,
    )

    assert callable(run_twas_association)
    assert callable(list_available_tissues)
    assert callable(check_dependencies)


def test_visualization_imports():
    """Test that visualization functions are importable."""
    from visualization import (
        create_fusion_locus_plot,
        create_tissue_heatmap,
    )

    assert callable(create_fusion_locus_plot)
    assert callable(create_tissue_heatmap)


def test_workflow_mock():
    """Test workflow with mocked data (no actual FUSION run)."""
    from scripts.fusion.parsers import TWASResult, results_to_dataframe
    from visualization.fusion_plots import create_tissue_heatmap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Create mock results
    results = [
        TWASResult(
            gene=f"GENE{i}",
            chromosome=1,
            start=1000 * i,
            end=1000 * i + 500,
            hsq=0.1,
            best_model="lasso",
            twas_z=float(i) - 2.5,
            twas_p=10 ** (-i),
            n_snps=100,
            n_weights=50,
            cv_r2=0.05,
            cv_pvalue=0.01,
            best_gwas_snp=f"rs{i}",
            best_gwas_z=2.0,
            eqtl_snp=f"rs{i}0",
            eqtl_r2=0.1,
            panel="GTEx",
            weight_file=f"gene{i}.wgt.RDat",
        )
        for i in range(1, 6)
    ]

    # Convert to DataFrame
    df = results_to_dataframe(results)
    assert len(df) == 5
    assert "gene" in df.columns
    assert "twas_z" in df.columns

    # Create multi-tissue mock for heatmap
    tissues = ["Whole_Blood", "Brain_Cortex", "Liver"]
    multi_tissue = pd.DataFrame({
        "gene": ["GENE1", "GENE2"] * 3,
        "tissue": tissues * 2,
        "twas_z": [2.5, 1.2, 3.1, 0.8, 1.9, 2.2],
    })

    fig = create_tissue_heatmap(multi_tissue)
    assert fig is not None
    plt.close(fig)


@pytest.mark.skipif(
    not pytest.importorskip("subprocess").run(
        ["which", "Rscript"], capture_output=True
    ).returncode == 0,
    reason="R not installed"
)
def test_dependency_check_with_r():
    """Test dependency check when R is available."""
    from scripts.fusion import check_dependencies

    deps = check_dependencies()
    assert deps["R"] is True

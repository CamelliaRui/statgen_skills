# tests/ldsc/test_munge.py
import pytest
import pandas as pd
from pathlib import Path
import tempfile


def test_munge_sumstats_validates_required_columns():
    from scripts.ldsc.munge import munge_sumstats

    # Missing required columns
    df = pd.DataFrame({"A": [1], "B": [2]})

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        with pytest.raises(ValueError, match="Missing required"):
            munge_sumstats(f.name, "/tmp/out")


def test_munge_sumstats_returns_output_path():
    from scripts.ldsc.munge import munge_sumstats

    # Valid sumstats
    df = pd.DataFrame({
        "SNP": ["rs1", "rs2"],
        "A1": ["A", "G"],
        "A2": ["G", "T"],
        "N": [10000, 10000],
        "P": [0.01, 0.5],
        "BETA": [0.1, -0.05],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "sumstats.csv"
        df.to_csv(input_path, index=False)

        output_prefix = Path(tmpdir) / "munged"
        result = munge_sumstats(str(input_path), str(output_prefix))

        assert "output_path" in result
        assert Path(result["output_path"]).exists()

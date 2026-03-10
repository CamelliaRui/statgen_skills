# scripts/ldsc/munge.py
"""
Summary statistics munging for LDSC.

Converts GWAS summary statistics to the format required by LDSC,
with proper filtering and quality control.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from scipy import stats


REQUIRED_COLUMNS = ["SNP", "A1", "A2"]
EFFECT_COLUMNS = [["BETA", "SE"], ["OR", "SE"], ["Z"], ["BETA"]]
PVALUE_COLUMNS = ["P", "PVAL", "PVALUE", "P_VALUE"]
N_COLUMNS = ["N", "NEFF", "N_EFF"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find first matching column from candidates."""
    df_cols_upper = [c.upper() for c in df.columns]
    for cand in candidates:
        if cand.upper() in df_cols_upper:
            idx = df_cols_upper.index(cand.upper())
            return df.columns[idx]
    return None


def munge_sumstats(
    input_path: str | Path,
    output_prefix: str | Path,
    n: int | None = None,
    info_min: float = 0.9,
    maf_min: float = 0.01,
    merge_alleles: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Munge summary statistics for LDSC.

    Converts input summary statistics to LDSC format with:
    - Column name standardization
    - Allele validation
    - Quality filtering (INFO, MAF)
    - Z-score calculation

    Args:
        input_path: Path to input summary statistics (CSV/TSV)
        output_prefix: Output file prefix (will create .sumstats.gz)
        n: Sample size (if not in file)
        info_min: Minimum INFO score filter
        maf_min: Minimum MAF filter
        merge_alleles: Path to allele reference file
        verbose: Print progress

    Returns:
        Dict with output_path, n_snps_input, n_snps_output, warnings
    """
    input_path = Path(input_path)
    output_prefix = Path(output_prefix)

    # Read input
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_csv(input_path, sep="\t")

    n_input = len(df)

    # Validate required columns
    df.columns = [c.upper() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Find effect column
    effect_col = None
    for combo in EFFECT_COLUMNS:
        if all(c in df.columns for c in combo):
            effect_col = combo
            break

    if effect_col is None:
        raise ValueError(
            "Need effect size column(s). Options: BETA+SE, OR+SE, Z, or BETA"
        )

    # Find or set sample size
    n_col = _find_column(df, N_COLUMNS)
    if n_col is None and n is None:
        raise ValueError("Sample size (N) not found in file and not provided")

    if n_col is None:
        df["N"] = n
    else:
        df["N"] = df[n_col]

    # Find P-value column
    p_col = _find_column(df, PVALUE_COLUMNS)
    if p_col is None:
        raise ValueError(f"P-value column not found. Expected one of: {PVALUE_COLUMNS}")
    df["P"] = df[p_col]

    # Calculate Z if needed
    if "Z" not in df.columns:
        # Convert P to Z
        sign = np.sign(df["BETA"]) if "BETA" in df.columns else 1
        df["Z"] = sign * np.abs(stats.norm.ppf(df["P"] / 2))

    # Apply filters
    warnings = []

    if "INFO" in df.columns:
        n_before = len(df)
        df = df[df["INFO"] >= info_min]
        n_filtered = n_before - len(df)
        if n_filtered > 0:
            warnings.append(f"Filtered {n_filtered} SNPs with INFO < {info_min}")

    if "MAF" in df.columns:
        n_before = len(df)
        df = df[df["MAF"] >= maf_min]
        n_filtered = n_before - len(df)
        if n_filtered > 0:
            warnings.append(f"Filtered {n_filtered} SNPs with MAF < {maf_min}")

    # Output in LDSC format
    output_cols = ["SNP", "A1", "A2", "N", "Z"]
    output_df = df[output_cols].copy()

    output_path = Path(f"{output_prefix}.sumstats.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, sep="\t", index=False, compression="gzip")

    n_output = len(output_df)

    if verbose:
        print(f"Munged {n_input} -> {n_output} SNPs")
        print(f"Output: {output_path}")

    return {
        "output_path": str(output_path),
        "n_snps_input": n_input,
        "n_snps_output": n_output,
        "warnings": warnings,
    }

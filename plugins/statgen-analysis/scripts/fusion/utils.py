# scripts/fusion/utils.py
"""
Utility functions for FUSION TWAS.

Handles:
- Dependency checking (R, PLINK, FUSION)
- FUSION installation from GitHub
- Summary statistics validation and formatting
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def get_fusion_dir() -> Path:
    """
    Return the directory where FUSION TWAS is installed.

    Returns:
        Path to ~/.statgen_skills/fusion_twas
    """
    return Path.home() / ".statgen_skills" / "fusion_twas"


def check_r_installed() -> bool:
    """
    Check if Rscript is available in the system PATH.

    Returns:
        True if Rscript is available, False otherwise
    """
    return shutil.which("Rscript") is not None


def check_plink_installed() -> bool:
    """
    Check if PLINK is available in the system PATH.

    Returns:
        True if plink is available, False otherwise
    """
    return shutil.which("plink") is not None


def check_fusion_installed() -> bool:
    """
    Check if FUSION scripts exist in the FUSION directory.

    Returns:
        True if FUSION.assoc_test.R exists, False otherwise
    """
    fusion_dir = get_fusion_dir()
    fusion_script = fusion_dir / "FUSION.assoc_test.R"
    return fusion_script.exists()


def check_dependencies() -> dict[str, bool]:
    """
    Check all dependencies required for running FUSION.

    Returns:
        Dict with keys 'R', 'FUSION', 'PLINK' and boolean values
    """
    return {
        "R": check_r_installed(),
        "FUSION": check_fusion_installed(),
        "PLINK": check_plink_installed(),
    }


def download_fusion(force: bool = False, verbose: bool = True) -> Path:
    """
    Clone FUSION from GitHub into the FUSION directory.

    Args:
        force: Re-download even if FUSION already exists
        verbose: Print progress messages

    Returns:
        Path to the FUSION installation directory

    Raises:
        RuntimeError: If git clone fails
    """
    fusion_dir = get_fusion_dir()

    if check_fusion_installed() and not force:
        if verbose:
            print(f"FUSION already installed at {fusion_dir}")
        return fusion_dir

    # Create parent directory
    fusion_dir.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing directory if force
    if fusion_dir.exists() and force:
        shutil.rmtree(fusion_dir)

    if verbose:
        print("Cloning FUSION from GitHub...")

    # Clone FUSION repository
    try:
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/gusevlab/fusion_twas.git",
                str(fusion_dir)
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone FUSION: {e.stderr}")

    if verbose:
        print(f"FUSION installed at {fusion_dir}")

    return fusion_dir


def validate_sumstats_columns(columns: list[str]) -> bool:
    """
    Validate that summary statistics have required columns for FUSION.

    FUSION requires:
    - SNP: SNP identifier
    - A1: Effect allele
    - A2: Other allele
    - Z: Z-score (or BETA + SE to compute Z)

    Args:
        columns: List of column names in the summary statistics

    Returns:
        True if required columns are present, False otherwise
    """
    columns_upper = [c.upper() for c in columns]

    # Required columns
    required = ["SNP", "A1", "A2"]
    for col in required:
        if col not in columns_upper:
            return False

    # Need either Z or (BETA and SE)
    has_z = "Z" in columns_upper
    has_beta_se = "BETA" in columns_upper and "SE" in columns_upper

    return has_z or has_beta_se


def format_sumstats_for_fusion(
    sumstats: Any,
    snp_col: str = "SNP",
    a1_col: str = "A1",
    a2_col: str = "A2",
    z_col: str | None = "Z",
    beta_col: str | None = None,
    se_col: str | None = None,
    output_path: Path | str | None = None,
) -> Any:
    """
    Format summary statistics for FUSION input.

    FUSION expects a tab-separated file with columns:
    SNP, A1, A2, Z

    Args:
        sumstats: pandas DataFrame with summary statistics
        snp_col: Column name for SNP identifiers
        a1_col: Column name for effect allele
        a2_col: Column name for other allele
        z_col: Column name for Z-score (if available)
        beta_col: Column name for effect size (used with se_col to compute Z)
        se_col: Column name for standard error (used with beta_col to compute Z)
        output_path: Optional path to write formatted file

    Returns:
        pandas DataFrame formatted for FUSION

    Raises:
        ImportError: If pandas is not installed
        ValueError: If required columns are missing or Z cannot be computed
    """
    if not HAS_PANDAS:
        raise ImportError(
            "pandas required for formatting summary statistics. "
            "Install with: pip install pandas"
        )

    # Create output DataFrame
    formatted = pd.DataFrame()
    formatted["SNP"] = sumstats[snp_col]
    formatted["A1"] = sumstats[a1_col]
    formatted["A2"] = sumstats[a2_col]

    # Get or compute Z-score
    if z_col is not None and z_col in sumstats.columns:
        formatted["Z"] = sumstats[z_col]
    elif beta_col is not None and se_col is not None:
        if beta_col not in sumstats.columns or se_col not in sumstats.columns:
            raise ValueError(
                f"Columns {beta_col} and {se_col} required to compute Z-score"
            )
        formatted["Z"] = sumstats[beta_col] / sumstats[se_col]
    else:
        raise ValueError(
            "Must provide either z_col or both beta_col and se_col"
        )

    # Write to file if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        formatted.to_csv(output_path, sep="\t", index=False)

    return formatted

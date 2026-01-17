"""
validate_input.py - Input validation utilities for statgen-skills

Validates summary statistics, LD matrices, and individual-level data
before running SuSiE fine-mapping.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


def validate_summary_stats(filepath: str | Path) -> dict:
    """
    Validate GWAS summary statistics file.

    Args:
        filepath: Path to summary statistics CSV/TSV file

    Returns:
        dict with validation results and data info

    Raises:
        ValidationError: If validation fails
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise ValidationError(f"File not found: {filepath}")

    # Detect separator
    with open(filepath) as f:
        first_line = f.readline()
        sep = "\t" if "\t" in first_line else ","

    # Load data
    try:
        df = pd.read_csv(filepath, sep=sep)
    except Exception as e:
        raise ValidationError(f"Failed to read file: {e}")

    # Standardize column names
    df.columns = df.columns.str.upper()

    # Check required columns
    required = ["SNP", "CHR", "BP"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {', '.join(missing)}")

    # Check for effect size columns
    has_beta_se = "BETA" in df.columns and "SE" in df.columns
    has_z = "Z" in df.columns

    if not has_beta_se and not has_z:
        raise ValidationError(
            "Summary stats must have either BETA+SE or Z columns. "
            f"Found columns: {', '.join(df.columns)}"
        )

    # Check for missing values in key columns
    for col in required:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            raise ValidationError(f"Column '{col}' has {n_missing} missing values")

    # Check effect size column for missing values
    effect_col = "Z" if has_z else "BETA"
    n_missing = df[effect_col].isna().sum()
    if n_missing > 0:
        raise ValidationError(f"Column '{effect_col}' has {n_missing} missing values")

    # Validate effect sizes are reasonable
    if has_beta_se:
        max_beta = df["BETA"].abs().max()
        if max_beta > 10:
            print(
                f"Warning: Large effect sizes detected (max |beta| = {max_beta:.2f}). "
                "Verify your data is correct.",
                file=sys.stderr,
            )

    # Check for duplicate variants
    n_duplicates = df.duplicated(subset=["CHR", "BP"]).sum()
    if n_duplicates > 0:
        raise ValidationError(
            f"Found {n_duplicates} duplicate variants (same CHR:BP). "
            "Please deduplicate your summary statistics."
        )

    return {
        "valid": True,
        "n_variants": len(df),
        "columns": list(df.columns),
        "has_beta_se": has_beta_se,
        "has_z": has_z,
        "chromosomes": sorted(df["CHR"].unique().tolist()),
    }


def validate_ld_matrix(filepath: str | Path, n_variants: int | None = None) -> dict:
    """
    Validate LD matrix file.

    Args:
        filepath: Path to LD matrix (.npy, .txt, or .rds)
        n_variants: Expected number of variants (optional)

    Returns:
        dict with validation results

    Raises:
        ValidationError: If validation fails
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise ValidationError(f"LD matrix file not found: {filepath}")

    # Load based on extension
    ext = filepath.suffix.lower()

    try:
        if ext == ".npy":
            R = np.load(filepath)
        elif ext in [".txt", ".csv", ".tsv"]:
            R = np.loadtxt(filepath)
        elif ext == ".rds":
            raise ValidationError(
                "RDS files must be loaded in R. "
                "Convert to .npy or text format for Python validation."
            )
        else:
            # Try loading as text
            R = np.loadtxt(filepath)
    except Exception as e:
        raise ValidationError(f"Failed to load LD matrix: {e}")

    # Check square
    if R.ndim != 2:
        raise ValidationError(f"LD matrix must be 2D, got {R.ndim}D array")

    if R.shape[0] != R.shape[1]:
        raise ValidationError(
            f"LD matrix must be square. Got shape {R.shape[0]} x {R.shape[1]}"
        )

    # Check dimension matches variants
    if n_variants is not None and R.shape[0] != n_variants:
        raise ValidationError(
            f"LD matrix dimension ({R.shape[0]}) does not match "
            f"number of variants ({n_variants})"
        )

    # Check diagonal is approximately 1
    diag = np.diag(R)
    if not np.allclose(diag, 1.0, atol=0.01):
        diag_issues = np.sum(np.abs(diag - 1.0) > 0.01)
        raise ValidationError(
            f"LD matrix diagonal should be 1.0. "
            f"Found {diag_issues} entries deviating by >0.01"
        )

    # Check symmetric
    if not np.allclose(R, R.T, atol=1e-6):
        raise ValidationError("LD matrix is not symmetric")

    # Check positive semi-definite (eigenvalues >= 0)
    try:
        eigenvalues = np.linalg.eigvalsh(R)
        min_eigenvalue = eigenvalues.min()
        if min_eigenvalue < -0.01:
            print(
                f"Warning: LD matrix has negative eigenvalue ({min_eigenvalue:.4f}). "
                "Matrix may not be positive semi-definite.",
                file=sys.stderr,
            )
    except np.linalg.LinAlgError:
        print("Warning: Could not compute eigenvalues to check PSD", file=sys.stderr)

    return {
        "valid": True,
        "shape": R.shape,
        "n_variants": R.shape[0],
        "min_value": float(R.min()),
        "max_value": float(R.max()),
    }


def validate_plink_files(prefix: str | Path) -> dict:
    """
    Validate PLINK binary files exist.

    Args:
        prefix: PLINK file prefix (without .bed/.bim/.fam extension)

    Returns:
        dict with validation results

    Raises:
        ValidationError: If validation fails
    """
    prefix = Path(prefix)

    required_extensions = [".bed", ".bim", ".fam"]
    missing = []

    for ext in required_extensions:
        filepath = prefix.with_suffix(ext)
        if not filepath.exists():
            # Also check with extension appended
            filepath = Path(str(prefix) + ext)
            if not filepath.exists():
                missing.append(ext)

    if missing:
        raise ValidationError(
            f"Missing PLINK files: {', '.join(missing)}. "
            f"Expected files with prefix: {prefix}"
        )

    # Read .bim to get variant count
    bim_path = prefix.with_suffix(".bim")
    if not bim_path.exists():
        bim_path = Path(str(prefix) + ".bim")

    try:
        bim = pd.read_csv(
            bim_path, sep="\t", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"]
        )
        n_variants = len(bim)
    except Exception as e:
        raise ValidationError(f"Failed to read .bim file: {e}")

    # Read .fam to get sample count
    fam_path = prefix.with_suffix(".fam")
    if not fam_path.exists():
        fam_path = Path(str(prefix) + ".fam")

    try:
        fam = pd.read_csv(
            fam_path,
            sep=r"\s+",
            header=None,
            names=["FID", "IID", "PID", "MID", "SEX", "PHENO"],
        )
        n_samples = len(fam)
    except Exception as e:
        raise ValidationError(f"Failed to read .fam file: {e}")

    return {
        "valid": True,
        "n_variants": n_variants,
        "n_samples": n_samples,
        "chromosomes": sorted(bim["CHR"].unique().tolist()),
    }


def validate_phenotype_file(
    filepath: str | Path, sample_ids: list[str] | None = None
) -> dict:
    """
    Validate phenotype file.

    Args:
        filepath: Path to phenotype file
        sample_ids: Expected sample IDs (optional, for matching check)

    Returns:
        dict with validation results

    Raises:
        ValidationError: If validation fails
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise ValidationError(f"Phenotype file not found: {filepath}")

    try:
        # Try tab-delimited first, then whitespace
        try:
            df = pd.read_csv(filepath, sep="\t")
        except Exception:
            df = pd.read_csv(filepath, sep=r"\s+")
    except Exception as e:
        raise ValidationError(f"Failed to read phenotype file: {e}")

    # Standardize column names
    df.columns = df.columns.str.upper()

    # Check for required columns
    if "IID" not in df.columns:
        # Try to infer - second column is often IID
        if len(df.columns) >= 2:
            df.columns = ["FID", "IID"] + list(df.columns[2:])
        else:
            raise ValidationError(
                "Phenotype file must have IID column or be in FID IID PHENO format"
            )

    # Check for phenotype column
    pheno_cols = [col for col in df.columns if col not in ["FID", "IID"]]
    if len(pheno_cols) == 0:
        raise ValidationError("Phenotype file must have at least one phenotype column")

    # Check for matching sample IDs
    if sample_ids is not None:
        file_ids = set(df["IID"].astype(str))
        expected_ids = set(str(x) for x in sample_ids)

        missing = expected_ids - file_ids
        if len(missing) > 0:
            raise ValidationError(
                f"{len(missing)} samples in genotype data not found in phenotype file. "
                f"Examples: {list(missing)[:5]}"
            )

    # Check for missing phenotype values
    for col in pheno_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(
                f"Warning: Phenotype '{col}' has {n_missing} missing values",
                file=sys.stderr,
            )

    return {
        "valid": True,
        "n_samples": len(df),
        "phenotype_columns": pheno_cols,
        "has_fid": "FID" in df.columns,
    }


def main():
    """CLI entry point for testing validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate input files for SuSiE")
    parser.add_argument("--sumstats", help="Summary statistics file")
    parser.add_argument("--ld", help="LD matrix file")
    parser.add_argument("--plink", help="PLINK file prefix")
    parser.add_argument("--pheno", help="Phenotype file")

    args = parser.parse_args()

    if args.sumstats:
        try:
            result = validate_summary_stats(args.sumstats)
            print(f"Summary stats valid: {result}")
        except ValidationError as e:
            print(f"Validation failed: {e}", file=sys.stderr)
            sys.exit(1)

    if args.ld:
        try:
            result = validate_ld_matrix(args.ld)
            print(f"LD matrix valid: {result}")
        except ValidationError as e:
            print(f"Validation failed: {e}", file=sys.stderr)
            sys.exit(1)

    if args.plink:
        try:
            result = validate_plink_files(args.plink)
            print(f"PLINK files valid: {result}")
        except ValidationError as e:
            print(f"Validation failed: {e}", file=sys.stderr)
            sys.exit(1)

    if args.pheno:
        try:
            result = validate_phenotype_file(args.pheno)
            print(f"Phenotype file valid: {result}")
        except ValidationError as e:
            print(f"Validation failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

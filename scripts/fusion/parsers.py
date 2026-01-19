# scripts/fusion/parsers.py
"""
Parsers for FUSION TWAS output files.

Converts FUSION output to Python dataclasses and DataFrames.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


@dataclass
class TWASResult:
    """Single gene TWAS result."""
    gene: str
    chromosome: int
    start: int
    end: int
    hsq: float
    best_model: str
    twas_z: float
    twas_p: float
    n_snps: int
    n_weights: int
    cv_r2: float
    cv_pvalue: float
    best_gwas_snp: str
    best_gwas_z: float
    eqtl_snp: str
    eqtl_r2: float
    panel: str
    weight_file: str


def parse_twas_results(filepath: str | Path) -> list[TWASResult]:
    """Parse FUSION association test output file."""
    df = pd.read_csv(filepath, sep="\t")

    results = []
    for _, row in df.iterrows():
        result = TWASResult(
            gene=row["ID"],
            chromosome=int(row["CHR"]),
            start=int(row["P0"]),
            end=int(row["P1"]),
            hsq=float(row["HSQ"]) if pd.notna(row["HSQ"]) else 0.0,
            best_model=str(row["MODEL"]).lower(),
            twas_z=float(row["TWAS.Z"]),
            twas_p=float(row["TWAS.P"]),
            n_snps=int(row["NSNP"]),
            n_weights=int(row["NWGT"]),
            cv_r2=float(row["MODELCV.R2"]) if pd.notna(row["MODELCV.R2"]) else 0.0,
            cv_pvalue=float(row["MODELCV.PV"]) if pd.notna(row["MODELCV.PV"]) else 1.0,
            best_gwas_snp=str(row["BEST.GWAS.ID"]),
            best_gwas_z=float(row["BEST.GWAS.Z"]) if pd.notna(row["BEST.GWAS.Z"]) else 0.0,
            eqtl_snp=str(row["EQTL.ID"]),
            eqtl_r2=float(row["EQTL.R2"]) if pd.notna(row["EQTL.R2"]) else 0.0,
            panel=str(row["PANEL"]),
            weight_file=str(row["FILE"]),
        )
        results.append(result)

    return results


def results_to_dataframe(results: list[TWASResult]) -> pd.DataFrame:
    """Convert list of TWASResult to pandas DataFrame."""
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([asdict(r) for r in results])


def get_significant_results(
    results: list[TWASResult],
    threshold: float = 2.5e-6,
) -> list[TWASResult]:
    """Filter results by significance threshold."""
    return [r for r in results if r.twas_p < threshold]


def save_results(
    results: list[TWASResult],
    output_path: str | Path,
    significant_only: bool = False,
    threshold: float = 2.5e-6,
) -> Path:
    """Save results to CSV file."""
    if significant_only:
        results = get_significant_results(results, threshold)
    df = results_to_dataframe(results)
    df.to_csv(output_path, index=False)
    return Path(output_path)

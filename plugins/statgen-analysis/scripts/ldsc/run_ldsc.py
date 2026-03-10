# scripts/ldsc/run_ldsc.py
"""
Main LDSC analysis runner.

Provides high-level API for:
- estimate_heritability(): SNP h2 from GWAS summary stats
- genetic_correlation(): rg between traits
- partitioned_heritability(): s-LDSC by annotations
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .parsers import parse_h2_log, parse_rg_log, parse_partitioned_h2_log
from .reference_data import ensure_reference, get_population_paths


def _find_ldsc_executable() -> str:
    """Find the ldsc.py executable."""
    # Try to find ldsc.py in PATH
    ldsc_path = shutil.which("ldsc.py")
    if ldsc_path:
        return ldsc_path
    
    # Fallback: try common locations
    import site
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        candidate = Path(site_dir).parent.parent / "bin" / "ldsc.py"
        if candidate.exists():
            return str(candidate)
    
    raise RuntimeError(
        "ldsc.py not found. Install with: pip install git+https://github.com/CBIIT/ldsc.git"
    )


def _build_output_json(
    success: bool,
    analysis_type: str,
    results: dict[str, Any] | None = None,
    files: dict[str, str] | None = None,
    error: str | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build standardized JSON output structure."""
    return {
        "success": success,
        "analysis_type": analysis_type,
        "timestamp": datetime.now().isoformat(),
        "results": results or {},
        "files": files or {},
        "error": error,
        "warnings": warnings or [],
    }


def estimate_heritability(
    sumstats: str | Path,
    output_dir: str | Path,
    population: str = "EUR",
    sample_prevalence: float | None = None,
    population_prevalence: float | None = None,
    no_intercept: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Estimate SNP heritability using LDSC.

    Args:
        sumstats: Path to munged summary statistics (.sumstats.gz)
        output_dir: Directory for output files
        population: Reference population for LD scores
        sample_prevalence: For case-control, sample prevalence
        population_prevalence: For case-control, population prevalence
        no_intercept: Constrain intercept to 1
        verbose: Print progress

    Returns:
        Dict with h2, h2_se, lambda_gc, intercept, and file paths
    """
    sumstats = Path(sumstats)
    output_dir = Path(output_dir)

    if not sumstats.exists():
        raise FileNotFoundError(f"Summary stats not found: {sumstats}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure reference data available
    ref_paths = ensure_reference(population, verbose=verbose)

    # Find ldsc.py
    ldsc_exe = _find_ldsc_executable()

    # Build LDSC command
    output_prefix = output_dir / f"h2_{sumstats.stem}"

    cmd = [
        ldsc_exe,
        "--h2", str(sumstats),
        "--ref-ld-chr", str(ref_paths["ld_scores"]) + "/",
        "--w-ld-chr", str(ref_paths["weights"]) + "/",
        "--out", str(output_prefix),
    ]

    if sample_prevalence is not None and population_prevalence is not None:
        cmd.extend([
            "--samp-prev", str(sample_prevalence),
            "--pop-prev", str(population_prevalence),
        ])

    if no_intercept:
        cmd.append("--no-intercept")

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    # Execute
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return _build_output_json(
            success=False,
            analysis_type="h2",
            error=f"LDSC failed: {e.stderr}",
        )

    # Parse results
    log_path = Path(f"{output_prefix}.log")
    if log_path.exists():
        log_content = log_path.read_text()
        parsed = parse_h2_log(log_content)
    else:
        parsed = {}

    return _build_output_json(
        success=True,
        analysis_type="h2",
        results=parsed,
        files={
            "log": str(log_path),
        },
    )


def genetic_correlation(
    sumstats: list[str | Path],
    output_dir: str | Path,
    population: str = "EUR",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Estimate genetic correlations between traits.

    Args:
        sumstats: List of paths to munged summary statistics (at least 2)
        output_dir: Directory for output files
        population: Reference population for LD scores
        verbose: Print progress

    Returns:
        Dict with correlations matrix and file paths
    """
    if len(sumstats) < 2:
        raise ValueError("Need at least 2 traits for genetic correlation")

    sumstats = [Path(s) for s in sumstats]
    output_dir = Path(output_dir)

    for ss in sumstats:
        if not ss.exists():
            raise FileNotFoundError(f"Summary stats not found: {ss}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure reference data
    ref_paths = ensure_reference(population, verbose=verbose)

    # Find ldsc.py
    ldsc_exe = _find_ldsc_executable()

    # Build command
    output_prefix = output_dir / "rg"
    sumstats_str = ",".join(str(s) for s in sumstats)

    cmd = [
        ldsc_exe,
        "--rg", sumstats_str,
        "--ref-ld-chr", str(ref_paths["ld_scores"]) + "/",
        "--w-ld-chr", str(ref_paths["weights"]) + "/",
        "--out", str(output_prefix),
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    # Execute
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return _build_output_json(
            success=False,
            analysis_type="rg",
            error=f"LDSC failed: {e.stderr}",
        )

    # Parse results
    log_path = Path(f"{output_prefix}.log")
    if log_path.exists():
        log_content = log_path.read_text()
        parsed = parse_rg_log(log_content)
    else:
        parsed = {}

    return _build_output_json(
        success=True,
        analysis_type="rg",
        results=parsed,
        files={
            "log": str(log_path),
        },
    )


def partitioned_heritability(
    sumstats: str | Path,
    output_dir: str | Path,
    annotations: str | Path,
    population: str = "EUR",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Partition heritability by functional annotations (s-LDSC).

    Args:
        sumstats: Path to munged summary statistics
        output_dir: Directory for output files
        annotations: Path to annotation LD scores (prefix)
        population: Reference population
        verbose: Print progress

    Returns:
        Dict with enrichments by category and file paths
    """
    sumstats = Path(sumstats)
    output_dir = Path(output_dir)
    annotations = Path(annotations)

    if not sumstats.exists():
        raise FileNotFoundError(f"Summary stats not found: {sumstats}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure reference data
    ref_paths = ensure_reference(population, verbose=verbose)

    # Find ldsc.py
    ldsc_exe = _find_ldsc_executable()

    # Build command
    output_prefix = output_dir / f"sldsc_{sumstats.stem}"

    cmd = [
        ldsc_exe,
        "--h2", str(sumstats),
        "--ref-ld-chr", str(annotations) + "/",
        "--w-ld-chr", str(ref_paths["weights"]) + "/",
        "--overlap-annot",
        "--frqfile-chr", str(ref_paths["frq"]) + "/",
        "--out", str(output_prefix),
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    # Execute
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return _build_output_json(
            success=False,
            analysis_type="sldsc",
            error=f"LDSC failed: {e.stderr}",
        )

    # Parse results
    log_path = Path(f"{output_prefix}.log")
    results_path = Path(f"{output_prefix}.results")

    parsed = {}
    if log_path.exists():
        log_content = log_path.read_text()
        parsed = parse_partitioned_h2_log(log_content)

    return _build_output_json(
        success=True,
        analysis_type="sldsc",
        results=parsed,
        files={
            "log": str(log_path),
            "results": str(results_path) if results_path.exists() else None,
        },
    )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LDSC analyses")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # h2 subcommand
    h2_parser = subparsers.add_parser("h2", help="Estimate heritability")
    h2_parser.add_argument("--sumstats", required=True, help="Munged sumstats")
    h2_parser.add_argument("--output", required=True, help="Output directory")
    h2_parser.add_argument("--pop", default="EUR", help="Population")
    h2_parser.add_argument("--samp-prev", type=float, help="Sample prevalence")
    h2_parser.add_argument("--pop-prev", type=float, help="Population prevalence")

    # rg subcommand
    rg_parser = subparsers.add_parser("rg", help="Genetic correlation")
    rg_parser.add_argument("--sumstats", nargs="+", required=True)
    rg_parser.add_argument("--output", required=True, help="Output directory")
    rg_parser.add_argument("--pop", default="EUR", help="Population")

    # sldsc subcommand
    sldsc_parser = subparsers.add_parser("sldsc", help="Partitioned h2")
    sldsc_parser.add_argument("--sumstats", required=True)
    sldsc_parser.add_argument("--output", required=True)
    sldsc_parser.add_argument("--annot", required=True, help="Annotation prefix")
    sldsc_parser.add_argument("--pop", default="EUR")

    args = parser.parse_args()

    if args.command == "h2":
        result = estimate_heritability(
            sumstats=args.sumstats,
            output_dir=args.output,
            population=args.pop,
            sample_prevalence=args.samp_prev,
            population_prevalence=args.pop_prev,
        )
    elif args.command == "rg":
        result = genetic_correlation(
            sumstats=args.sumstats,
            output_dir=args.output,
            population=args.pop,
        )
    elif args.command == "sldsc":
        result = partitioned_heritability(
            sumstats=args.sumstats,
            output_dir=args.output,
            annotations=args.annot,
            population=args.pop,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

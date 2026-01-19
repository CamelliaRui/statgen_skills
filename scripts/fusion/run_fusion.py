# scripts/fusion/run_fusion.py
"""
Main FUSION TWAS runner.

Provides the primary entry point for running FUSION transcriptome-wide
association studies using pre-computed gene expression weights.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .utils import (
    get_fusion_dir,
    check_r_installed,
    check_fusion_installed,
    download_fusion,
    format_sumstats_for_fusion,
)
from .reference_data import (
    validate_tissue_name,
    validate_population,
    ensure_weights,
    ensure_ld_reference,
    get_weights_dir,
    get_ld_reference_dir,
)
from .parsers import (
    TWASResult,
    parse_twas_results,
    get_significant_results,
    save_results,
)


@dataclass
class TWASResults:
    """Container for TWAS analysis results.

    Attributes:
        all_results: List of all TWASResult objects from the analysis
        significant: List of significant TWASResult objects (p < threshold)
        output_dir: Directory where output files were saved
        tissue: GTEx tissue name used for the analysis
        n_genes_tested: Total number of genes tested across all chromosomes
    """
    all_results: list[TWASResult]
    significant: list[TWASResult]
    output_dir: Path
    tissue: str
    n_genes_tested: int


def _validate_inputs(
    sumstats: Path | str,
    tissue: str,
    population: str,
) -> Path:
    """
    Validate inputs for TWAS analysis.

    Args:
        sumstats: Path to summary statistics file
        tissue: GTEx v8 tissue name
        population: Population code for LD reference

    Returns:
        Path to validated sumstats file

    Raises:
        FileNotFoundError: If sumstats file does not exist
        ValueError: If tissue or population is invalid
    """
    sumstats_path = Path(sumstats)
    if not sumstats_path.exists():
        raise FileNotFoundError(f"Summary statistics file not found: {sumstats_path}")

    # These will raise ValueError if invalid
    validate_tissue_name(tissue)
    validate_population(population)

    return sumstats_path


def _ensure_dependencies(verbose: bool = True) -> Path:
    """
    Ensure R and FUSION are installed.

    Args:
        verbose: Print progress messages

    Returns:
        Path to FUSION installation directory

    Raises:
        RuntimeError: If R is not installed
    """
    if not check_r_installed():
        raise RuntimeError(
            "Rscript not found. Please install R: https://cran.r-project.org/"
        )

    if not check_fusion_installed():
        if verbose:
            print("FUSION not found, downloading...")
        download_fusion(verbose=verbose)

    return get_fusion_dir()


def build_fusion_command(
    sumstats: Path,
    weights_pos: Path,
    weights_dir: Path,
    ld_ref: Path,
    chromosome: int,
    output: Path,
    gwas_n: int | None = None,
    coloc: bool = False,
) -> list[str]:
    """
    Build the FUSION Rscript command.

    Args:
        sumstats: Path to formatted summary statistics file
        weights_pos: Path to weights .pos file listing all weight files
        weights_dir: Directory containing weight files
        ld_ref: Path to LD reference panel prefix (without .bed/.bim/.fam)
        chromosome: Chromosome number to analyze
        output: Path for output file
        gwas_n: Sample size of GWAS (optional, for COLOC)
        coloc: Whether to run colocalization analysis

    Returns:
        List of command arguments for subprocess
    """
    fusion_dir = get_fusion_dir()
    fusion_script = fusion_dir / "FUSION.assoc_test.R"

    cmd = [
        "Rscript",
        str(fusion_script),
        "--sumstats", str(sumstats),
        "--weights", str(weights_pos),
        "--weights_dir", str(weights_dir),
        "--ref_ld_chr", str(ld_ref),
        "--chr", str(chromosome),
        "--out", str(output),
    ]

    if gwas_n is not None:
        cmd.extend(["--GWASN", str(gwas_n)])

    if coloc:
        cmd.append("--coloc_P")

    return cmd


def run_twas_association(
    sumstats: Path | str,
    tissue: str,
    output_dir: Path | str,
    population: str = "EUR",
    chromosomes: list[int] | None = None,
    gwas_n: int | None = None,
    coloc: bool = False,
    verbose: bool = True,
) -> TWASResults:
    """
    Run FUSION TWAS association analysis.

    This is the main entry point for running FUSION. It:
    1. Validates inputs
    2. Ensures dependencies (R, FUSION) are installed
    3. Downloads weights and LD reference if needed
    4. Formats summary statistics for FUSION
    5. Runs FUSION for each chromosome
    6. Parses and aggregates results
    7. Saves results to CSV

    Args:
        sumstats: Path to summary statistics file (must have SNP, A1, A2, Z columns)
        tissue: GTEx v8 tissue name (e.g., "Whole_Blood", "Brain_Cortex")
        output_dir: Directory to save output files
        population: Population for LD reference (EUR, EAS, AFR). Default: EUR
        chromosomes: List of chromosomes to analyze. Default: 1-22
        gwas_n: GWAS sample size (required for COLOC)
        coloc: Whether to run colocalization analysis
        verbose: Print progress messages

    Returns:
        TWASResults container with all results and metadata

    Raises:
        FileNotFoundError: If sumstats file does not exist
        ValueError: If tissue or population is invalid
        RuntimeError: If R is not installed or FUSION fails
    """
    # Step 1: Validate inputs
    sumstats_path = _validate_inputs(sumstats, tissue, population)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if chromosomes is None:
        chromosomes = list(range(1, 23))

    if verbose:
        print(f"Running TWAS for tissue: {tissue}")
        print(f"Population: {population}")
        print(f"Chromosomes: {chromosomes}")

    # Step 2: Ensure dependencies
    fusion_dir = _ensure_dependencies(verbose=verbose)

    # Step 3: Download weights and LD reference if needed
    if verbose:
        print(f"Ensuring weights for {tissue}...")
    weights_dir = ensure_weights(tissue, verbose=verbose)
    weights_pos = weights_dir / f"GTEx.{tissue}.pos"

    if verbose:
        print(f"Ensuring LD reference for {population}...")
    ld_ref_dir = ensure_ld_reference(population, verbose=verbose)
    ld_ref_prefix = ld_ref_dir / f"1000G.{population}."

    # Step 4: Format summary statistics for FUSION
    if verbose:
        print("Formatting summary statistics...")

    import pandas as pd
    sumstats_df = pd.read_csv(sumstats_path, sep="\t")
    formatted_sumstats_path = output_path / "sumstats_formatted.txt"
    format_sumstats_for_fusion(sumstats_df, output_path=formatted_sumstats_path)

    # Step 5: Run FUSION for each chromosome
    all_results: list[TWASResult] = []
    log_dir = output_path / "logs"
    log_dir.mkdir(exist_ok=True)

    for chrom in chromosomes:
        if verbose:
            print(f"Processing chromosome {chrom}...")

        chrom_output = output_path / f"chr{chrom}.dat"
        log_file = log_dir / f"chr{chrom}.log"

        cmd = build_fusion_command(
            sumstats=formatted_sumstats_path,
            weights_pos=weights_pos,
            weights_dir=weights_dir,
            ld_ref=ld_ref_prefix,
            chromosome=chrom,
            output=chrom_output,
            gwas_n=gwas_n,
            coloc=coloc,
        )

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            # Save log
            with open(log_file, "w") as f:
                f.write(f"COMMAND: {' '.join(cmd)}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)

            # Parse results if output file was created
            if chrom_output.exists():
                chrom_results = parse_twas_results(chrom_output)
                all_results.extend(chrom_results)
                if verbose:
                    print(f"  Chromosome {chrom}: {len(chrom_results)} genes tested")

        except subprocess.CalledProcessError as e:
            # Log the error but continue with other chromosomes
            with open(log_file, "w") as f:
                f.write(f"COMMAND: {' '.join(cmd)}\n\n")
                f.write("ERROR:\n")
                f.write(e.stderr if e.stderr else str(e))

            if verbose:
                print(f"  Warning: Chromosome {chrom} failed: {e.stderr[:200] if e.stderr else str(e)}")

    # Step 6: Get significant results
    significant = get_significant_results(all_results)

    # Step 7: Save results to CSV
    if all_results:
        all_results_path = output_path / "twas_results_all.csv"
        save_results(all_results, all_results_path)
        if verbose:
            print(f"All results saved to: {all_results_path}")

        if significant:
            sig_results_path = output_path / "twas_results_significant.csv"
            save_results(significant, sig_results_path)
            if verbose:
                print(f"Significant results saved to: {sig_results_path}")

    if verbose:
        print(f"\nTWAS complete!")
        print(f"  Total genes tested: {len(all_results)}")
        print(f"  Significant associations: {len(significant)}")

    return TWASResults(
        all_results=all_results,
        significant=significant,
        output_dir=output_path,
        tissue=tissue,
        n_genes_tested=len(all_results),
    )

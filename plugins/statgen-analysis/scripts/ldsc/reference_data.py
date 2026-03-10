# scripts/ldsc/reference_data.py
"""
Reference data management for LDSC analyses.

Handles downloading and caching of LD score reference files for
1000 Genomes populations: EUR, EAS, AFR, SAS, AMR.
"""

import tarfile
import urllib.request
import shutil
from pathlib import Path
from typing import TypedDict

SUPPORTED_POPULATIONS = ["EUR", "EAS", "AFR", "SAS", "AMR"]

# 1000G Phase 3 LD scores hosted by the Broad Institute
REFERENCE_URLS = {
    "EUR": "https://data.broadinstitute.org/alkesgroup/LDSCORE/eur_w_ld_chr.tar.bz2",
    "EAS": "https://data.broadinstitute.org/alkesgroup/LDSCORE/eas_ldscores.tar.bz2",
    # Note: AFR, SAS, AMR may need different sources - placeholder for now
}


class PopulationPaths(TypedDict):
    ld_scores: Path
    weights: Path
    frq: Path


def get_reference_dir() -> Path:
    """Return the directory where LDSC reference files are stored."""
    return Path.home() / ".statgen_skills" / "ldsc_references"


def get_population_paths(population: str) -> PopulationPaths:
    """
    Get paths to reference files for a given population.

    Args:
        population: Population code (EUR, EAS, AFR, SAS, AMR)

    Returns:
        Dict with paths to ld_scores, weights, and frq files

    Raises:
        ValueError: If population not supported
    """
    if population.upper() not in SUPPORTED_POPULATIONS:
        raise ValueError(
            f"Population '{population}' not supported. "
            f"Choose from: {SUPPORTED_POPULATIONS}"
        )

    pop = population.upper()
    ref_dir = get_reference_dir()

    return PopulationPaths(
        ld_scores=ref_dir / pop / "ld_scores",
        weights=ref_dir / pop / "weights",
        frq=ref_dir / pop / "frq",
    )


def is_reference_available(population: str) -> bool:
    """Check if reference files for a population are downloaded."""
    paths = get_population_paths(population)
    # Check if the ld_scores directory exists and has files
    ld_path = paths["ld_scores"]
    if not ld_path.exists():
        return False
    # Check for at least one chromosome file
    chr_files = list(ld_path.glob("*.l2.ldscore.gz"))
    return len(chr_files) >= 1


def download_reference(
    population: str,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download reference LD scores for a population.

    Args:
        population: Population code (EUR, EAS, AFR, SAS, AMR)
        force: Re-download even if files exist
        verbose: Print progress messages

    Returns:
        Path to the downloaded reference directory
    """
    pop = population.upper()
    if pop not in SUPPORTED_POPULATIONS:
        raise ValueError(
            f"Population '{pop}' not supported. "
            f"Choose from: {SUPPORTED_POPULATIONS}"
        )

    if pop not in REFERENCE_URLS:
        raise ValueError(
            f"Download URL not available for {pop}. "
            f"Please download manually."
        )

    ref_dir = get_reference_dir()
    pop_dir = ref_dir / pop

    if is_reference_available(pop) and not force:
        if verbose:
            print(f"Reference data for {pop} already exists at {pop_dir}")
        return pop_dir

    # Create directories
    ref_dir.mkdir(parents=True, exist_ok=True)
    pop_dir.mkdir(exist_ok=True)

    url = REFERENCE_URLS[pop]
    archive_path = ref_dir / f"{pop}_temp.tar.bz2"

    if verbose:
        print(f"Downloading {pop} reference data from {url}...")

    # Download
    urllib.request.urlretrieve(url, archive_path)

    if verbose:
        print(f"Extracting to {pop_dir}...")

    # Extract
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(pop_dir)

    # Cleanup
    archive_path.unlink()

    if verbose:
        print(f"Reference data for {pop} ready at {pop_dir}")

    return pop_dir


def ensure_reference(population: str, verbose: bool = True) -> PopulationPaths:
    """
    Ensure reference data is available, downloading if needed.

    Args:
        population: Population code
        verbose: Print progress

    Returns:
        Paths to reference files
    """
    if not is_reference_available(population):
        download_reference(population, verbose=verbose)
    return get_population_paths(population)

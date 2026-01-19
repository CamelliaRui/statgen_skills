# scripts/fusion/reference_data.py
"""
Reference data management for FUSION TWAS.

Handles:
- GTEx v8 tissue expression weights download and caching
- 1000 Genomes LD reference panels (EUR, EAS, AFR)
- Validation of tissue names and populations
"""

import subprocess
import tarfile
from pathlib import Path


# GTEx v8 tissue names (49 tissues)
GTEX_V8_TISSUES = [
    "Adipose_Subcutaneous",
    "Adipose_Visceral_Omentum",
    "Adrenal_Gland",
    "Artery_Aorta",
    "Artery_Coronary",
    "Artery_Tibial",
    "Brain_Amygdala",
    "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia",
    "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum",
    "Brain_Cortex",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus",
    "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1",
    "Brain_Substantia_nigra",
    "Breast_Mammary_Tissue",
    "Cells_Cultured_fibroblasts",
    "Cells_EBV-transformed_lymphocytes",
    "Colon_Sigmoid",
    "Colon_Transverse",
    "Esophagus_Gastroesophageal_Junction",
    "Esophagus_Mucosa",
    "Esophagus_Muscularis",
    "Heart_Atrial_Appendage",
    "Heart_Left_Ventricle",
    "Kidney_Cortex",
    "Liver",
    "Lung",
    "Minor_Salivary_Gland",
    "Muscle_Skeletal",
    "Nerve_Tibial",
    "Ovary",
    "Pancreas",
    "Pituitary",
    "Prostate",
    "Skin_Not_Sun_Exposed_Suprapubic",
    "Skin_Sun_Exposed_Lower_leg",
    "Small_Intestine_Terminal_Ileum",
    "Spleen",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
    "Vagina",
    "Whole_Blood",
]

# Supported populations for LD reference
SUPPORTED_POPULATIONS = ["EUR", "EAS", "AFR"]

# Base URL for GTEx weights
GTEX_WEIGHTS_BASE_URL = "https://data.broadinstitute.org/alkesgroup/FUSION/WGT"

# Base URL for 1000G LD reference
LD_REFERENCE_BASE_URL = "https://data.broadinstitute.org/alkesgroup/FUSION/LDREF"


def get_weights_dir() -> Path:
    """
    Return the directory where FUSION weights are stored.

    Returns:
        Path to ~/.statgen_skills/fusion_weights
    """
    return Path.home() / ".statgen_skills" / "fusion_weights"


def get_ld_reference_dir() -> Path:
    """
    Return the directory where LD reference panels are stored.

    Returns:
        Path to ~/.statgen_skills/ld_reference
    """
    return Path.home() / ".statgen_skills" / "ld_reference"


def list_available_tissues() -> list[str]:
    """
    Return the list of available GTEx v8 tissues.

    Returns:
        List of 49 GTEx v8 tissue names
    """
    return GTEX_V8_TISSUES.copy()


def validate_tissue_name(tissue: str) -> None:
    """
    Validate that a tissue name is a valid GTEx v8 tissue.

    Args:
        tissue: Tissue name to validate

    Raises:
        ValueError: If tissue is not a valid GTEx v8 tissue name
    """
    if tissue not in GTEX_V8_TISSUES:
        raise ValueError(
            f"Invalid tissue name: '{tissue}'. "
            f"Use list_available_tissues() to see valid options."
        )


def validate_population(population: str) -> None:
    """
    Validate that a population is supported.

    Args:
        population: Population code to validate (EUR, EAS, AFR)

    Raises:
        ValueError: If population is not supported
    """
    if population not in SUPPORTED_POPULATIONS:
        raise ValueError(
            f"Invalid population: '{population}'. "
            f"Valid populations: {', '.join(SUPPORTED_POPULATIONS)}"
        )


def weights_available(tissue: str) -> bool:
    """
    Check if weights for a tissue have been downloaded.

    Args:
        tissue: GTEx v8 tissue name

    Returns:
        True if weights directory exists and contains files
    """
    validate_tissue_name(tissue)
    weights_path = get_weights_dir() / tissue
    if not weights_path.exists():
        return False
    # Check if directory has any .wgt.RDat files
    wgt_files = list(weights_path.glob("*.wgt.RDat"))
    return len(wgt_files) > 0


def ld_reference_available(population: str) -> bool:
    """
    Check if LD reference for a population has been downloaded.

    Args:
        population: Population code (EUR, EAS, AFR)

    Returns:
        True if LD reference files exist
    """
    validate_population(population)
    ld_path = get_ld_reference_dir() / population
    if not ld_path.exists():
        return False
    # Check for bed/bim/fam files (PLINK format)
    bed_files = list(ld_path.glob("*.bed"))
    return len(bed_files) > 0


def download_weights(tissue: str, force: bool = False, verbose: bool = True) -> Path:
    """
    Download GTEx v8 weights for a specific tissue.

    Downloads from:
    https://data.broadinstitute.org/alkesgroup/FUSION/WGT/GTEx.v8.ALL.{tissue}.tar.bz2

    Args:
        tissue: GTEx v8 tissue name
        force: Re-download even if weights already exist
        verbose: Print progress messages

    Returns:
        Path to the weights directory

    Raises:
        ValueError: If tissue name is invalid
        RuntimeError: If download fails
    """
    validate_tissue_name(tissue)

    weights_dir = get_weights_dir()
    tissue_dir = weights_dir / tissue

    if weights_available(tissue) and not force:
        if verbose:
            print(f"Weights for {tissue} already available at {tissue_dir}")
        return tissue_dir

    # Create weights directory
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Download URL
    url = f"{GTEX_WEIGHTS_BASE_URL}/GTEx.v8.ALL.{tissue}.tar.bz2"
    tar_path = weights_dir / f"GTEx.v8.ALL.{tissue}.tar.bz2"

    if verbose:
        print(f"Downloading weights for {tissue}...")
        print(f"URL: {url}")

    try:
        # Download using curl or wget
        subprocess.run(
            ["curl", "-L", "-o", str(tar_path), url],
            check=True,
            capture_output=not verbose,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download weights for {tissue}: {e}")

    if verbose:
        print(f"Extracting weights...")

    # Extract the tar.bz2 file
    try:
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(path=weights_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to extract weights for {tissue}: {e}")

    # Clean up tar file
    tar_path.unlink()

    if verbose:
        print(f"Weights for {tissue} installed at {tissue_dir}")

    return tissue_dir


def download_ld_reference(
    population: str, force: bool = False, verbose: bool = True
) -> Path:
    """
    Download 1000 Genomes LD reference panel for a population.

    Args:
        population: Population code (EUR, EAS, AFR)
        force: Re-download even if reference already exists
        verbose: Print progress messages

    Returns:
        Path to the LD reference directory

    Raises:
        ValueError: If population is invalid
        RuntimeError: If download fails
    """
    validate_population(population)

    ld_dir = get_ld_reference_dir()
    pop_dir = ld_dir / population

    if ld_reference_available(population) and not force:
        if verbose:
            print(f"LD reference for {population} already available at {pop_dir}")
        return pop_dir

    # Create LD reference directory
    ld_dir.mkdir(parents=True, exist_ok=True)

    # Download URL - FUSION provides 1000G LD reference
    url = f"{LD_REFERENCE_BASE_URL}/1000G.{population}.tar.bz2"
    tar_path = ld_dir / f"1000G.{population}.tar.bz2"

    if verbose:
        print(f"Downloading LD reference for {population}...")
        print(f"URL: {url}")

    try:
        subprocess.run(
            ["curl", "-L", "-o", str(tar_path), url],
            check=True,
            capture_output=not verbose,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download LD reference for {population}: {e}")

    if verbose:
        print(f"Extracting LD reference...")

    # Extract the tar.bz2 file
    try:
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(path=ld_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to extract LD reference for {population}: {e}")

    # Clean up tar file
    tar_path.unlink()

    # Rename extracted directory if needed
    if not pop_dir.exists():
        # The extracted directory might have a different name
        extracted_dirs = [
            d for d in ld_dir.iterdir()
            if d.is_dir() and population in d.name and d != pop_dir
        ]
        if extracted_dirs:
            extracted_dirs[0].rename(pop_dir)

    if verbose:
        print(f"LD reference for {population} installed at {pop_dir}")

    return pop_dir


def ensure_weights(tissue: str, verbose: bool = True) -> Path:
    """
    Ensure weights for a tissue are available, downloading if needed.

    Args:
        tissue: GTEx v8 tissue name
        verbose: Print progress messages

    Returns:
        Path to the weights directory
    """
    if weights_available(tissue):
        return get_weights_dir() / tissue
    return download_weights(tissue, verbose=verbose)


def ensure_ld_reference(population: str, verbose: bool = True) -> Path:
    """
    Ensure LD reference for a population is available, downloading if needed.

    Args:
        population: Population code (EUR, EAS, AFR)
        verbose: Print progress messages

    Returns:
        Path to the LD reference directory
    """
    if ld_reference_available(population):
        return get_ld_reference_dir() / population
    return download_ld_reference(population, verbose=verbose)

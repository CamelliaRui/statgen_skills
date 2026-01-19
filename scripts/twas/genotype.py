# scripts/twas/genotype.py
"""
Genotype loading and management for TWAS simulation.

Handles:
- PLINK file loading via pandas-plink
- Cis-region subsetting
- Sample splitting for train/test
- Optional 1000G reference download
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    from pandas_plink import read_plink1_bin
    HAS_PANDAS_PLINK = True
except ImportError:
    HAS_PANDAS_PLINK = False


class GenotypeData(NamedTuple):
    """Container for genotype data."""
    genotypes: np.ndarray  # (n_samples, n_snps)
    snp_ids: np.ndarray    # SNP identifiers
    positions: np.ndarray  # Base pair positions
    chromosomes: np.ndarray  # Chromosome numbers
    sample_ids: np.ndarray   # Sample identifiers
    a1: np.ndarray  # Effect allele
    a2: np.ndarray  # Other allele


def get_reference_dir() -> Path:
    """Return directory for TWAS reference files."""
    return Path.home() / ".statgen_skills" / "twas_references"


def load_plink(prefix: str | Path) -> GenotypeData:
    """
    Load genotype data from PLINK binary files.
    
    Args:
        prefix: Path prefix for .bed/.bim/.fam files
        
    Returns:
        GenotypeData with genotypes and variant info
        
    Raises:
        ImportError: If pandas-plink not installed
        FileNotFoundError: If PLINK files not found
    """
    if not HAS_PANDAS_PLINK:
        raise ImportError(
            "pandas-plink required for PLINK file loading. "
            "Install with: pip install pandas-plink"
        )
    
    prefix = Path(prefix)
    bed_file = prefix.with_suffix(".bed")
    if not bed_file.exists():
        # Try adding .bed if prefix already has it
        if not prefix.exists():
            raise FileNotFoundError(f"PLINK files not found: {prefix}")
    
    # Read PLINK files
    (bim, fam, bed) = read_plink1_bin(str(prefix))
    
    # Convert to numpy, handling missing values
    genotypes = bed.compute().values
    genotypes = np.nan_to_num(genotypes, nan=0).astype(np.int8)
    
    return GenotypeData(
        genotypes=genotypes,
        snp_ids=bim["snp"].values,
        positions=bim["pos"].values.astype(np.int64),
        chromosomes=bim["chrom"].values,
        sample_ids=fam["iid"].values,
        a1=bim["a0"].values,  # pandas-plink uses a0/a1
        a2=bim["a1"].values,
    )


def subset_to_cis_region(
    genotypes: np.ndarray,
    positions: np.ndarray,
    gene_position: int,
    window: int = 500000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subset genotypes to SNPs within a cis-window of a gene.
    
    Args:
        genotypes: (n_samples, n_snps) genotype matrix
        positions: Array of SNP positions
        gene_position: Gene TSS position
        window: Cis-window size in bp (default 500kb)
        
    Returns:
        Tuple of (subset_genotypes, mask)
    """
    mask = np.abs(positions - gene_position) <= window
    return genotypes[:, mask], mask


def sample_individuals(
    n_total: int,
    n_sample: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Randomly sample individual indices.
    
    Args:
        n_total: Total number of individuals
        n_sample: Number to sample
        seed: Random seed for reproducibility
        
    Returns:
        Array of sampled indices
    """
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_sample, replace=False)


def split_samples(
    n_total: int,
    n_eqtl: int,
    n_gwas: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split samples into eQTL training, GWAS, and test sets.
    
    Args:
        n_total: Total number of samples
        n_eqtl: eQTL training set size
        n_gwas: GWAS set size
        seed: Random seed
        
    Returns:
        Tuple of (eqtl_indices, gwas_indices, test_indices)
    """
    if n_eqtl + n_gwas > n_total:
        raise ValueError(
            f"Requested {n_eqtl} + {n_gwas} = {n_eqtl + n_gwas} samples, "
            f"but only {n_total} available"
        )
    
    rng = np.random.default_rng(seed)
    all_indices = rng.permutation(n_total)
    
    eqtl_idx = all_indices[:n_eqtl]
    gwas_idx = all_indices[n_eqtl:n_eqtl + n_gwas]
    test_idx = all_indices[n_eqtl + n_gwas:]
    
    return eqtl_idx, gwas_idx, test_idx

# tests/twas/test_genotype.py
"""Tests for genotype loading and management."""

import pytest
import numpy as np
import tempfile
from pathlib import Path


def test_load_plink_returns_genotype_data():
    """Test loading genotypes from PLINK files."""
    from scripts.twas.genotype import load_plink

    # Will fail with FileNotFoundError or ImportError (if pandas-plink not installed)
    with pytest.raises((FileNotFoundError, ImportError)):
        load_plink("nonexistent_prefix")


def test_get_reference_dir_returns_path():
    """Test reference directory path."""
    from scripts.twas.genotype import get_reference_dir

    ref_dir = get_reference_dir()
    assert ref_dir == Path.home() / ".statgen_skills" / "twas_references"


def test_subset_to_cis_region():
    """Test subsetting genotypes to cis-region around a gene."""
    from scripts.twas.genotype import subset_to_cis_region

    # Mock genotype data
    n_samples = 100
    n_snps = 50
    genotypes = np.random.randint(0, 3, (n_samples, n_snps))
    positions = np.arange(1000000, 1000000 + n_snps * 1000, 1000)

    # Gene at position 1025000, cis window 500kb
    gene_pos = 1025000
    window = 500000

    subset, mask = subset_to_cis_region(
        genotypes, positions, gene_pos, window
    )

    assert subset.shape[0] == n_samples
    assert subset.shape[1] <= n_snps
    assert np.all(np.abs(positions[mask] - gene_pos) <= window)


def test_sample_individuals():
    """Test random sampling of individuals."""
    from scripts.twas.genotype import sample_individuals

    n_total = 1000
    n_sample = 100

    indices = sample_individuals(n_total, n_sample, seed=42)

    assert len(indices) == n_sample
    assert len(set(indices)) == n_sample  # All unique
    assert all(0 <= i < n_total for i in indices)

    # Same seed gives same result
    indices2 = sample_individuals(n_total, n_sample, seed=42)
    assert np.array_equal(indices, indices2)

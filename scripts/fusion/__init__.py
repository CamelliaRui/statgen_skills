# scripts/fusion/__init__.py
"""
FUSION TWAS module.

Provides utilities for running FUSION Transcriptome-Wide Association Studies
using pre-computed gene expression weights and GWAS summary statistics.
"""

from .utils import (
    get_fusion_dir,
    check_r_installed,
    check_plink_installed,
    check_fusion_installed,
    check_dependencies,
    download_fusion,
    validate_sumstats_columns,
    format_sumstats_for_fusion,
)

__all__ = [
    "get_fusion_dir",
    "check_r_installed",
    "check_plink_installed",
    "check_fusion_installed",
    "check_dependencies",
    "download_fusion",
    "validate_sumstats_columns",
    "format_sumstats_for_fusion",
]

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

from .reference_data import (
    list_available_tissues,
    download_weights,
    download_ld_reference,
    weights_available,
    ld_reference_available,
)

from .run_fusion import run_twas_association, TWASResults
from .parsers import TWASResult

__all__ = [
    "get_fusion_dir",
    "check_r_installed",
    "check_plink_installed",
    "check_fusion_installed",
    "check_dependencies",
    "download_fusion",
    "validate_sumstats_columns",
    "format_sumstats_for_fusion",
    "list_available_tissues",
    "download_weights",
    "download_ld_reference",
    "weights_available",
    "ld_reference_available",
    "run_twas_association",
    "TWASResults",
    "TWASResult",
]

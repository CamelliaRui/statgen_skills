"""Ensembl REST API queries for variant annotation."""

from .annotate import annotate_by_rsid, annotate_variants, get_variant_info

__all__ = ["annotate_variants", "annotate_by_rsid", "get_variant_info"]

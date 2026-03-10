"""GWAS Catalog REST API queries."""

from .lookup import (
    bulk_lookup_variants,
    lookup_gene,
    lookup_study,
    lookup_trait,
    lookup_variant,
)

__all__ = [
    "lookup_variant",
    "lookup_gene",
    "lookup_trait",
    "lookup_study",
    "bulk_lookup_variants",
]

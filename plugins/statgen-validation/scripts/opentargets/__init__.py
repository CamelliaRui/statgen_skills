"""OpenTargets Platform API queries."""

from .query import query_target, query_disease_associations, query_variant, search

__all__ = ["query_target", "query_disease_associations", "query_variant", "search"]

"""
OpenTargets Platform GraphQL API client.

Queries the OpenTargets Platform for gene-disease associations,
target tractability, genetic constraint, and variant evidence.

Endpoint: https://api.platform.opentargets.org/api/v4/graphql
Authentication: None required.
"""

import json
from typing import Any

import requests

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"


def _graphql(query: str, variables: dict[str, Any] | None = None) -> dict:
    """Execute a GraphQL query against the OpenTargets API."""
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables

    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()
    if "errors" in result:
        raise RuntimeError(
            f"OpenTargets GraphQL errors: {json.dumps(result['errors'], indent=2)}"
        )
    return result["data"]


def query_target(ensembl_id: str) -> dict:
    """
    Query a gene/target by Ensembl gene ID.

    Args:
        ensembl_id: Ensembl gene ID (e.g., "ENSG00000169174" for PCSK9).

    Returns:
        Dict with keys: id, symbol, biotype, tractability, genetic_constraint,
        disease_count.
    """
    query = """
    query($id: String!) {
      target(ensemblId: $id) {
        id
        approvedSymbol
        biotype
        tractability {
          label
          modality
          value
        }
        geneticConstraint {
          constraintType
          score
        }
        associatedDiseases(page: {index: 0, size: 0}) {
          count
        }
      }
    }
    """
    data = _graphql(query, {"id": ensembl_id})
    target = data["target"]
    if target is None:
        return {"id": ensembl_id, "found": False}

    return {
        "id": target["id"],
        "symbol": target["approvedSymbol"],
        "biotype": target["biotype"],
        "found": True,
        "tractability": target.get("tractability", []),
        "genetic_constraint": target.get("geneticConstraint", []),
        "disease_count": target["associatedDiseases"]["count"],
    }


def query_disease_associations(
    ensembl_id: str, max_results: int = 25
) -> list[dict]:
    """
    Get diseases associated with a target, ranked by overall score.

    Args:
        ensembl_id: Ensembl gene ID.
        max_results: Maximum number of associations to return.

    Returns:
        List of dicts with keys: disease_id, disease_name, score, datatype_scores.
    """
    query = """
    query($id: String!, $size: Int!) {
      target(ensemblId: $id) {
        associatedDiseases(page: {index: 0, size: $size}) {
          rows {
            disease {
              id
              name
            }
            score
            datatypeScores {
              id
              score
            }
          }
        }
      }
    }
    """
    data = _graphql(query, {"id": ensembl_id, "size": max_results})
    target = data["target"]
    if target is None:
        return []

    return [
        {
            "disease_id": row["disease"]["id"],
            "disease_name": row["disease"]["name"],
            "score": row["score"],
            "datatype_scores": {
                ds["id"]: ds["score"] for ds in row.get("datatypeScores", [])
            },
        }
        for row in target["associatedDiseases"]["rows"]
    ]


def query_variant(variant_id: str) -> dict:
    """
    Query a variant by OpenTargets variant ID (chr_pos_ref_alt format).

    Args:
        variant_id: Variant ID in format "1_55039974_G_A".

    Returns:
        Dict with variant annotation data.
    """
    query = """
    query($id: String!) {
      variant(variantId: $id) {
        id
        rsId
        chromosome
        position
        refAllele
        altAllele
        nearestCodingGene {
          id
          approvedSymbol
        }
        mostSevereConsequence
      }
    }
    """
    data = _graphql(query, {"id": variant_id})
    variant = data.get("variant")
    if variant is None:
        return {"id": variant_id, "found": False}

    nearest = variant.get("nearestCodingGene") or {}
    return {
        "id": variant["id"],
        "rsid": variant.get("rsId"),
        "chromosome": variant["chromosome"],
        "position": variant["position"],
        "ref": variant["refAllele"],
        "alt": variant["altAllele"],
        "found": True,
        "nearest_gene": nearest.get("approvedSymbol"),
        "nearest_gene_id": nearest.get("id"),
        "consequence": variant.get("mostSevereConsequence"),
    }


def search(
    query_string: str, entity_types: list[str] | None = None
) -> list[dict]:
    """
    Free-text search across targets, diseases, and drugs.

    Args:
        query_string: Search term (e.g., "LDL cholesterol", "PCSK9").
        entity_types: Optional filter for entity types
            (e.g., ["target"], ["disease"], ["drug"]).

    Returns:
        List of dicts with keys: id, name, entity_type, score, description.
    """
    entity_filter = ""
    if entity_types:
        names = ", ".join(f'"{e}"' for e in entity_types)
        entity_filter = f", entityNames: [{names}]"

    query = f"""
    query($q: String!) {{
      search(queryString: $q{entity_filter}, page: {{index: 0, size: 25}}) {{
        total
        hits {{
          id
          entity
          name
          score
          description
        }}
      }}
    }}
    """
    data = _graphql(query, {"q": query_string})
    hits = data["search"]["hits"]

    return [
        {
            "id": hit["id"],
            "name": hit["name"],
            "entity_type": hit["entity"],
            "score": hit["score"],
            "description": hit.get("description"),
        }
        for hit in hits
    ]

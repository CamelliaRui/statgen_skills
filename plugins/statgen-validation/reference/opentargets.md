# OpenTargets Platform API Reference

## Overview

GraphQL API for gene-disease associations, target tractability, and variant evidence.

- **Endpoint**: `https://api.platform.opentargets.org/api/v4/graphql`
- **Authentication**: None required
- **Rate limits**: Generous fair-usage limits per IP
- **Interactive browser**: `https://api.platform.opentargets.org/api/v4/graphql/browser`

## Scripts

- `scripts/opentargets/query.py` — All query functions

## API Functions

### `query_target(ensembl_id: str) -> dict`

Query a gene/target by Ensembl ID. Returns disease associations, tractability, and genetic constraint.

```python
from scripts.opentargets.query import query_target

result = query_target("ENSG00000169174")  # PCSK9
# Returns: symbol, diseases, tractability, genetic constraint (loeuf, mis_z)
```

### `query_disease_associations(ensembl_id: str, max_results: int = 25) -> list[dict]`

Get diseases associated with a target, ranked by overall association score.

```python
from scripts.opentargets.query import query_disease_associations

associations = query_disease_associations("ENSG00000169174")
# Returns: [{"disease_id": ..., "disease_name": ..., "score": ..., "evidence_count": ...}, ...]
```

### `query_variant(variant_id: str) -> dict`

Query a variant by Open Targets variant ID (chr_pos_ref_alt format).

```python
from scripts.opentargets.query import query_variant

result = query_variant("1_55039974_G_A")
# Returns: rsid, consequence, nearest genes, credible set membership
```

### `search(query: str, entity_types: list[str] | None = None) -> list[dict]`

Free-text search across targets, diseases, and drugs.

```python
from scripts.opentargets.query import search

results = search("LDL cholesterol", entity_types=["disease"])
# Returns: [{"id": ..., "name": ..., "entity_type": ..., "score": ...}, ...]
```

## GraphQL Query Templates

### Target query
```graphql
query {
  target(ensemblId: "ENSG00000169174") {
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
    associatedDiseases(page: {index: 0, size: 25}) {
      count
      rows {
        disease { id name }
        score
        datatypeScores { id score }
      }
    }
  }
}
```

### Search query
```graphql
query {
  search(queryString: "LDL cholesterol", entityNames: ["disease"]) {
    total
    hits {
      id
      entity
      name
      score
    }
  }
}
```

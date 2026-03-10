# GWAS Catalog API Reference

## Overview

REST API for curated GWAS associations from the EBI GWAS Catalog.

- **Base URL**: `https://www.ebi.ac.uk/gwas/rest/api`
- **Authentication**: None required
- **Rate limits**: Enforced per IP (429 on excess)
- **Python client**: `pandasgwas` (recommended, handles pagination)

## Scripts

- `scripts/gwas_catalog/lookup.py` — All lookup functions

## API Functions

### `lookup_variant(rsid: str) -> dict`

Look up a variant by rsID. Returns known associations, mapped genes, traits, and study metadata.

```python
from scripts.gwas_catalog.lookup import lookup_variant

result = lookup_variant("rs4420638")
# Returns: {"rsid": ..., "associations": [...], "mapped_genes": [...], "traits": [...]}
```

### `lookup_gene(gene_name: str) -> dict`

Look up associations for a gene by symbol.

```python
from scripts.gwas_catalog.lookup import lookup_gene

result = lookup_gene("PCSK9")
# Returns: {"gene": ..., "associations": [...], "traits": [...]}
```

### `lookup_trait(trait: str) -> dict`

Search for GWAS associations by trait keyword or EFO ID.

```python
from scripts.gwas_catalog.lookup import lookup_trait

result = lookup_trait("LDL cholesterol")
# Returns: {"trait": ..., "efo_id": ..., "associations": [...], "studies": [...]}
```

### `lookup_study(study_id: str) -> dict`

Get details for a specific GWAS study by accession (e.g., GCST000854).

```python
from scripts.gwas_catalog.lookup import lookup_study

result = lookup_study("GCST000854")
# Returns: {"study_id": ..., "title": ..., "trait": ..., "associations": [...]}
```

### `bulk_lookup_variants(rsids: list[str]) -> list[dict]`

Look up multiple variants. Returns association summary for each.

```python
from scripts.gwas_catalog.lookup import bulk_lookup_variants

results = bulk_lookup_variants(["rs4420638", "rs429358", "rs7412"])
# Returns: [{"rsid": ..., "n_associations": ..., "traits": [...], "top_pvalue": ...}, ...]
```

## REST Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /singleNucleotidePolymorphisms/{rsId}` | Variant details + associations |
| `GET /singleNucleotidePolymorphisms/{rsId}/associations` | Associations for variant |
| `GET /associations` | Query associations (filterable) |
| `GET /efoTraits/search/findBySearchTerm?searchTerm=X` | Search traits |
| `GET /studies/{accessionId}` | Study details |
| `GET /studies/{accessionId}/associations` | Associations for study |

## Dependencies

```bash
uv pip install pandasgwas requests
```

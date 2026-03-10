# Ensembl REST API Reference

## Overview

REST API for variant annotation via the Variant Effect Predictor (VEP) and general variant info.

- **Base URL**: `https://rest.ensembl.org`
- **Authentication**: None required
- **Rate limits**: 55,000 requests/hour (~15/sec) per IP
- **Rate limit headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

## Scripts

- `scripts/ensembl/annotate.py` — All annotation functions

## API Functions

### `annotate_variants(variants: list[str], species: str = "homo_sapiens") -> list[dict]`

Annotate variants in VCF-like format using VEP POST endpoint. Batch up to 200 variants per request.

```python
from scripts.ensembl.annotate import annotate_variants

results = annotate_variants([
    "21 26960070 rs116645811 G A",
    "21 26965148 rs1135638 G A",
])
# Returns: [{"input": ..., "most_severe_consequence": ..., "transcript_consequences": [...], ...}]
```

### `annotate_by_rsid(rsids: list[str]) -> list[dict]`

Annotate variants by rsID using VEP ID POST endpoint. Batch up to 200 per request.

```python
from scripts.ensembl.annotate import annotate_by_rsid

results = annotate_by_rsid(["rs4420638", "rs429358"])
# Returns per variant: consequence, SIFT, PolyPhen, CADD, allele frequencies
```

### `get_variant_info(rsid: str, species: str = "homo_sapiens") -> dict`

Get basic variant info (alleles, MAF, location, clinical significance).

```python
from scripts.ensembl.annotate import get_variant_info

info = get_variant_info("rs4420638")
# Returns: {"rsid": ..., "alleles": ..., "maf": ..., "clinical_significance": [...]}
```

## REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vep/:species/region` | POST | Annotate by genomic region (batch) |
| `/vep/:species/id` | POST | Annotate by variant ID (batch) |
| `/vep/:species/id/:id` | GET | Annotate single variant by ID |
| `/variation/:species/:id` | GET | Basic variant information |

## VEP Response Fields

Key fields in each annotation result:
- `most_severe_consequence` — e.g., missense_variant, regulatory_region_variant
- `transcript_consequences[]` — per-transcript: gene_symbol, consequence_terms, sift_prediction, polyphen_prediction, cadd_phred
- `regulatory_feature_consequences[]` — regulatory region impacts
- `colocated_variants[]` — known variants at the same position with frequencies

## Dependencies

```bash
uv pip install requests
```

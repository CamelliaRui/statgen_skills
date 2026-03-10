---
name: biological-validation
description: >-
  Use when validating GWAS findings, checking variant-gene-disease links,
  looking up known associations for SNPs or genes, annotating variants with
  functional consequences, or cross-referencing results against OpenTargets,
  GWAS Catalog, or Ensembl.
user-invocable: false
metadata:
  short-description: Biological validation of statistical genetics findings
---

# Biological Validation

Validate statistical genetics findings against public databases. All APIs are open-access (no authentication required).

## Tools

### OpenTargets

Query the OpenTargets Platform (GraphQL) for gene-disease evidence, target tractability, and genetic constraint.

**API:** `query_target()`, `query_disease_associations()`, `query_variant()`, `search()`

See [reference/opentargets.md](../../reference/opentargets.md) for query patterns and examples.

### GWAS Catalog

Query the EBI GWAS Catalog for known associations by variant, gene, or trait. Returns curated GWAS associations with p-values, effect sizes, and study metadata.

**API:** `lookup_variant()`, `lookup_gene()`, `lookup_trait()`, `lookup_study()`

See [reference/gwas-catalog.md](../../reference/gwas-catalog.md) for endpoints and examples.

### Ensembl VEP

Annotate variants with functional consequences (missense, regulatory, splice), allele frequencies, and CADD/SIFT/PolyPhen scores via the Ensembl Variant Effect Predictor REST API.

**API:** `annotate_variants()`, `annotate_by_rsid()`, `get_variant_info()`

See [reference/ensembl.md](../../reference/ensembl.md) for endpoints and examples.

## Validation Workflows

### Post-Fine-Mapping Validation
After SuSiE fine-mapping, validate credible set variants:
1. Look up each credible set variant in GWAS Catalog for prior associations
2. Annotate with Ensembl VEP for functional consequences
3. Query OpenTargets for gene-disease evidence at the locus

### Post-TWAS Validation
After FUSION TWAS, validate significant genes:
1. Query OpenTargets for known disease associations
2. Check target tractability and genetic constraint
3. Cross-reference with GWAS Catalog for supporting evidence

### Replication Check
For any GWAS hit list:
1. Bulk lookup in GWAS Catalog for independent replication
2. Filter by ancestry and trait relevance
3. Report novel vs. replicated findings

## Example Prompts

```
"Validate my top 10 fine-mapping variants against GWAS Catalog and OpenTargets"
"Annotate these credible set SNPs with VEP consequences"
"Check if gene PCSK9 has known associations with LDL cholesterol in OpenTargets"
"Look up rs4420638 in GWAS Catalog"
"What are the functional consequences of variants in my credible set?"
```

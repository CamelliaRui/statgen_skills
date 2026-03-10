"""
GWAS Catalog REST API client.

Queries the EBI GWAS Catalog for known associations by variant, gene,
trait, or study accession.

Base URL: https://www.ebi.ac.uk/gwas/rest/api
Authentication: None required.
Rate limits: Enforced per IP (429 on excess).
"""

import time

import requests

BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"


def _get(endpoint: str, params: dict | None = None) -> dict:
    """Make a GET request to the GWAS Catalog API with retry on 429."""
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(3):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 5))
            time.sleep(retry_after)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"GWAS Catalog rate limit exceeded after 3 retries: {url}")


def _extract_associations(data: dict) -> list[dict]:
    """Extract association records from GWAS Catalog API response."""
    associations = []
    embedded = data.get("_embedded", {})
    for assoc in embedded.get("associations", []):
        strongest = assoc.get("strongestRiskAlleles", [{}])
        risk_allele = strongest[0].get("riskAlleleName", "") if strongest else ""

        traits = []
        for trait in assoc.get("efoTraits", []):
            traits.append(trait.get("trait", ""))

        genes = []
        for locus in assoc.get("loci", []):
            for gene_group in locus.get("authorReportedGenes", []):
                name = gene_group.get("geneName", "")
                if name:
                    genes.append(name)

        associations.append({
            "pvalue": assoc.get("pvalue"),
            "pvalue_mantissa": assoc.get("pvalueMantissa"),
            "pvalue_exponent": assoc.get("pvalueExponent"),
            "risk_allele": risk_allele,
            "or_beta": assoc.get("orPerCopyNum"),
            "ci": assoc.get("range"),
            "traits": traits,
            "genes": genes,
        })
    return associations


def lookup_variant(rsid: str) -> dict:
    """
    Look up a variant by rsID in the GWAS Catalog.

    Args:
        rsid: dbSNP rsID (e.g., "rs4420638").

    Returns:
        Dict with keys: rsid, found, associations, mapped_genes, traits.
    """
    try:
        data = _get(f"/singleNucleotidePolymorphisms/{rsid}/associations")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {"rsid": rsid, "found": False, "associations": []}
        raise

    associations = _extract_associations(data)
    all_genes = set()
    all_traits = set()
    for assoc in associations:
        all_genes.update(assoc["genes"])
        all_traits.update(assoc["traits"])

    return {
        "rsid": rsid,
        "found": len(associations) > 0,
        "n_associations": len(associations),
        "associations": associations,
        "mapped_genes": sorted(all_genes),
        "traits": sorted(all_traits),
    }


def lookup_gene(gene_name: str) -> dict:
    """
    Look up GWAS associations for a gene by symbol.

    Args:
        gene_name: Gene symbol (e.g., "PCSK9").

    Returns:
        Dict with keys: gene, found, associations, traits.
    """
    try:
        data = _get(f"/singleNucleotidePolymorphisms/search/findByGene", params={"geneName": gene_name})
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {"gene": gene_name, "found": False, "associations": []}
        raise

    # Extract variants and their associations
    embedded = data.get("_embedded", {})
    variants = embedded.get("singleNucleotidePolymorphisms", [])

    all_traits = set()
    variant_list = []
    for snp in variants:
        rsid = snp.get("rsId", "")
        variant_list.append(rsid)

    return {
        "gene": gene_name,
        "found": len(variant_list) > 0,
        "n_variants": len(variant_list),
        "variants": variant_list[:50],  # Limit to avoid huge responses
    }


def lookup_trait(trait: str) -> dict:
    """
    Search GWAS Catalog for associations by trait keyword.

    Args:
        trait: Trait keyword (e.g., "LDL cholesterol") or EFO ID.

    Returns:
        Dict with keys: trait, found, efo_traits, n_associations.
    """
    data = _get("/efoTraits/search/findBySearchTerm", params={"searchTerm": trait})
    embedded = data.get("_embedded", {})
    efo_traits = embedded.get("efoTraits", [])

    results = []
    for efo in efo_traits:
        results.append({
            "trait": efo.get("trait", ""),
            "short_form": efo.get("shortForm", ""),
            "uri": efo.get("uri", ""),
        })

    return {
        "query": trait,
        "found": len(results) > 0,
        "n_traits": len(results),
        "efo_traits": results[:25],
    }


def lookup_study(study_id: str) -> dict:
    """
    Get details for a specific GWAS study by accession.

    Args:
        study_id: Study accession (e.g., "GCST000854").

    Returns:
        Dict with keys: study_id, found, title, trait, sample_size.
    """
    try:
        data = _get(f"/studies/{study_id}")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {"study_id": study_id, "found": False}
        raise

    return {
        "study_id": study_id,
        "found": True,
        "title": data.get("publicationInfo", {}).get("title", ""),
        "pubmed_id": data.get("publicationInfo", {}).get("pubmedId", ""),
        "initial_sample_size": data.get("initialSampleSize", ""),
        "replication_sample_size": data.get("replicationSampleSize", ""),
    }


def bulk_lookup_variants(rsids: list[str]) -> list[dict]:
    """
    Look up multiple variants in the GWAS Catalog.

    Args:
        rsids: List of rsIDs to look up.

    Returns:
        List of summary dicts per variant.
    """
    results = []
    for rsid in rsids:
        result = lookup_variant(rsid)
        results.append({
            "rsid": rsid,
            "found": result["found"],
            "n_associations": result.get("n_associations", 0),
            "traits": result.get("traits", []),
            "mapped_genes": result.get("mapped_genes", []),
        })
    return results

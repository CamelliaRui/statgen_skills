"""
Ensembl REST API client for variant annotation.

Uses the Variant Effect Predictor (VEP) endpoint for functional
consequence annotation, SIFT/PolyPhen predictions, and CADD scores.

Base URL: https://rest.ensembl.org
Authentication: None required.
Rate limits: 55,000 requests/hour per IP.
"""

import time

import requests

BASE_URL = "https://rest.ensembl.org"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
MAX_BATCH_SIZE = 200


def _post(endpoint: str, data: dict) -> list[dict]:
    """POST to Ensembl REST API with rate limit handling."""
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(3):
        response = requests.post(url, headers=HEADERS, json=data, timeout=60)
        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", 5))
            time.sleep(retry_after)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Ensembl rate limit exceeded after 3 retries: {url}")


def _get(endpoint: str) -> dict:
    """GET from Ensembl REST API with rate limit handling."""
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(3):
        response = requests.get(url, headers=HEADERS, timeout=30)
        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", 5))
            time.sleep(retry_after)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Ensembl rate limit exceeded after 3 retries: {url}")


def _parse_vep_result(result: dict) -> dict:
    """Parse a single VEP result into a clean dict."""
    parsed = {
        "input": result.get("input", result.get("id", "")),
        "most_severe_consequence": result.get("most_severe_consequence"),
        "transcript_consequences": [],
        "colocated_variants": [],
    }

    for tc in result.get("transcript_consequences", []):
        parsed["transcript_consequences"].append({
            "gene_symbol": tc.get("gene_symbol"),
            "gene_id": tc.get("gene_id"),
            "consequence_terms": tc.get("consequence_terms", []),
            "impact": tc.get("impact"),
            "biotype": tc.get("biotype"),
            "sift_prediction": tc.get("sift_prediction"),
            "sift_score": tc.get("sift_score"),
            "polyphen_prediction": tc.get("polyphen_prediction"),
            "polyphen_score": tc.get("polyphen_score"),
            "cadd_phred": tc.get("cadd_phred"),
            "cadd_raw": tc.get("cadd_raw"),
            "amino_acids": tc.get("amino_acids"),
            "codons": tc.get("codons"),
        })

    for cv in result.get("colocated_variants", []):
        parsed["colocated_variants"].append({
            "id": cv.get("id"),
            "allele_string": cv.get("allele_string"),
            "minor_allele": cv.get("minor_allele"),
            "minor_allele_freq": cv.get("minor_allele_freq"),
            "clinical_significance": cv.get("clin_sig", []),
        })

    return parsed


def annotate_variants(
    variants: list[str], species: str = "homo_sapiens"
) -> list[dict]:
    """
    Annotate variants using VEP region POST endpoint.

    Args:
        variants: List of variants in VCF-like format:
            "chr pos id ref alt" (e.g., "21 26960070 rs116645811 G A").
        species: Species name (default: "homo_sapiens").

    Returns:
        List of parsed VEP annotation dicts.
    """
    all_results = []
    for i in range(0, len(variants), MAX_BATCH_SIZE):
        batch = variants[i : i + MAX_BATCH_SIZE]
        results = _post(
            f"/vep/{species}/region",
            {"variants": batch},
        )
        all_results.extend(_parse_vep_result(r) for r in results)
    return all_results


def annotate_by_rsid(
    rsids: list[str], species: str = "homo_sapiens"
) -> list[dict]:
    """
    Annotate variants by rsID using VEP ID POST endpoint.

    Args:
        rsids: List of rsIDs (e.g., ["rs4420638", "rs429358"]).
        species: Species name (default: "homo_sapiens").

    Returns:
        List of parsed VEP annotation dicts.
    """
    all_results = []
    for i in range(0, len(rsids), MAX_BATCH_SIZE):
        batch = rsids[i : i + MAX_BATCH_SIZE]
        results = _post(
            f"/vep/{species}/id",
            {"ids": batch},
        )
        all_results.extend(_parse_vep_result(r) for r in results)
    return all_results


def get_variant_info(rsid: str, species: str = "homo_sapiens") -> dict:
    """
    Get basic variant information from Ensembl.

    Args:
        rsid: dbSNP rsID (e.g., "rs4420638").
        species: Species name (default: "homo_sapiens").

    Returns:
        Dict with keys: rsid, alleles, location, maf, clinical_significance.
    """
    result = _get(f"/variation/{species}/{rsid}")

    mappings = result.get("mappings", [])
    location = None
    if mappings:
        m = mappings[0]
        location = f"{m.get('seq_region_name')}:{m.get('start')}-{m.get('end')}"

    return {
        "rsid": rsid,
        "alleles": result.get("allele_string"),
        "ancestral_allele": result.get("ancestral_allele"),
        "minor_allele": result.get("minor_allele"),
        "maf": result.get("MAF"),
        "location": location,
        "clinical_significance": result.get("clinical_significance", []),
        "synonyms": result.get("synonyms", [])[:5],
    }

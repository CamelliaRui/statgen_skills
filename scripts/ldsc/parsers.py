# scripts/ldsc/parsers.py
"""
Parsers for LDSC log files.

Extracts structured data from LDSC text output.
"""

import re
from typing import Any


def parse_h2_log(log_content: str) -> dict[str, Any]:
    """
    Parse heritability estimation log.

    Args:
        log_content: Raw log file content

    Returns:
        Dict with h2, h2_se, lambda_gc, mean_chi2, intercept, intercept_se
    """
    result = {
        "h2": None,
        "h2_se": None,
        "lambda_gc": None,
        "mean_chi2": None,
        "intercept": None,
        "intercept_se": None,
        "ratio": None,
        "ratio_se": None,
    }

    # Total Observed scale h2: 0.1234 (0.0123)
    h2_match = re.search(
        r"Total Observed scale h2:\s*([-\d.]+)\s*\(([\d.]+)\)",
        log_content
    )
    if h2_match:
        result["h2"] = float(h2_match.group(1))
        result["h2_se"] = float(h2_match.group(2))

    # Lambda GC: 1.05
    lambda_match = re.search(r"Lambda GC:\s*([\d.]+)", log_content)
    if lambda_match:
        result["lambda_gc"] = float(lambda_match.group(1))

    # Mean Chi^2: 1.10
    chi2_match = re.search(r"Mean Chi\^2:\s*([\d.]+)", log_content)
    if chi2_match:
        result["mean_chi2"] = float(chi2_match.group(1))

    # Intercept: 1.0012 (0.0078)
    intercept_match = re.search(
        r"Intercept:\s*([-\d.]+)\s*\(([\d.]+)\)",
        log_content
    )
    if intercept_match:
        result["intercept"] = float(intercept_match.group(1))
        result["intercept_se"] = float(intercept_match.group(2))

    # Ratio: 0.0123 (0.0456)
    ratio_match = re.search(
        r"Ratio:\s*([-\d.]+)\s*\(([\d.]+)\)",
        log_content
    )
    if ratio_match:
        result["ratio"] = float(ratio_match.group(1))
        result["ratio_se"] = float(ratio_match.group(2))

    return result


def parse_rg_log(log_content: str) -> dict[str, Any]:
    """
    Parse genetic correlation log.

    Args:
        log_content: Raw log file content

    Returns:
        Dict with correlations list
    """
    result = {"correlations": []}

    # Find the summary table
    # p1	p2	rg	se	z	p
    lines = log_content.strip().split("\n")

    in_table = False
    for line in lines:
        if line.startswith("p1\tp2\trg"):
            in_table = True
            continue

        if in_table and line.strip():
            parts = line.split("\t")
            if len(parts) >= 6:
                result["correlations"].append({
                    "trait1": parts[0],
                    "trait2": parts[1],
                    "rg": float(parts[2]),
                    "se": float(parts[3]),
                    "z": float(parts[4]),
                    "p": float(parts[5]),
                })

    return result


def parse_partitioned_h2_log(log_content: str) -> dict[str, Any]:
    """
    Parse stratified LDSC (partitioned heritability) log.

    Args:
        log_content: Raw log file content

    Returns:
        Dict with categories list containing prop_snps, prop_h2, enrichment, etc.
    """
    result = {
        "total_h2": None,
        "total_h2_se": None,
        "categories": [],
    }

    # Total h2
    h2_match = re.search(
        r"Total Observed scale h2:\s*([-\d.]+)\s*\(([\d.]+)\)",
        log_content
    )
    if h2_match:
        result["total_h2"] = float(h2_match.group(1))
        result["total_h2_se"] = float(h2_match.group(2))

    # Parse categories table
    # Category	Prop._SNPs	Prop._h2	Enrichment	...
    lines = log_content.strip().split("\n")

    in_table = False
    for line in lines:
        if "Category" in line and "Prop._SNPs" in line:
            in_table = True
            continue

        if in_table and line.strip():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    result["categories"].append({
                        "category": parts[0],
                        "prop_snps": float(parts[1]),
                        "prop_h2": float(parts[2]),
                        "enrichment": float(parts[3]),
                        "enrichment_se": float(parts[4]) if len(parts) > 4 else None,
                        "enrichment_p": float(parts[5]) if len(parts) > 5 else None,
                    })
                except (ValueError, IndexError):
                    continue

    return result

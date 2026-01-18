# tests/ldsc/test_parsers.py
import pytest


def test_parse_h2_log_extracts_heritability():
    from scripts.ldsc.parsers import parse_h2_log

    log_content = """
*********************************************************************
* LD Score Regression (LDSC)
* Version 1.0.1
*********************************************************************

Total Observed scale h2: 0.1234 (0.0123)
Lambda GC: 1.05
Mean Chi^2: 1.10
Intercept: 1.0012 (0.0078)
"""

    result = parse_h2_log(log_content)

    assert result["h2"] == pytest.approx(0.1234, rel=1e-3)
    assert result["h2_se"] == pytest.approx(0.0123, rel=1e-3)
    assert result["lambda_gc"] == pytest.approx(1.05, rel=1e-2)
    assert result["intercept"] == pytest.approx(1.0012, rel=1e-3)


def test_parse_rg_log_extracts_correlation():
    from scripts.ldsc.parsers import parse_rg_log

    log_content = """
Summary of Genetic Correlation Results
p1	p2	rg	se	z	p
trait1.sumstats.gz	trait2.sumstats.gz	0.456	0.078	5.85	4.9e-09
"""

    result = parse_rg_log(log_content)

    assert len(result["correlations"]) == 1
    assert result["correlations"][0]["rg"] == pytest.approx(0.456, rel=1e-3)
    assert result["correlations"][0]["se"] == pytest.approx(0.078, rel=1e-3)

# SuSiE Fine-Mapping

Run Sum of Single Effects (SuSiE) regression to identify causal variants at GWAS loci.

## Capabilities

- Fine-mapping with GWAS summary statistics (Z-scores or beta/SE)
- Fine-mapping with individual-level genotype + phenotype data
- Automatic credible set construction with configurable coverage
- Detection of multiple independent causal signals
- Per-variant posterior inclusion probabilities (PIP)

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 10 | Maximum number of causal variants |
| `coverage` | 0.95 | Credible set coverage probability |
| `min_abs_corr` | 0.5 | Minimum LD for credible set membership |

## Scripts

- `scripts/susie/run_susie.R` - Core SuSiE execution wrapper
- `scripts/ld/compute_ld.R` - LD matrix computation
- `scripts/ld/fetch_ld_ref.py` - Reference panel retrieval
- `scripts/utils/validate_input.py` - Input validation

## Workflow

1. **Validate inputs**: Run `scripts/utils/validate_input.py` on summary stats + LD matrix
   - Confirm SNP order matches between summary stats and LD
   - Confirm no missing/infinite values in LD matrix
2. **Run SuSiE**: `Rscript scripts/susie/run_susie.R`
3. **Check convergence**: SuSiE reports convergence status — if not converged, increase iterations or reduce L
4. **Review credible sets**: Check set sizes — very large sets (>50 variants) suggest high LD or model misspecification
5. **Visualize**: Generate PIP plot (`visualization/pip_plot.py`) and locus zoom (`visualization/locus_zoom.py`)
6. **Build report**: `visualization/interactive_report.py` for HTML report with all figures

## Limitations

- Requires accurate LD information matching the GWAS population
- Assumes at most L causal variants (may miss signals if L is too low)
- Summary statistics analysis assumes linear model
- Not suitable for highly polygenic traits with many small effects

## Tips

- **Validate your LD matrix** — mismatched LD is the most common source of errors
- **Start with default L=10** — increase only if you expect many independent signals
- **Use matching populations** — LD reference must match your GWAS population
- In high-LD regions, expect larger credible sets and moderate PIPs across correlated variants

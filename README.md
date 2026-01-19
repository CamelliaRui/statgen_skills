# statgen_skills

A Claude custom skill for statistical genetics workflows, starting with SuSiE fine-mapping.

## Features

- **SuSiE Fine-Mapping**: Identify causal variants from GWAS summary statistics or individual-level data
- **Flexible LD Handling**: User-provided, reference panel, or computed from data
- **Publication-Ready Outputs**: CSV, Excel, PNG/PDF figures, interactive HTML reports
- **Adaptive Documentation**: Detailed explanations for newcomers, concise output for experts

## Installation

### R Dependencies

```r
install.packages(c("susieR", "data.table", "jsonlite", "optparse"))
```

### Python Dependencies

```bash
pip install pandas numpy matplotlib plotly openpyxl
```

## Quick Start

### Run SuSiE from Command Line

```bash
Rscript scripts/susie/run_susie.R \
    --input your_sumstats.csv \
    --input-type sumstats \
    --ld your_ld_matrix.npy \
    --n 50000 \
    --output results/ \
    --L 10 \
    --coverage 0.95
```

### Use with Claude

```
"Run SuSiE fine-mapping on my GWAS summary statistics.
Sample size is 50,000. Use EUR 1000G reference LD."
```

## Input Formats

### Summary Statistics

CSV with columns: `SNP`, `CHR`, `BP`, `BETA`, `SE` (or `Z`), and optionally `P`, `A1`, `A2`

### LD Matrix

- NumPy `.npy` file
- Text matrix (whitespace-delimited)
- R `.rds` file

## Outputs

| File | Description |
|------|-------------|
| `susie_results.csv` | Per-variant PIP and credible set membership |
| `susie_summary.json` | Analysis summary |
| `susie_fit.rds` | Full R object for advanced analysis |
| `report.html` | Interactive HTML report |

## Project Structure

```
statgen_skills/
├── SKILL.md                 # Claude skill definition
├── scripts/
│   ├── susie/
│   │   └── run_susie.R      # SuSiE CLI wrapper
│   └── utils/
│       └── validate_input.py
├── visualization/
│   ├── locus_zoom.py
│   ├── pip_plot.py
│   ├── credible_set.py
│   └── interactive_report.py
└── examples/
    ├── example_sumstats.csv
    └── example_workflow.md
```

## Roadmap

- [x] SuSiE fine-mapping
- [x] LDSC (LD Score Regression) - heritability, genetic correlation, s-LDSC
- [x] TWAS simulator
- [x] FUSION TWAS - gene-trait association testing

## License

MIT

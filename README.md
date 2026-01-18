# statgen_skills

A Claude custom skill for statistical genetics workflows, including SuSiE fine-mapping and LDSC heritability analysis.

## Features

- **SuSiE Fine-Mapping**: Identify causal variants from GWAS summary statistics or individual-level data
- **LDSC Analysis**: Estimate SNP heritability, genetic correlations, and partition heritability by annotations
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
pip install pandas numpy matplotlib plotly openpyxl seaborn scipy
pip install git+https://github.com/CBIIT/ldsc.git
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

### Run LDSC from Python

```python
from scripts.ldsc.munge import munge_sumstats
from scripts.ldsc.run_ldsc import estimate_heritability

# Munge summary statistics
munge_sumstats("gwas.csv", "output/munged", n=50000)

# Estimate heritability
result = estimate_heritability(
    "output/munged.sumstats.gz",
    "output/h2",
    population="EUR"
)
```

### Use with Claude

```
"Run SuSiE fine-mapping on my GWAS summary statistics.
Sample size is 50,000. Use EUR 1000G reference LD."

"Estimate the SNP heritability for my height GWAS using EUR reference."

"Calculate genetic correlations between height, BMI, and T2D."
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
| `h2_*.log` | LDSC heritability results |
| `rg_*.log` | Genetic correlation results |

## Project Structure

```
statgen_skills/
├── SKILL.md                 # Claude skill definition
├── scripts/
│   ├── susie/
│   │   └── run_susie.R      # SuSiE CLI wrapper
│   ├── ldsc/
│   │   ├── run_ldsc.py      # LDSC runner (h2, rg, s-LDSC)
│   │   ├── munge.py         # Summary stats preprocessing
│   │   ├── reference_data.py # Reference data management
│   │   └── parsers.py       # Log file parsing
│   └── utils/
│       └── validate_input.py
├── visualization/
│   ├── locus_zoom.py
│   ├── pip_plot.py
│   ├── credible_set.py
│   ├── interactive_report.py
│   └── ldsc_plots.py        # h² bar charts, rg heatmaps
└── examples/
    ├── example_sumstats.csv
    ├── example_workflow.md
    ├── ldsc_example_sumstats.csv
    └── ldsc_tutorial.md
```

## Roadmap

- [x] SuSiE fine-mapping
- [x] LDSC (LD Score Regression) - heritability, genetic correlation, s-LDSC
- [ ] TWAS (Transcriptome-Wide Association Studies)
- [ ] TWAS simulator

## License

MIT

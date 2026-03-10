# scientific-statgen-playbook

A [scientific-software-playbook](https://github.com/mancusolab/scientific-software-playbook)-compatible plugin collection for downstream statistical genetics: GWAS analysis, fine-mapping, heritability, TWAS, and biological validation against public databases.

## How It Fits Together

| Layer | Repository | Focus |
|-------|-----------|-------|
| **Upstream** | [scientific-software-playbook](https://github.com/mancusolab/scientific-software-playbook) | Model design, inference code, structured dev workflow |
| **Downstream** | This repo | GWAS analysis, fine-mapping, heritability, TWAS |
| **Validation** | This repo | Biological validation via OpenTargets, GWAS Catalog, Ensembl |

Install both to get an end-to-end agent-assisted statistical genetics workflow.

## Plugins

### `statgen-analysis`
GWAS analysis workflows: SuSiE fine-mapping, LDSC heritability, TWAS simulation, FUSION TWAS, and publication-ready visualization.

### `statgen-validation`
Biological validation of findings against public databases:
- **OpenTargets** — gene-disease associations, target tractability, genetic constraint
- **GWAS Catalog** — known associations by variant, gene, or trait
- **Ensembl VEP** — variant functional annotation (consequence, SIFT, PolyPhen, CADD)

## Installation

### Claude Code

```bash
# Install as a plugin
git clone https://github.com/CamelliaRui/scientific-statgen-playbook.git ~/.claude/plugins/scientific-statgen-playbook

# Or project-level
git clone https://github.com/CamelliaRui/scientific-statgen-playbook.git .claude/plugins/scientific-statgen-playbook
```

### OpenAI Codex

```bash
git clone https://github.com/CamelliaRui/scientific-statgen-playbook.git .skills/scientific-statgen-playbook
```

### Other Agent Skills-compatible tools

Clone the repo and point your tool to the directory. The plugins follow the [Agent Skills open standard](https://agentskills.io) and the [scientific-software-playbook](https://github.com/mancusolab/scientific-software-playbook) plugin format.

### Install Dependencies

**R** (required for SuSiE and FUSION):

```r
install.packages(c("susieR", "data.table", "jsonlite", "optparse", "glmnet"))
```

**Python** (required for LDSC, TWAS, visualization, validation):

```bash
uv pip install pandas numpy matplotlib plotly openpyxl seaborn scipy scikit-learn requests
```

## Quick Start

Once installed, just ask in natural language:

```
"Estimate the SNP heritability for my height GWAS using EUR reference"
"Run SuSiE on my GWAS summary stats with the provided LD matrix"
"Validate my top fine-mapping variants against GWAS Catalog and OpenTargets"
"Annotate these credible set SNPs with VEP consequences"
"Check if PCSK9 has known associations with LDL cholesterol in OpenTargets"
```

## What's Included

| Plugin | Tool | What it does |
|--------|------|-------------|
| **analysis** | SuSiE | Fine-map causal variants from GWAS summary stats or individual data |
| **analysis** | LDSC | SNP heritability, genetic correlations, partitioned heritability |
| **analysis** | TWAS Simulator | Simulate TWAS for power analysis and methods development |
| **analysis** | FUSION TWAS | Gene-trait associations using GTEx v8 weights (49 tissues) |
| **analysis** | Visualization | Publication-ready plots (locus zoom, Manhattan, PIP, heatmaps) |
| **validation** | OpenTargets | Gene-disease evidence, tractability, genetic constraint |
| **validation** | GWAS Catalog | Known associations by variant, gene, or trait |
| **validation** | Ensembl VEP | Variant annotation (consequence, SIFT, PolyPhen, CADD) |

## Project Structure

```
scientific-statgen-playbook/
├── .claude-plugin/
│   └── marketplace.json            # Plugin marketplace registry
├── plugins/
│   ├── statgen-analysis/           # GWAS analysis plugin
│   │   ├── .claude-plugin/
│   │   │   └── plugin.json
│   │   ├── skills/statgen-analysis/SKILL.md
│   │   ├── scripts/               # SuSiE, LDSC, TWAS, FUSION
│   │   ├── reference/             # Detailed docs
│   │   ├── visualization/         # Publication-ready plots
│   │   ├── examples/              # Example data and tutorials
│   │   └── tests/
│   └── statgen-validation/         # Biological validation plugin
│       ├── .claude-plugin/
│       │   └── plugin.json
│       ├── skills/biological-validation/SKILL.md
│       ├── scripts/               # OpenTargets, GWAS Catalog, Ensembl
│       ├── reference/             # API docs
│       └── tests/
├── AGENTS.md                       # Plugin asset source of truth
├── README.md
└── pytest.ini
```

## Using with the Scientific Software Playbook

These plugins are designed to complement the [scientific-software-playbook](https://github.com/mancusolab/scientific-software-playbook). When both are installed:

1. Use the playbook's `scientific-plan-execute` for model design and structured development
2. Use `statgen-analysis` for downstream GWAS analysis and visualization
3. Use `statgen-validation` to validate findings against public biological databases
4. The playbook's house-style skills (JAX/Equinox, testing) apply to code written in both

## License

MIT

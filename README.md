# statgen-skills

A [scientific-software-playbook](https://github.com/mancusolab/scientific-software-playbook)-compatible plugin for downstream statistical genetics analysis: SuSiE fine-mapping, LDSC heritability, TWAS simulation, FUSION TWAS, and publication-ready visualization.

## How It Fits Together

| Layer | Repository | Focus |
|-------|-----------|-------|
| **Upstream** | [scientific-software-playbook](https://github.com/mancusolab/scientific-software-playbook) | Model design, inference code, structured dev workflow |
| **Downstream** | This repo (`statgen-analysis` plugin) | GWAS analysis, fine-mapping, heritability, TWAS |

Install both to get an end-to-end agent-assisted statistical genetics workflow.

## Installation

### Claude Code

```bash
# Install as a plugin
git clone https://github.com/CamelliaRui/statgen_skills.git ~/.claude/plugins/statgen-skills

# Or project-level
git clone https://github.com/CamelliaRui/statgen_skills.git .claude/plugins/statgen-skills
```

### OpenAI Codex

```bash
git clone https://github.com/CamelliaRui/statgen_skills.git .skills/statgen-skills
```

### Other Agent Skills-compatible tools

Clone the repo and point your tool to the directory. The plugin follows the [Agent Skills open standard](https://agentskills.io) and the [scientific-software-playbook](https://github.com/mancusolab/scientific-software-playbook) plugin format.

### Install Dependencies

**R** (required for SuSiE and FUSION):

```r
install.packages(c("susieR", "data.table", "jsonlite", "optparse", "glmnet"))
```

**Python** (required for LDSC, TWAS, visualization):

```bash
uv pip install pandas numpy matplotlib plotly openpyxl seaborn scipy scikit-learn
```

## Quick Start

Once the plugin is installed, just ask in natural language:

```
"Estimate the SNP heritability for my height GWAS using EUR reference"
"Run SuSiE on my GWAS summary stats with the provided LD matrix"
"Simulate a TWAS with 100 genes, 10 causal, h2_cis=0.1"
"Run TWAS on my schizophrenia GWAS using GTEx brain cortex"
```

## What's Included

| Tool | What it does |
|------|-------------|
| **SuSiE** | Fine-map causal variants from GWAS summary stats or individual data |
| **LDSC** | SNP heritability, genetic correlations, partitioned heritability |
| **TWAS Simulator** | Simulate TWAS for power analysis and methods development |
| **FUSION TWAS** | Gene-trait associations using GTEx v8 weights (49 tissues) |
| **Visualization** | Publication-ready plots (locus zoom, Manhattan, PIP, heatmaps) |

## Project Structure

```
statgen_skills/
├── .claude-plugin/
│   └── marketplace.json            # Plugin marketplace registry
├── plugins/
│   └── statgen-analysis/
│       ├── .claude-plugin/
│       │   └── plugin.json         # Plugin manifest
│       ├── skills/
│       │   └── statgen-analysis/
│       │       └── SKILL.md        # Skill definition (loaded by agent)
│       ├── scripts/                # Implementation code
│       │   ├── susie/              # SuSiE R wrapper
│       │   ├── ldsc/               # LDSC Python runner
│       │   ├── twas/               # TWAS simulator + models
│       │   ├── fusion/             # FUSION TWAS runner
│       │   └── utils/
│       ├── reference/              # Detailed docs (loaded on-demand)
│       ├── visualization/          # Publication-ready plots
│       ├── examples/               # Example data and tutorials
│       └── tests/                  # pytest test suites
├── AGENTS.md                       # Plugin asset source of truth
├── README.md
└── pytest.ini
```

## Using with the Scientific Software Playbook

This plugin is designed to complement the [scientific-software-playbook](https://github.com/mancusolab/scientific-software-playbook). When both are installed:

1. Use the playbook's `scientific-plan-execute` for model design and structured development
2. Use `statgen-analysis` for downstream GWAS analysis and visualization
3. The playbook's house-style skills (JAX/Equinox, testing) apply to code written in both

## License

MIT

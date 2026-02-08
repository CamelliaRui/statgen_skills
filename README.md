# statgen_skills

An [Agent Skill](https://agentskills.io) for statistical genetics workflows: SuSiE fine-mapping, LDSC heritability, TWAS simulation, FUSION TWAS, and JAX/Equinox coding guidelines.

Compatible with any tool that supports the Agent Skills open standard, including **Claude Code**, **OpenAI Codex**, **GitHub Copilot**, **Gemini CLI**, **Cursor**, and more.

## Installation

### Claude Code

```bash
# Personal skill (all projects)
git clone https://github.com/CamelliaRui/statgen_skills.git ~/.claude/skills/statgen-skills

# Or project skill (one project only)
git clone https://github.com/CamelliaRui/statgen_skills.git .claude/skills/statgen-skills

# Or additional directory
git clone https://github.com/CamelliaRui/statgen_skills.git ~/statgen_skills
claude --add-dir ~/statgen_skills
```

### OpenAI Codex

```bash
# Clone into your project's .skills directory
git clone https://github.com/CamelliaRui/statgen_skills.git .skills/statgen-skills
```

### Other Agent Skills-compatible tools

Clone the repo and point your tool to the directory containing `SKILL.md`. The skill follows the [Agent Skills open standard](https://agentskills.io), so any compatible tool will discover it automatically.

### Verify installation

```
> What skills are available?
# Should list "statgen-skills" with its description
```

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

Once the skill is installed, just ask in natural language:

```
"Estimate the SNP heritability for my height GWAS using EUR reference"

"Run SuSiE on my GWAS summary stats with the provided LD matrix"

"Simulate a TWAS with 100 genes, 10 causal, h2_cis=0.1"

"Run TWAS on my schizophrenia GWAS using GTEx brain cortex"
```

The agent automatically loads the skill when your request matches statistical genetics topics. In Claude Code, you can also invoke it directly:

```
/statgen-skills Run LDSC genetic correlation between height and BMI
```

### Tips

- **Provide context**: mention sample size, population ancestry, and analysis goals
- **Specify file paths**: use absolute paths or ensure files are in the working directory
- **Ask for explanations**: the agent can interpret PIPs, credible sets, heritability estimates
- **Iterate**: start with defaults, then refine based on initial results

## What's Included

| Tool | What it does |
|------|-------------|
| **SuSiE** | Fine-map causal variants from GWAS summary stats or individual data |
| **LDSC** | SNP heritability, genetic correlations, partitioned heritability |
| **TWAS Simulator** | Simulate TWAS for power analysis and methods development |
| **FUSION TWAS** | Gene-trait associations using GTEx v8 weights (49 tissues) |
| **JAX/Equinox Guidelines** | Coding rules, checklists, and snippets for numerical computing |

## Project Structure

```
statgen_skills/
├── SKILL.md                    # Skill entry point (loaded by agent)
├── reference/                  # Detailed docs (loaded on-demand)
│   ├── susie.md
│   ├── ldsc.md
│   ├── twas-sim.md
│   ├── fusion.md
│   ├── input-formats.md
│   └── jax-equinox.md
├── scripts/                    # Implementation code
│   ├── susie/                  # SuSiE R wrapper
│   ├── ldsc/                   # LDSC Python runner
│   ├── twas/                   # TWAS simulator + models
│   ├── fusion/                 # FUSION TWAS runner
│   └── utils/
├── visualization/              # Publication-ready plots
├── tests/                      # pytest test suites
└── examples/                   # Example data and tutorials
```

## License

MIT

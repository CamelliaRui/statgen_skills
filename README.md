# statgen_skills

A Claude skill for statistical genetics workflows: SuSiE fine-mapping, LDSC heritability, TWAS simulation, FUSION TWAS, presentation generation, and JAX/Equinox coding guidelines.

## Install as a Claude Code Skill

### Option 1: Personal skill (available in all your projects)

```bash
# Clone into your personal skills directory
git clone https://github.com/CamelliaRui/statgen_skills.git ~/.claude/skills/statgen-skills
```

### Option 2: Project skill (available in one project only)

```bash
# From your project root
git clone https://github.com/CamelliaRui/statgen_skills.git .claude/skills/statgen-skills
```

### Option 3: Additional directory

```bash
# Clone anywhere and point Claude Code to it
git clone https://github.com/CamelliaRui/statgen_skills.git ~/statgen_skills
claude --add-dir ~/statgen_skills
```

After installing, verify the skill is loaded:

```
> What skills are available?
# Claude should list "statgen-skills" with its description
```

### Install Dependencies

**R** (required for SuSiE and FUSION):

```r
install.packages(c("susieR", "data.table", "jsonlite", "optparse", "glmnet"))
```

**Python** (required for LDSC, TWAS, visualization):

```bash
pip install pandas numpy matplotlib plotly openpyxl seaborn scipy scikit-learn python-pptx
```

## Quick Start

Once the skill is installed, just ask Claude in natural language:

```
"Estimate the SNP heritability for my height GWAS using EUR reference"

"Run SuSiE on my GWAS summary stats with the provided LD matrix"

"Simulate a TWAS with 100 genes, 10 causal, h2_cis=0.1"

"Run TWAS on my schizophrenia GWAS using GTEx brain cortex"

"Create a journal club presentation from this paper"
```

Claude automatically loads the skill when your request matches statistical genetics topics. You can also invoke it directly:

```
/statgen-skills Run LDSC genetic correlation between height and BMI
```

### Tips

- **Provide context**: mention sample size, population ancestry, and analysis goals
- **Specify file paths**: use absolute paths or ensure files are in the working directory
- **Ask for explanations**: Claude can interpret PIPs, credible sets, heritability estimates
- **Iterate**: start with defaults, then refine based on initial results

## What's Included

| Tool | What it does |
|------|-------------|
| **SuSiE** | Fine-map causal variants from GWAS summary stats or individual data |
| **LDSC** | SNP heritability, genetic correlations, partitioned heritability |
| **TWAS Simulator** | Simulate TWAS for power analysis and methods development |
| **FUSION TWAS** | Gene-trait associations using GTEx v8 weights (49 tissues) |
| **Presentation Generator** | PPTX slides from research papers (journal club, lab meeting, conference) |
| **JAX/Equinox Guidelines** | Coding rules, checklists, and snippets for numerical computing |

## Project Structure

```
statgen_skills/
├── SKILL.md                    # Skill entry point (loaded by Claude)
├── reference/                  # Detailed docs (loaded on-demand)
│   ├── susie.md
│   ├── ldsc.md
│   ├── twas-sim.md
│   ├── fusion.md
│   ├── presentation.md
│   ├── input-formats.md
│   └── jax-equinox.md
├── scripts/                    # Implementation code
│   ├── susie/                  # SuSiE R wrapper
│   ├── ldsc/                   # LDSC Python runner
│   ├── twas/                   # TWAS simulator + models
│   ├── fusion/                 # FUSION TWAS runner
│   ├── presentation/           # PDF-to-PPTX generator
│   └── utils/
├── visualization/              # Publication-ready plots
├── templates/                  # PPTX templates
├── tests/                      # pytest test suites
└── examples/                   # Example data and tutorials
```

## License

MIT

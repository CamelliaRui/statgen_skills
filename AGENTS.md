## scientific-statgen-playbook (Plugin Format)

This repository hosts one plugin for downstream statistical genetics analysis workflows.

Scope note: this repository hosts one plugin:
1. `statgen-analysis`

Dependency contract:
1. `statgen-analysis` is self-contained for downstream analysis workflows.
2. For upstream model design and structured development workflows, install `scientific-plan-execute` from the [scientific-software-playbook](https://github.com/mancusolab/scientific-software-playbook).
3. When both are installed, the playbook handles model design/inference and `statgen-analysis` handles GWAS analysis, fine-mapping, heritability, and TWAS.

## Plugin Assets (Source Of Truth)

### Skills (`statgen-analysis`)
- `statgen-analysis`: `plugins/statgen-analysis/skills/statgen-analysis/SKILL.md`

### Assets
- Scripts: `plugins/statgen-analysis/scripts/`
- Reference docs: `plugins/statgen-analysis/reference/`
- Visualization: `plugins/statgen-analysis/visualization/`
- Examples: `plugins/statgen-analysis/examples/`
- Tests: `plugins/statgen-analysis/tests/`

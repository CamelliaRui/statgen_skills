# Scientific Presentation Generator - Design Document

**Date:** 2026-01-24
**Status:** Draft

## Overview

Generate publication-style presentations from research papers (PDF) using a template PPTX for visual styling. Supports journal clubs, lab meetings, and conference talks with customizable section structures.

## User Flow

```
1. Upload template PPTX (school logo, colors, headers)
2. Upload paper PDF
3. Choose presentation type:
   (a) Journal Club - others' papers, standard length
   (b) Lab Meeting - own work, ~1hr, detailed
   (c) Conference Talk - 15/30 min, high-level
4. Choose structure:
   (a) Strict IMRAD
   (b) Flexible sections
5. Specify slide counts per section
6. Choose figures: main only vs. include supplementary
7. Choose content style: extractive vs. generative
8. Generate PPTX → upload to Google Slides if desired
```

## Technical Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         SKILL STRUCTURE                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ Template    │    │ Paper       │    │ Content Generator   │  │
│  │ Parser      │    │ Extractor   │    │ (Claude LLM)        │  │
│  │             │    │             │    │                     │  │
│  │ • Colors    │    │ • markitdown│    │ • Summarize sections│  │
│  │ • Fonts     │    │ • Figure    │    │ • Map to template   │  │
│  │ • Logo      │    │   extraction│    │ • Generate bullets  │  │
│  │ • Layout    │    │ • Table     │    │                     │  │
│  └──────┬──────┘    │   parsing   │    └──────────┬──────────┘  │
│         │           └──────┬──────┘               │              │
│         │                  │                      │              │
│         └────────────┬─────┴──────────────────────┘              │
│                      ▼                                           │
│              ┌───────────────┐                                   │
│              │ PPTX Builder  │                                   │
│              │ (python-pptx) │                                   │
│              └───────┬───────┘                                   │
│                      ▼                                           │
│              ┌───────────────┐                                   │
│              │ output.pptx   │ → Upload to Google Slides         │
│              └───────────────┘                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `markitdown` | Extract text/structure from PDF |
| `python-pptx` | Generate PowerPoint files |
| `pymupdf` (fitz) | Extract figures/images from PDF |
| `Pillow` | Image processing |

## Presentation Type Configurations

### Comparison

| Aspect | Journal Club | Lab Meeting | Conference Talk |
|--------|-------------|-------------|-----------------|
| **Duration** | 30-45 min | ~60 min | 15 or 30 min (user picks) |
| **Source** | Others' papers | Own manuscript | Own work |
| **Detail level** | Standard | High (methods, implementation) | High-level, skip technicals |
| **Default slides** | ~15-20 | ~25-35 | ~10-15 or ~20-25 |
| **Figures** | Key results | All relevant | Highlight figures only |

### Default Slide Distributions

**Journal Club (20 slides):**
- Title: 1
- Introduction/Background: 3
- Methods: 3
- Results: 8
- Discussion: 3
- Conclusions: 1
- Questions: 1

**Lab Meeting (30 slides):**
- Title: 1
- Introduction/Background: 4
- Methods: 6 (detailed)
- Results: 12
- Discussion: 4
- Future Directions: 2
- Questions: 1

**Conference Talk - 15 min (12 slides):**
- Title: 1
- Background: 2
- Methods: 1 (brief)
- Results: 5
- Conclusions: 2
- Questions: 1

## File Structure

```
statgen_skills/
├── SKILL.md                      # Update with presentation generator
├── scripts/
│   └── presentation/
│       ├── __init__.py
│       ├── generator.py          # Main orchestrator
│       ├── template_parser.py    # Extract style from template PPTX
│       ├── paper_extractor.py    # Extract content + figures from PDF
│       ├── content_mapper.py     # Map paper sections to slides
│       └── pptx_builder.py       # Generate final PPTX
├── templates/
│   └── presentation/
│       ├── default_template.pptx # Fallback template
│       └── configs/
│           ├── journal_club.yaml
│           ├── lab_meeting.yaml
│           └── conference_talk.yaml
└── tests/
    └── test_presentation.py
```

## Interactive Prompts Flow

When user invokes the skill, Claude asks:

**Step 1:** "What type of presentation?"
- (a) Journal Club
- (b) Lab Meeting
- (c) Conference Talk

**Step 2:** "What structure format?"
- (a) Strict IMRAD
- (b) Flexible sections

If flexible: "Which sections? (e.g., Background, Methods, Results, Future Directions)"

**Step 3:** "How many slides per section?"
- Show defaults based on presentation type
- Let user adjust (e.g., "Introduction: 3, Methods: 3, Results: 8")

**Step 4:** "Which figures to include?"
- (a) Main figures only
- (b) Include supplementary figures

**Step 5:** "Content style for bullet points?"
- (a) Extractive (direct quotes from paper)
- (b) Generative (Claude summarizes)

**Step 6:** "Please upload:"
- Template PPTX (or use default)
- Paper PDF

## Config File Format

Example `journal_club.yaml`:

```yaml
name: Journal Club
duration_minutes: 30-45
detail_level: standard
default_slides:
  title: 1
  introduction: 3
  methods: 3
  results: 8
  discussion: 3
  conclusions: 1
  questions: 1
figure_preference: key_results
```

## Future Extensions

- **URL/DOI input:** Extract paper from PubMed, bioRxiv, or journal websites
- **Google Slides API:** Direct export to Google Slides (requires OAuth setup)
- **Multi-paper presentations:** Combine multiple papers for review talks
- **Citation slides:** Auto-generate reference slides

## Implementation Phases

### Phase 1: Core MVP
- Template PPTX parsing (extract colors, fonts, logo)
- PDF text extraction with markitdown
- Basic PPTX generation with python-pptx
- Journal Club preset only

### Phase 2: Full Feature Set
- Figure extraction from PDF
- All three presentation types
- Flexible section configuration
- Extractive vs. generative content options

### Phase 3: Extensions
- URL/DOI paper fetching
- Google Slides integration
- Multi-paper support

# Presentation Generator

Generate scientific presentations from research papers (PDF) with customizable templates and formats.

## Capabilities

- Extract content and figures from paper PDFs
- Apply custom template styling (logo, colors, fonts)
- Support for Journal Club, Lab Meeting, and Conference Talk formats
- IMRAD or flexible section structures
- Extractive or generative bullet point styles
- Export to PPTX (compatible with Google Slides)

## API Functions

- `generate_presentation(pdf, output, type, ...)` - Generate presentation
- `PresentationGenerator(config)` - Full control over generation
- `load_config(name)` - Load preset configuration

## Presentation Types

| Type | Duration | Detail Level | Default Slides |
|------|----------|--------------|----------------|
| Journal Club | 30-45 min | Standard | ~20 |
| Lab Meeting | ~60 min | High (methods, implementation) | ~30 |
| Conference Talk | 15-30 min | High-level | ~12-25 |

## Scripts

- `scripts/presentation/generator.py` - Main orchestrator with PresentationConfig dataclass
- `scripts/presentation/paper_extractor.py` - PDF content extraction
- `scripts/presentation/content_mapper.py` - Section-to-slide mapping
- `scripts/presentation/pptx_builder.py` - PPTX creation utilities
- `scripts/presentation/template_parser.py` - Template style parsing
- `templates/presentation/configs/journal_club.yaml` - Preset configurations

## Workflow

1. **Choose format**: Select presentation type (journal_club, lab_meeting, conference_talk)
2. **Extract content**: Paper PDF is parsed for text, figures, and section structure
3. **Map content to slides**: Sections are mapped to slides based on presentation type
4. **Generate PPTX**: Slides are built with template styling applied
5. **Review and refine**: Open the PPTX and adjust content, add speaker notes, rearrange slides as needed

## Example

```python
from scripts.presentation import generate_presentation

generate_presentation(
    pdf_path="paper.pdf",
    output_path="presentation.pptx",
    presentation_type="journal_club",
    presenter_name="Your Name",
)
```

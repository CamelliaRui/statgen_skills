"""
Presentation generator module.

Generates scientific presentations from research papers (PDF) using
template PPTX for styling. Supports journal clubs, lab meetings,
and conference talks.
"""

from .generator import (
    PresentationConfig,
    PresentationGenerator,
    generate_presentation,
    load_config,
)

__all__ = [
    "PresentationConfig",
    "PresentationGenerator",
    "generate_presentation",
    "load_config",
]

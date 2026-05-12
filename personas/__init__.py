"""Persona library: stratified sampling from ACS PUMS with ANES-derived issue-position priors.

See docs/adr/0002-persona-library-v1.md for the methodology.
"""

from personas.schema import Persona, PopulationSpec
from personas.sample import sample_personas, K_ANONYMITY_MIN

__all__ = ["Persona", "PopulationSpec", "sample_personas", "K_ANONYMITY_MIN"]

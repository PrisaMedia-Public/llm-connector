"""
Capa de generación: conecta reglas de generación de texto con la llamada al
modelo, incluyendo conteo de tokens, chunking preventivo y resumen jerárquico.
"""

from llm_engines.generation.chunking import split_text
from llm_engines.generation.pipeline import GenerationResult, TextGenerator
from llm_engines.generation.prompt_builder import PromptBuilder
from llm_engines.generation.rules import GenerationRules
from llm_engines.generation.tokens import count_tokens

__all__ = [
    "GenerationResult",
    "GenerationRules",
    "PromptBuilder",
    "TextGenerator",
    "count_tokens",
    "split_text",
]

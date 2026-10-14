"""
llm-engines
===========

Conectores (wrappers) homogéneos para múltiples proveedores de LLM
(OpenAI, Anthropic, Amazon Bedrock y Google Vertex AI / Model Garden) bajo una
interfaz común (`LLMEngine`), más una capa de generación que conecta reglas de
texto con la llamada al modelo (conteo de tokens, chunking y resumen jerárquico).

Los engines (`base`, `openai`, `anthropic`, `bedrock`, `gemini`) se publican
tal cual se usan en producción.

Uso rápido (pipeline completo regla -> modelo):

    from llm_engines import create_engine
    from llm_engines.generation import TextGenerator, GenerationRules

    engine = create_engine("openai", api_key="sk-...")
    generator = TextGenerator(engine)
    result = generator.generate(
        texto_largo,
        model="gpt-4.1-mini",
        instructions="Resume el texto destacando los temas principales.",
        temperature=0.3,
        maxTokens=400,
    )
    print(result.text, result.total_tokens)

Uso directo de un engine (sin pipeline):

    engine = create_engine("openai", api_key="sk-...")
    messages = [{"role": "user", "content": "Hola"}]
    response = engine.generate(messages, "gpt-4.1-mini", temperature=0.3)
    print(engine.extract_text(response))
"""

import importlib

from llm_engines.base import LLMEngine
from llm_engines.factory import create_engine, available_providers, register_engine
from llm_engines.generation import (
    GenerationResult,
    GenerationRules,
    PromptBuilder,
    TextGenerator,
    count_tokens,
    split_text,
)

__version__ = "1.1.0"

# Los engines concretos dependen de SDKs opcionales: se cargan de forma perezosa.
_LAZY_ENGINES = {
    "OpenAIEngine": "llm_engines.openai",
    "AnthropicEngine": "llm_engines.anthropic",
    "BedrockEngine": "llm_engines.bedrock",
    "GeminiEngine": "llm_engines.gemini",
}


def __getattr__(name):
    if name in _LAZY_ENGINES:
        module = importlib.import_module(_LAZY_ENGINES[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LLMEngine",
    "OpenAIEngine",
    "AnthropicEngine",
    "BedrockEngine",
    "GeminiEngine",
    "create_engine",
    "available_providers",
    "register_engine",
    "GenerationResult",
    "GenerationRules",
    "PromptBuilder",
    "TextGenerator",
    "count_tokens",
    "split_text",
]

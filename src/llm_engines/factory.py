"""Factoría para instanciar engines por nombre de proveedor.

Los engines se importan bajo demanda (lazy) para que cada SDK de proveedor sea
una dependencia opcional: solo se importa el del proveedor que se usa.
"""

import importlib

from llm_engines.base import LLMEngine

# provider -> "modulo:Clase" (o la propia clase una vez importada)
_REGISTRY = {
    "openai": "llm_engines.openai:OpenAIEngine",
    "anthropic": "llm_engines.anthropic:AnthropicEngine",
    "bedrock": "llm_engines.bedrock:BedrockEngine",
    "gemini": "llm_engines.gemini:GeminiEngine",
}

_ALIASES = {
    "google": "gemini",
    "vertex": "gemini",
    "vertexai": "gemini",
    "aws": "bedrock",
}


def available_providers() -> list:
    """Devuelve la lista de identificadores de proveedor soportados."""
    return sorted(_REGISTRY)


def register_engine(provider: str, engine) -> None:
    """Registra un engine propio (clase o ruta 'modulo:Clase') o sustituye uno existente."""
    _REGISTRY[provider.lower()] = engine


def _resolve(provider: str):
    key = provider.lower().strip()
    key = _ALIASES.get(key, key)
    if key not in _REGISTRY:
        raise ValueError(
            f"Proveedor '{provider}' no soportado. "
            f"Disponibles: {', '.join(available_providers())}"
        )
    engine = _REGISTRY[key]
    if isinstance(engine, str):
        module_path, class_name = engine.split(":")
        engine = getattr(importlib.import_module(module_path), class_name)
        _REGISTRY[key] = engine
    return engine


def create_engine(provider: str, **kwargs) -> LLMEngine:
    """
    Crea una instancia de engine para el proveedor indicado. Los `kwargs` se
    pasan tal cual al constructor del engine. Ejemplos:

        create_engine("openai", api_key="sk-...")
        create_engine("anthropic", api_key="sk-ant-...")
        create_engine("bedrock", region_name="eu-west-1")
        create_engine("gemini", credentials_json={...}, project="my-gcp-project")

    Raises:
        ValueError: si el proveedor no está registrado.
        ImportError: si el SDK del proveedor no está instalado.
    """
    return _resolve(provider)(**kwargs)

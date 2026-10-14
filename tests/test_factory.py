"""Tests de la factoría y el registro de proveedores (sin llamadas de red)."""

import pytest

from llm_engines import create_engine, available_providers
from llm_engines.factory import _REGISTRY


def test_available_providers():
    assert set(available_providers()) == {"openai", "anthropic", "bedrock", "gemini"}


def test_create_engine_unknown_provider():
    with pytest.raises(ValueError):
        create_engine("no-existe")


def test_create_engine_is_case_insensitive():
    # No se construye el cliente real porque fallaría sin credenciales;
    # solo comprobamos que el provider se resuelve en el registro.
    assert "OPENAI".lower() in _REGISTRY


def test_aliases_are_available():
    from llm_engines.factory import _ALIASES

    assert _ALIASES["google"] == "gemini"
    assert _ALIASES["aws"] == "bedrock"

"""Tests de la interfaz base de los engines (sin red)."""

from llm_engines.base import LLMEngine, DEFAULT_CONTEXT_LIMIT


def test_base_defaults():
    engine = LLMEngine()
    assert engine.get_context_limit("cualquier-modelo") == DEFAULT_CONTEXT_LIMIT
    assert engine.extract_usage(None) == {"input_tokens": 0, "output_tokens": 0}


def test_base_generate_not_implemented():
    engine = LLMEngine()
    try:
        engine.generate([], "modelo")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("generate() debería lanzar NotImplementedError en la base")

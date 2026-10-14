"""Tests del pipeline de generación con un engine simulado (sin red)."""

import pytest

from llm_engines.base import LLMEngine
from llm_engines.generation import (
    GenerationRules,
    PromptBuilder,
    TextGenerator,
    count_tokens,
    split_text,
)


class FakeEngine(LLMEngine):
    """Engine de mentira: devuelve un texto fijo y registra las llamadas."""

    provider = "fake"

    def __init__(self, context_limit=1000):
        super().__init__()
        self._context_limit = context_limit
        self.calls = []

    def generate(self, model_input, model_name, **kwargs):
        self.calls.append(model_input)
        return {"text": "resumen simulado", "in": 10, "out": 5}

    def extract_text(self, response):
        return response["text"]

    def extract_usage(self, response):
        return {"input_tokens": response["in"], "output_tokens": response["out"]}

    def get_context_limit(self, model_name):
        return self._context_limit


def test_count_tokens_fallback_is_positive():
    assert count_tokens("hola mundo", "modelo-desconocido") > 0


def test_split_text_respects_max_chars():
    text = "párrafo uno.\n\n" + ("x" * 5000)
    chunks = split_text(text, max_chars=1000, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_pipeline_single_pass_no_chunking():
    engine = FakeEngine(context_limit=100000)
    generator = TextGenerator(engine)
    result = generator.generate(
        "Un texto corto.",
        model="modelo-x",
        instructions="Resume.",
    )
    assert result.text == "resumen simulado"
    assert result.chunks == 1
    assert result.total_tokens == 15
    assert len(engine.calls) == 1  # solo la llamada final


def test_pipeline_applies_chunking_on_long_text():
    # Contexto pequeño -> presupuesto pequeño -> fuerza chunking.
    engine = FakeEngine(context_limit=400)
    generator = TextGenerator(engine, chunk_overlap_chars=50, min_chunk_chars=100)
    long_text = ("Frase de relleno. " * 500)
    result = generator.generate(
        long_text,
        model="modelo-x",
        instructions="Resume.",
    )
    assert result.chunks > 1
    # Una llamada por fragmento + la final.
    assert len(engine.calls) == result.chunks + 1


def test_prompt_builder_final_messages_structure():
    rules = GenerationRules(
        system_prompt="Eres un asistente.",
        format_rules={"short": "80-120 palabras."},
    )
    builder = PromptBuilder(rules)
    messages = builder.build_final_messages(
        "TEXTO", instructions="Resume.", output_type="short", context={"title": "T"}
    )
    assert messages[0] == {"role": "system", "content": "Eres un asistente."}
    content = messages[1]["content"]
    assert "INSTRUCCIONES:" in content
    assert "short" in content
    assert "Título" not in content  # context_labels vacío -> clave original
    assert "- title: T" in content


def test_prompt_builder_unknown_output_type():
    builder = PromptBuilder(GenerationRules(format_rules={"a": "x"}))
    with pytest.raises(ValueError):
        builder.build_final_messages("t", instructions="i", output_type="desconocido")

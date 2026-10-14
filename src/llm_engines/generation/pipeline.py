"""Orquestador del pipeline: conecta una regla de generación con la llamada al modelo.

Reproduce la lógica del pipeline original (`summarize_article` de `llm.py`):

  1. Calcula el presupuesto de tokens disponible para el texto a partir de la
     ventana de contexto del modelo, el prompt, la salida y unos márgenes.
  2. Si el texto no cabe, lo divide en fragmentos (chunking preventivo) y resume
     cada uno; si un fragmento sigue sin caber, lo subdivide.
  3. Genera la salida final a partir del texto (o de la unión de resúmenes).

Funciona con cualquier engine de `llm_engines` sin modificarlo: solo usa su API
pública (`generate`, `extract_text`, `extract_usage`, `get_context_limit`).
"""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from llm_engines.base import LLMEngine
from llm_engines.generation.chunking import split_text
from llm_engines.generation.prompt_builder import PromptBuilder
from llm_engines.generation.rules import GenerationRules
from llm_engines.generation.tokens import CHARS_PER_TOKEN, count_tokens

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    word_count: int
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    chunks: int
    prompt_version: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {**asdict(self), "total_tokens": self.total_tokens}


class TextGenerator:
    """
    Conecta las reglas de generación con un engine, aplicando chunking y resumen
    jerárquico cuando el texto no cabe en el contexto del modelo.

    Ejemplo:
        from llm_engines import create_engine
        from llm_engines.generation import TextGenerator

        engine = create_engine("openai", api_key="sk-...")
        generator = TextGenerator(engine)
        result = generator.generate(
            article, model="gpt-4.1-mini",
            instructions="Resume el artículo para un lector no especializado.",
            temperature=0.3, maxTokens=400,
        )
        print(result.text, result.total_tokens)
    """

    def __init__(
        self,
        engine: LLMEngine,
        rules: Optional[GenerationRules] = None,
        *,
        prompt_builder: Optional[PromptBuilder] = None,
        safety_margin_tokens: int = 200,
        reasoning_ratio: float = 0.25,
        chunk_overlap_chars: int = 200,
        min_chunk_chars: int = 500,
    ):
        self.engine = engine
        self.prompt_builder = prompt_builder or PromptBuilder(rules)
        self.safety_margin_tokens = safety_margin_tokens
        self.reasoning_ratio = reasoning_ratio
        self.chunk_overlap_chars = chunk_overlap_chars
        self.min_chunk_chars = min_chunk_chars

    @property
    def rules(self) -> GenerationRules:
        return self.prompt_builder.rules

    def generate(
        self,
        text: str,
        *,
        model: str,
        instructions: str,
        system_prompt: Optional[str] = None,
        output_type: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        output_format: str = "",
        **model_params: Any,
    ) -> GenerationResult:
        """
        Genera texto a partir de `text` aplicando las reglas.

        `model_params` son los hiperparámetros normalizados (`temperature`,
        `topP`, `maxTokens`); el engine ignora los que no soporte.
        """
        final_kwargs = dict(
            instructions=instructions,
            system_prompt=system_prompt,
            output_type=output_type,
            context=context,
            output_format=output_format,
        )

        prompt_without_text = self.prompt_builder.build_final_messages("", **final_kwargs)
        budget = self._text_token_budget(model, prompt_without_text, model_params)

        if count_tokens(text, model) >= budget:
            logger.info("Texto demasiado largo para el contexto de '%s', aplicando chunking", model)
            chunks = split_text(text, int(budget * CHARS_PER_TOKEN), self.chunk_overlap_chars)
        else:
            chunks = [text]

        usage = {"input_tokens": 0, "output_tokens": 0}
        if len(chunks) > 1:
            input_text = "\n\n".join(self._summarize_chunks(chunks, model, model_params, usage))
        else:
            input_text = text

        messages = self.prompt_builder.build_final_messages(input_text, **final_kwargs)
        response = self.engine.generate(messages, model, **model_params)
        if response is None:
            raise RuntimeError("El engine no devolvió respuesta (reintentos agotados).")

        output = self.engine.extract_text(response)
        self._add_usage(usage, response)

        logger.info(json.dumps({
            "event": "generation_completed",
            "provider": self._provider(),
            "model": model,
            "chunks": len(chunks),
            "input_chars": len(input_text),
            **usage,
        }))

        return GenerationResult(
            text=output,
            word_count=len(output.split()),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            model=model,
            provider=self._provider(),
            chunks=len(chunks),
            prompt_version=self.rules.prompt_version,
        )

    # ------------------------------------------------------------------ internos

    def _provider(self) -> str:
        return getattr(self.engine, "provider", type(self.engine).__name__)

    def _text_token_budget(self, model: str, prompt_messages: list, model_params: Mapping[str, Any]) -> int:
        """Tokens disponibles para el texto, descontando prompt, salida y márgenes."""
        context_limit = self.engine.get_context_limit(model)
        prompt_tokens = sum(count_tokens(m["content"], model) for m in prompt_messages)
        budget = int(
            context_limit
            - prompt_tokens
            - self.safety_margin_tokens
            - self.reasoning_ratio * context_limit
            - (model_params.get("maxTokens") or 0)
        )
        if budget <= 0:
            raise ValueError(
                f"El modelo {model} no tiene contexto suficiente ({context_limit} tokens) para el prompt "
                f"y la salida solicitada. Reduce la longitud de las instrucciones o maxTokens."
            )
        return budget

    def _summarize_chunks(self, chunks: list, model: str, model_params: Mapping[str, Any], usage: dict) -> list:
        summaries = []
        pending = list(chunks)
        while pending:
            chunk = pending.pop(0)
            response = self.engine.generate(self.prompt_builder.build_chunk_messages(chunk), model, **model_params)
            if response is None:
                # Fallback: si el fragmento es grande, subdividir y reintentar.
                if len(chunk) >= 2 * self.min_chunk_chars:
                    logger.info("Un fragmento no se pudo resumir, subdividiendo")
                    middle = len(chunk) // 2
                    pending = [chunk[:middle], chunk[middle:]] + pending
                    continue
                raise RuntimeError("No se pudo resumir un fragmento (reintentos agotados).")

            summaries.append(self.engine.extract_text(response))
            self._add_usage(usage, response)
        if not summaries:
            raise RuntimeError("Todos los fragmentos fallaron.")
        return summaries

    def _add_usage(self, usage: dict, response: Any) -> None:
        response_usage = self.engine.extract_usage(response)
        usage["input_tokens"] += response_usage["input_tokens"]
        usage["output_tokens"] += response_usage["output_tokens"]

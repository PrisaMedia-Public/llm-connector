from anthropic import RateLimitError, APIError, Anthropic
import time

from llm_engines.constants import MAX_RETRIES, BACKOFF_FACTOR
from llm_engines.base import LLMEngine

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class AnthropicEngine(LLMEngine):
    # Anthropic no expone el límite de contexto via API, se mantiene un diccionario.
    # Referencia: https://docs.anthropic.com/en/docs/about-claude/models
    CONTEXT_LIMITS = {
        "claude-sonnet-4-20250514": 200000,
        "claude-sonnet-4-6": 200000,
        "claude-opus-4-0-20250514": 200000,
        "claude-opus-4-6": 200000,
        "claude-opus-4-7": 200000,
        "claude-opus-4-8": 200000,
        "claude-sonnet-5": 200000,
        "claude-haiku-4-5": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
    }

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.param_map = {
            "topP": "top_p",
            "maxTokens": "max_tokens",
            "temperature": "temperature",
        }
        self.unsupported_by_model = {
            "claude-opus-4-6": {"temperature", "top_p"},
            "claude-opus-4-7": {"temperature", "top_p"},
            "claude-opus-4-8": {"temperature", "top_p"},
            "claude-sonnet-5": {"temperature", "top_p"},
        }

    def get_context_limit(self, model_name: str) -> int:
        if model_name in self.CONTEXT_LIMITS:
            return self.CONTEXT_LIMITS[model_name]
        # Buscar coincidencia parcial (ej: "claude-sonnet-4" matchea "claude-sonnet-4-20250514")
        for key, limit in self.CONTEXT_LIMITS.items():
            if model_name in key or key in model_name:
                logger.info(f"Coincidencia parcial: '{model_name}' -> '{key}' ({limit} tokens)")
                return limit
        logger.warning(f"Modelo Anthropic '{model_name}' no encontrado en CONTEXT_LIMITS, usando fallback")
        return super().get_context_limit(model_name)

    def build_engine_args(self, model_name, kwargs):
        """
        Convierte los parámetros de entrada a los específicos de la API de Anthropic,
        y filtra los que no son soportados por el modelo.
        """
        args = {
            api: kwargs[k]
            for k, api in self.param_map.items()
            if k in kwargs and kwargs[k] is not None
        }

        # Filter out unsupported params for specific models
        unsupported = self.unsupported_by_model.get(model_name, set())
        args = {k: v for k, v in args.items() if k not in unsupported}

        # Anthropic does not allow both temperature and top_p at the same time
        if "temperature" in args and "top_p" in args:
            del args["top_p"]

        return args

    def generate(self, model_input, model_name: str, **kwargs):
        """
        Llamada al LLM con manejo de errores y retries exponenciales.
        """
        retry = 0
        wait = 1
        engine_args = self.build_engine_args(model_name, kwargs)

        # Separate system message from conversation messages
        system = None
        messages = []
        for msg in model_input:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                messages.append({"role": msg["role"], "content": msg["content"]})

        if system:
            engine_args["system"] = system

        while retry < MAX_RETRIES:
            try:
                response = self.client.messages.create(
                    model=model_name,
                    messages=messages,
                    **engine_args
                )
                return response
            except (RateLimitError, APIError) as e:
                logging.error(f"Error llamando al LLM: {str(e)}")
                time.sleep(wait)
                wait *= BACKOFF_FACTOR
                retry += 1

        # fallback si falla tras todos los retries
        return None

    def extract_text(self, response):
        text = response.content[0].text.strip()
        # Strip markdown code fences if present (e.g. ```json ... ```)
        if text.startswith("```"):
            # Remove opening fence (with optional language tag)
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            # Remove closing fence
            if text.endswith("```"):
                text = text[:-3].strip()
        return text

    def extract_usage(self, response):
        return {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }

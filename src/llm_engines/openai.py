from openai import OpenAI, RateLimitError, APIError, OpenAIError
import time

from llm_engines.constants import MAX_RETRIES, BACKOFF_FACTOR
from llm_engines.base import LLMEngine

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class OpenAIEngine(LLMEngine):
    CONTEXT_LIMITS = {
        "gpt-4.1-mini": 1000000,
        "gpt-4.1": 1000000,
        "gpt-3.5-turbo": 16000,
        "gpt-5": 400000,
        "gpt-5.4": 1000000,
        "gpt-5.4-mini": 1000000,
        "gpt-5.4-nano": 1000000,
        "o3": 200000,
        "o4-mini": 200000,
    }

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.param_map = {
            "topP": "top_p",
            "maxTokens": "max_output_tokens",
            "temperature": "temperature",
        }
        self.unsupported_by_model = {
            "o3": {"temperature", "top_p"},
            "o4-mini": {"temperature", "top_p"},
            "gpt-5": {"temperature", "top_p"},
            "gpt-5.4": {"temperature", "top_p"},
            "gpt-5.4-mini": {"temperature", "top_p"},
            "gpt-5.4-nano": {"temperature", "top_p"},
        }

    def get_context_limit(self, model_name: str) -> int:
        if model_name in self.CONTEXT_LIMITS:
            return self.CONTEXT_LIMITS[model_name]
        logger.warning(f"Modelo OpenAI '{model_name}' no encontrado en CONTEXT_LIMITS, usando fallback")
        return super().get_context_limit(model_name)

    def build_engine_args(self, model_name, kwargs):
        """
        Convierte los parámetros de entrada a los específicos de la API de OpenAI,
        y filtra los que no son soportados por el modelo.
        """
        args = {
            api: kwargs[k]
            for k, api in self.param_map.items()
            if k in kwargs and kwargs[k] is not None
        }

        unsupported = self.unsupported_by_model.get(model_name, set())
        return {k: v for k, v in args.items() if k not in unsupported}

    def generate(self, model_input, model_name: str, **kwargs):
        """
        Llamada al LLM con manejo de errores y retries exponenciales.
        """
        retry = 0
        wait = 1
        engine_args = self.build_engine_args(model_name, kwargs)

        while retry < MAX_RETRIES:
            try:
                response = self.client.responses.create(
                    model=model_name,
                    input=model_input,
                    **engine_args
                )
                return response
            except (RateLimitError, APIError) as e:
                logging.error(f"Error llamando al LLM: {str(e)}")
                time.sleep(wait)
                wait *= BACKOFF_FACTOR
                retry += 1
            except OpenAIError:
                raise

        # fallback si falla tras todos los retries
        return None

    def extract_text(self, response):
        return response.output_text

    def extract_usage(self, response):
        return {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0)
        }

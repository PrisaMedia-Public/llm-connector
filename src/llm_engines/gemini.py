from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
import time

from llm_engines.constants import MAX_RETRIES, BACKOFF_FACTOR
from llm_engines.base import LLMEngine

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class GeminiEngine(LLMEngine):
    """
    Engine for Google Model Garden (Vertex AI) supporting Gemini and Anthropic models.
    Authentication is done via a service account JSON.
    """

    # Diccionario por defecto cuando la API no devuelve input_token_limit.
    # Referencia: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
    CONTEXT_LIMITS = {
        "gemini-2.5-pro": 1048576,
        "gemini-2.5-flash": 1048576,
        "gemini-2.5-flash-lite": 1048576,
        "gemini-2.0-flash": 1048576,
        "gemini-2.0-flash-lite": 1048576,
        "gemini-1.5-pro": 2097152,
        "gemini-1.5-flash": 1048576,
        # Modelos Anthropic servidos via Model Garden
        "claude-sonnet-4-20250514": 200000,
        "claude-sonnet-4-6": 200000,
        "claude-opus-4-0-20250514": 200000,
        "claude-opus-4-6": 200000,
        "claude-haiku-4-5": 200000,
        "claude-3-5-sonnet-v2": 200000,
        "claude-3-5-haiku": 200000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
    }

    def __init__(self, credentials_json: dict, project: str, location: str = "us-central1"):
        """
        Initialize Model Garden client with service account credentials.

        Args:
            credentials_json: Service account JSON credentials as dict.
            project: GCP project ID.
            location: GCP region (default: us-central1).
        """
        credentials = service_account.Credentials.from_service_account_info(
            credentials_json,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials
        )

        self.param_map = {
            "topP": "top_p",
            "maxTokens": "max_output_tokens",
            "temperature": "temperature",
        }

        # Modelos Claude reasoning que no soportan parámetros de muestreo
        self.unsupported_by_model = {
            "claude-opus-4-6": {"temperature", "top_p"},
            "claude-opus-4-7": {"temperature", "top_p"},
            "claude-opus-4-8": {"temperature", "top_p"},
            "claude-sonnet-5": {"temperature", "top_p"},
        }

        # Cache para evitar llamadas repetidas a la API de modelos
        self._context_limit_cache: dict[str, int] = {}

    def get_context_limit(self, model_name: str) -> int:
        """
        Consulta dinámicamente el límite de contexto del modelo via la API de Google.
        Usa client.models.get() que devuelve input_token_limit.
        """
        if model_name in self._context_limit_cache:
            return self._context_limit_cache[model_name]

        try:
            formatted_model = self._format_model_name(model_name)
            model_info = self.client.models.get(model=formatted_model)
            limit = getattr(model_info, "input_token_limit", None)
            if limit:
                logger.info(f"Límite de contexto obtenido via API para '{model_name}': {limit} tokens")
                self._context_limit_cache[model_name] = limit
                return limit
            else:
                logger.warning(f"input_token_limit no disponible via API para '{model_name}', buscando en diccionario local")
                return self._fallback_context_limit(model_name)
        except Exception as e:
            logger.warning(f"Error obteniendo límite de contexto para '{model_name}': {e}. Buscando en diccionario local.")
            return self._fallback_context_limit(model_name)

    def _fallback_context_limit(self, model_name: str) -> int:
        """
        Busca el límite de contexto en el diccionario local CONTEXT_LIMITS.
        Soporta match exacto y parcial antes de caer al fallback base.
        """
        if model_name in self.CONTEXT_LIMITS:
            logger.info(f"Límite de contexto (dict local) para '{model_name}': {self.CONTEXT_LIMITS[model_name]} tokens")
            return self.CONTEXT_LIMITS[model_name]

        # Coincidencia parcial (ej: "gemini-2.5-pro-preview-06-05" matchea "gemini-2.5-pro")
        for key, limit in self.CONTEXT_LIMITS.items():
            if key in model_name or model_name in key:
                logger.info(f"Coincidencia parcial Gemini: '{model_name}' -> '{key}' ({limit} tokens)")
                return limit

        logger.warning(f"Modelo '{model_name}' no encontrado en diccionario local, usando fallback base")
        return super().get_context_limit(model_name)

    def _is_anthropic_model(self, model_name: str) -> bool:
        """Check if the model is an Anthropic model on Model Garden."""
        return "claude" in model_name.lower() or "anthropic" in model_name.lower()

    def _format_model_name(self, model_name: str) -> str:
        """
        Format model name for Model Garden API.
        Anthropic models should use the format: publishers/anthropic/models/<model-name>
        Gemini models are used as-is.
        """
        if self._is_anthropic_model(model_name):
            if not model_name.startswith("publishers/"):
                return f"publishers/anthropic/models/{model_name}"
        return model_name

    def _format_contents(self, model_input: list) -> tuple:
        """
        Format input messages separating system instruction from conversation messages.
        Converts OpenAI-style message dicts to google-genai Content objects.

        Returns:
            tuple: (system_instruction, contents)
        """
        system_instruction = None
        contents = []

        for msg in model_input:
            role = msg["role"]
            text = msg["content"]

            if role == "system":
                system_instruction = text
            else:
                # Map 'assistant' role to 'model' for Gemini/Model Garden API
                mapped_role = "model" if role == "assistant" else role
                contents.append(
                    types.Content(
                        role=mapped_role,
                        parts=[types.Part.from_text(text=text)]
                    )
                )

        return system_instruction, contents

    def build_engine_args(self, model_name, kwargs):
        """
        Convierte los parámetros de entrada a los específicos de la API de Model Garden,
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

        # Anthropic no permite temperature y top_p a la vez (modelos Claude servidos via Model Garden)
        if self._is_anthropic_model(model_name) and "temperature" in args and "top_p" in args:
            del args["top_p"]

        return args

    def generate(self, model_input, model_name: str, **kwargs):
        """
        Llamada al LLM con manejo de errores y retries exponenciales.
        Soporta tanto modelos Gemini como Anthropic (Claude) a través de Vertex AI Model Garden.
        """
        retry = 0
        wait = 1
        engine_args = self.build_engine_args(model_name, kwargs)
        formatted_model = self._format_model_name(model_name)

        system_instruction, contents = self._format_contents(model_input)

        config = types.GenerateContentConfig(**engine_args)
        if system_instruction:
            config.system_instruction = system_instruction

        while retry < MAX_RETRIES:
            try:
                response = self.client.models.generate_content(
                    model=formatted_model,
                    contents=contents,
                    config=config
                )
                return response
            except (ResourceExhausted, GoogleAPIError) as e:
                logging.error(f"Error llamando al LLM: {str(e)}")
                time.sleep(wait)
                wait *= BACKOFF_FACTOR
                retry += 1

        # fallback si falla tras todos los retries
        return None

    def extract_text(self, response):
        text = response.text.strip()
        # Strip markdown code fences if present (e.g. ```json ... ```)
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()
        return text

    def extract_usage(self, response):
        usage = getattr(response, "usage_metadata", None)
        if usage:
            return {
                "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0) or 0
            }
        return {
            "input_tokens": 0,
            "output_tokens": 0
        }

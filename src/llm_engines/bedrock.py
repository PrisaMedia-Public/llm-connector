import boto3
import json
import time

from botocore.exceptions import ClientError

from llm_engines.constants import MAX_RETRIES, BACKOFF_FACTOR
from llm_engines.base import LLMEngine

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class BedrockEngine(LLMEngine):
    """
    Engine para modelos de Anthropic (Claude) desplegados en Amazon Bedrock.
    Utiliza la API Converse de Bedrock para invocar los modelos.
    """

    def __init__(self, region_name: str = "eu-west-1", profile_name: str = None):
        """
        Inicializa el cliente de Bedrock Runtime.

        Args:
            region_name: Región de AWS donde está disponible el modelo (default: eu-west-1).
            profile_name: Perfil de AWS opcional (útil para desarrollo local).
        """
        session_kwargs = {"region_name": region_name}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime")
        self.region_name = region_name

        self.param_map = {
            "topP": "topP",
            "maxTokens": "maxTokens",
            "temperature": "temperature",
        }

        # Modelos Claude reasoning que no soportan parámetros de muestreo
        self.unsupported_by_model = {
            "anthropic.claude-opus-4-6-v1:0": {"temperature", "topP"},
            "anthropic.claude-opus-4-7-v1:0": {"temperature", "topP"},
            "anthropic.claude-opus-4-8-v1:0": {"temperature", "topP"},
            "anthropic.claude-sonnet-5-v1:0": {"temperature", "topP"},
        }

        # Mapeo de prefijo de región para inference profiles de Bedrock
        self.region_prefix_map = {
            "us-east-1": "us",
            "us-west-2": "us",
            "eu-west-1": "eu",
            "eu-west-3": "eu",
            "eu-central-1": "eu",
            "ap-northeast-1": "ap",
            "ap-southeast-1": "ap",
            "ap-southeast-2": "ap",
        }

        # Bedrock no expone límites de contexto via API de forma fiable.
        # Referencia: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
        self.CONTEXT_LIMITS = {
            "anthropic.claude-sonnet-4-20250514-v1:0": 200000,
            "anthropic.claude-sonnet-4-6-v1:0": 200000,
            "anthropic.claude-opus-4-0-20250514-v1:0": 200000,
            "anthropic.claude-opus-4-6-v1:0": 200000,
            "anthropic.claude-haiku-4-5-v1:0": 200000,
            "anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
            "anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
            "anthropic.claude-3-opus-20240229-v1:0": 200000,
            "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
            "anthropic.claude-3-haiku-20240307-v1:0": 200000,
            "amazon.nova-pro-v1:0": 300000,
            "amazon.nova-lite-v1:0": 300000,
            "amazon.nova-micro-v1:0": 128000,
        }

    def get_context_limit(self, model_name: str) -> int:
        # Intentar match exacto (con o sin prefijo regional)
        clean_name = model_name
        for prefix in ("us.", "eu.", "ap."):
            if model_name.startswith(prefix):
                clean_name = model_name[len(prefix):]
                break

        if clean_name in self.CONTEXT_LIMITS:
            return self.CONTEXT_LIMITS[clean_name]

        # Buscar coincidencia parcial
        for key, limit in self.CONTEXT_LIMITS.items():
            if clean_name in key or key in clean_name:
                logger.info(f"Coincidencia parcial Bedrock: '{model_name}' -> '{key}' ({limit} tokens)")
                return limit

        logger.warning(f"Modelo Bedrock '{model_name}' no encontrado en CONTEXT_LIMITS, usando fallback")
        return super().get_context_limit(model_name)

    def _get_inference_profile_id(self, model_name: str) -> str:
        """
        Genera el ID de inference profile cross-region añadiendo el prefijo regional.

        Ejemplo:
            'anthropic.claude-sonnet-4-20250514-v1:0' -> 'eu.anthropic.claude-sonnet-4-20250514-v1:0'
        """
        region_prefix = self.region_prefix_map.get(self.region_name, "us")
        return f"{region_prefix}.{model_name}"

    def _resolve_model_id(self, model_name: str) -> str:
        """
        Determina el model ID correcto para la API Converse.
        
        - Si ya tiene prefijo regional (us., eu., ap.) o es un ARN, se devuelve tal cual.
        - Si no, se devuelve el model_name sin modificar (se intentará primero directo,
          y si falla se reintentará con inference profile en generate()).
        """
        if model_name.startswith("arn:"):
            return model_name

        known_prefixes = ("us.", "eu.", "ap.")
        if model_name.startswith(known_prefixes):
            return model_name

        return model_name

    def _format_contents(self, model_input: list) -> tuple:
        """
        Convierte la lista de mensajes (formato OpenAI-style) al formato
        esperado por la API Converse de Bedrock.

        Returns:
            tuple: (system_prompts, messages) donde system_prompts es una lista
                   de bloques de sistema y messages es la lista de mensajes de conversación.
        """
        system_prompts = []
        messages = []

        for msg in model_input:
            role = msg["role"]
            text = msg["content"]

            if role == "system":
                system_prompts.append({"text": text})
            else:
                messages.append({
                    "role": role,
                    "content": [{"text": text}]
                })

        return system_prompts, messages

    def build_engine_args(self, model_name, kwargs):
        """
        Convierte los parámetros de entrada a los específicos de la API Converse de Bedrock,
        y filtra los que no son soportados.
        """
        args = {
            api: kwargs[k]
            for k, api in self.param_map.items()
            if k in kwargs and kwargs[k] is not None
        }

        # Filter out unsupported params for specific models
        unsupported = self.unsupported_by_model.get(model_name, set())
        args = {k: v for k, v in args.items() if k not in unsupported}

        # Anthropic no permite temperature y topP a la vez
        if "temperature" in args and "topP" in args:
            del args["topP"]

        return args

    def generate(self, model_input, model_name: str, **kwargs):
        """
        Llamada al LLM con manejo de errores y retries exponenciales.
        Utiliza la API Converse de Bedrock para modelos Anthropic Claude.

        Intenta primero con el model ID directo. Si Bedrock devuelve un
        ValidationException indicando que se necesita un inference profile,
        reintenta automáticamente con el ID de inference profile cross-region.
        """
        retry = 0
        wait = 1
        engine_args = self.build_engine_args(model_name, kwargs)

        system_prompts, messages = self._format_contents(model_input)
        resolved_model = self._resolve_model_id(model_name)

        inference_config = {}
        if "maxTokens" in engine_args:
            inference_config["maxTokens"] = engine_args["maxTokens"]
        if "temperature" in engine_args:
            inference_config["temperature"] = engine_args["temperature"]
        if "topP" in engine_args:
            inference_config["topP"] = engine_args["topP"]

        request_params = {
            "modelId": resolved_model,
            "messages": messages,
        }

        if system_prompts:
            request_params["system"] = system_prompts

        if inference_config:
            request_params["inferenceConfig"] = inference_config

        used_inference_profile = False

        while retry < MAX_RETRIES:
            try:
                response = self.client.converse(**request_params)
                return response
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_message = e.response["Error"].get("Message", "")

                # Si requiere inference profile y aún no lo hemos intentado, reintentar con prefijo regional
                if (
                    error_code == "ValidationException"
                    and "inference profile" in error_message.lower()
                    and not used_inference_profile
                    and not model_name.startswith(("arn:", "us.", "eu.", "ap."))
                ):
                    inference_profile_id = self._get_inference_profile_id(model_name)
                    logger.info(f"Modelo requiere inference profile, reintentando con: {inference_profile_id}")
                    request_params["modelId"] = inference_profile_id
                    used_inference_profile = True
                    continue

                if error_code in ("ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException"):
                    logging.error(f"Error llamando al LLM (Bedrock): {str(e)}")
                    time.sleep(wait)
                    wait *= BACKOFF_FACTOR
                    retry += 1
                else:
                    raise

        # fallback si falla tras todos los retries
        return None

    def extract_text(self, response):
        """
        Extrae el texto de la respuesta de la API Converse de Bedrock.
        """
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        texts = []
        for block in content_blocks:
            if "text" in block:
                texts.append(block["text"])

        return "\n".join(texts)

    def extract_usage(self, response):
        """
        Extrae los tokens de uso de la respuesta de Bedrock Converse.
        """
        usage = response.get("usage", {})
        return {
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        }

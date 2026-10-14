import logging

logger = logging.getLogger()

DEFAULT_CONTEXT_LIMIT = 128000


class LLMEngine:
    """
    Interfaz genérica para un motor de LLM.
    """
    def __init__(self):
        pass

    def generate(self, model_input, model_name: str, **kwargs):
        """
        Genera una respuesta a partir del input.
        Debe devolver un objeto con al menos:
        - output_text: str
        - usage: dict con input_tokens y output_tokens
        """
        raise NotImplementedError("Método generate() debe implementarse")

    def extract_text(self, response):
        raise NotImplementedError("Método extract_text() debe implementarse")

    def extract_usage(self, response):
        return {"input_tokens": 0, "output_tokens": 0}

    def get_context_limit(self, model_name: str) -> int:
        """
        Devuelve el límite de tokens de contexto para un modelo dado.
        Las subclases deben sobreescribir este método.
        """
        logger.warning(f"get_context_limit no implementado para este engine, usando fallback {DEFAULT_CONTEXT_LIMIT} para modelo {model_name}")
        return DEFAULT_CONTEXT_LIMIT

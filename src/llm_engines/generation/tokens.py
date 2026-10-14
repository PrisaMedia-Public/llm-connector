"""Conteo de tokens, portado del pipeline original (`llm.py`).

Usa `tiktoken` si está instalado (exacto para OpenAI, aproximado para el resto);
si no, aplica la aproximación 1 token ≈ 4 caracteres. No depende de ningún
engine, de modo que los conectores se mantienen intactos.
"""

CHARS_PER_TOKEN = 4


def count_tokens(text: str, model_name: str) -> int:
    """Número (estimado) de tokens de `text` para `model_name`."""
    try:
        from tiktoken import encoding_for_model, get_encoding
    except ImportError:
        return len(text) // CHARS_PER_TOKEN + 1

    try:
        enc = encoding_for_model(model_name)
    except KeyError:
        enc = get_encoding("o200k_base")
    return len(enc.encode(text))

"""
Ejemplo mínimo: usar un engine directamente (sin pipeline).

Requisitos:
    pip install "llm-engines[openai]"
    export OPENAI_API_KEY=sk-...
"""

import os

from llm_engines import create_engine


def main():
    engine = create_engine("openai", api_key=os.environ["OPENAI_API_KEY"])

    messages = [
        {"role": "system", "content": "Eres un asistente conciso y preciso."},
        {"role": "user", "content": "Explica qué es un paywall en una sola frase."},
    ]
    response = engine.generate(messages, "gpt-4.1-mini", temperature=0.3, maxTokens=100)

    print(engine.extract_text(response))
    print("Uso de tokens:", engine.extract_usage(response))


if __name__ == "__main__":
    main()

"""
Ejemplo clave: pipeline completo que conecta una *regla de generación* con la
llamada al modelo, de forma independiente del proveedor.

`TextGenerator` se encarga de todo el pipeline:
  - construye el prompt a partir de las reglas y las instrucciones,
  - cuenta tokens y calcula el presupuesto de contexto,
  - aplica chunking preventivo y resumen jerárquico si el texto es muy largo,
  - realiza la llamada final al modelo.

La misma definición de reglas vale para cualquier proveedor: solo cambia el
engine y el nombre del modelo.

Requisitos (según el proveedor que uses):
    pip install "llm-engines[all]"
"""

import os

from llm_engines import create_engine
from llm_engines.generation import GenerationRules, TextGenerator


# 1) Reglas de generación: se definen UNA vez, de forma declarativa.
RULES = GenerationRules(
    system_prompt=(
        "Eres un sistema editorial. Genera resúmenes informativos y neutrales, "
        "usando solo la información del texto proporcionado."
    ),
    format_rules={"resumen_corto": "Un único párrafo de 80-120 palabras."},
    prompt_version="demo_v1",
)

ARTICLE = """
Pega aquí el texto (largo) del artículo que quieres resumir...
"""


def run(provider: str, model_name: str, **engine_kwargs):
    engine = create_engine(provider, **engine_kwargs)
    generator = TextGenerator(engine, RULES)

    # 2) Una sola llamada: el pipeline resuelve chunking, tokens y la generación.
    result = generator.generate(
        ARTICLE,
        model=model_name,
        instructions="Resume el artículo destacando los temas principales.",
        output_type="resumen_corto",
        temperature=0.4,
        topP=0.9,
        maxTokens=400,
    )

    print(f"--- {provider} / {model_name} ---")
    print(result.text)
    print(f"Palabras: {result.word_count} | Tokens: {result.total_tokens} | Fragmentos: {result.chunks}")
    print()


if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        run("openai", "gpt-4.1-mini", api_key=os.environ["OPENAI_API_KEY"])

    if os.environ.get("ANTHROPIC_API_KEY"):
        run("anthropic", "claude-3-5-haiku-20241022", api_key=os.environ["ANTHROPIC_API_KEY"])

    # Bedrock usa las credenciales AWS del entorno (perfil, rol IAM, etc.).
    # run("bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0", region_name="eu-west-1")

# Changelog

Todas las novedades relevantes de este proyecto se documentan aquí.
El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/)
y el proyecto sigue [Versionado Semántico](https://semver.org/lang/es/).

## [1.1.0] - 2026-10-09

### Añadido
- Capa de generación completa (`generation/`) que conecta una regla con la
  llamada al modelo, portada del pipeline de producción (`llm.py`) de forma
  genérica (sin prompts específicos de ningún caso de uso):
  - `GenerationRules` (reglas declarativas) y `PromptBuilder` (construcción de
    prompts con delimitadores).
  - `count_tokens` (conteo de tokens, `tiktoken` opcional) y `split_text`
    (*chunking*, `langchain-text-splitters` opcional, con fallbacks).
  - `TextGenerator` / `GenerationResult`: orquesta presupuesto de contexto,
    *chunking* preventivo, resumen jerárquico de fragmentos y llamada final.
- Extras opcionales `tokens`, `chunking` y `generation` en `pyproject.toml`.
- Ejemplo `examples/pipeline_rules_to_llm.py` y tests `tests/test_generation.py`.

### Cambiado
- Sustituida la capa fina `prompting.py` por la capa `generation/`.

## [1.0.0] - 2026-10-09

### Añadido
- Publicación de los engines de producción tal cual: `base.py`, `openai.py`,
  `anthropic.py`, `bedrock.py`, `gemini.py` y `constants.py`.
- Factoría `create_engine(provider, ...)` con carga perezosa de SDKs, alias de
  proveedor y `register_engine` para engines de terceros.
- Reintento automático con *inference profile* cross-region en Bedrock.
- Dependencias por proveedor como *extras* opcionales.
- Ejemplos (`examples/`) y tests sin red (`tests/`).
- Documentación: README, ARCHITECTURE, CONTRIBUTING, SECURITY.

### Nota
- Respecto al código original interno, la **única** modificación en los engines
  es la ruta de importación de `constants`
  (`from constants import ...` → `from llm_engines.constants import ...`).
  `base.py` y `constants.py` son idénticos al original.

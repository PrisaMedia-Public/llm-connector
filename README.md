# llm-engines

Conectores (wrappers) ligeros y homogéneos para múltiples proveedores de LLM
bajo **una única interfaz común**. Permite conectar *reglas de generación de
texto* con el modelo de lenguaje sin acoplar tu código a un proveedor concreto.

Proveedores soportados:

| Proveedor | Engine | SDK subyacente |
|-----------|--------|----------------|
| OpenAI | `OpenAIEngine` | `openai` (Responses API) |
| Anthropic | `AnthropicEngine` | `anthropic` (Messages API) |
| Amazon Bedrock | `BedrockEngine` | `boto3` (Converse API) |
| Google Vertex AI / Model Garden | `GeminiEngine` | `google-genai` |

> Este repositorio comparte de forma abierta la capa de integración y
> optimización usada para generar contenido con IA, para que otros equipos y
> proyectos puedan reutilizarla en distintos casos de uso.

---

## ¿Por qué?

Cada proveedor de LLM tiene su propio SDK, su formato de mensajes, sus nombres
de parámetros (`top_p` vs `topP`, `max_tokens` vs `max_output_tokens`...) y sus
particularidades (modelos de razonamiento que no admiten `temperature`,
inference profiles de Bedrock, etc.).

`llm-engines` esconde todas esas diferencias tras una interfaz común:

```python
engine.generate(messages, model_name, temperature=0.3, topP=0.9, maxTokens=400)
text = engine.extract_text(response)
usage = engine.extract_usage(response)
```

Cambiar de proveedor es cambiar **una línea**.

---

## Instalación

Instala solo el/los proveedor(es) que necesites (extras opcionales):

```bash
pip install "llm-engines[openai]"
pip install "llm-engines[anthropic]"
pip install "llm-engines[bedrock]"
pip install "llm-engines[gemini]"
pip install "llm-engines[generation]"  # tiktoken + chunking para el pipeline
pip install "llm-engines[all]"         # todo
```

Instalación local desde el código fuente:

```bash
pip install -e ".[all,dev]"
```

---

## Uso rápido

### Opción A — pipeline completo (regla → modelo)

`TextGenerator` cubre todo el pipeline: construye el prompt a partir de las
reglas, cuenta tokens, aplica *chunking* preventivo y resumen jerárquico si el
texto no cabe en el contexto, y hace la llamada final.

```python
from llm_engines import create_engine
from llm_engines.generation import TextGenerator, GenerationRules

engine = create_engine("openai", api_key="sk-...")

rules = GenerationRules(
    system_prompt="Eres un asistente conciso y preciso.",
    format_rules={"resumen": "Un párrafo de 80-120 palabras."},
)
generator = TextGenerator(engine, rules)

result = generator.generate(
    texto_largo,
    model="gpt-4.1-mini",
    instructions="Resume el texto destacando los temas principales.",
    output_type="resumen",
    temperature=0.4,
    topP=0.9,
    maxTokens=400,
)

print(result.text)
print(result.total_tokens, result.chunks, result.word_count)
```

### Opción B — engine directo (sin pipeline)

```python
from llm_engines import create_engine

engine = create_engine("openai", api_key="sk-...")
messages = [
    {"role": "system", "content": "Eres un asistente conciso."},
    {"role": "user", "content": "Resume este texto: ..."},
]
response = engine.generate(messages, "gpt-4.1-mini", temperature=0.3, maxTokens=400)

print(engine.extract_text(response))
print(engine.extract_usage(response))  # {'input_tokens': ..., 'output_tokens': ...}
```

### Cambiar de proveedor

La lógica no cambia; solo el engine y el nombre del modelo:

```python
# Anthropic
generator = TextGenerator(create_engine("anthropic", api_key="sk-ant-..."))
result = generator.generate(texto, model="claude-3-5-haiku-20241022", instructions="Resume.")

# Amazon Bedrock (usa credenciales AWS del entorno / rol IAM)
generator = TextGenerator(create_engine("bedrock", region_name="eu-west-1"))
result = generator.generate(texto, model="anthropic.claude-3-5-haiku-20241022-v1:0", instructions="Resume.")

# Google Vertex AI / Model Garden
generator = TextGenerator(create_engine("gemini", credentials_json=sa_json, project="my-gcp-project"))
result = generator.generate(texto, model="gemini-2.5-flash", instructions="Resume.")
```

---

## Conceptos clave

### `LLMEngine` (interfaz común)

Todos los engines implementan:

- `generate(model_input, model_name, **kwargs)` → respuesta nativa del proveedor.
- `extract_text(response)` → `str` con el texto generado.
- `extract_usage(response)` → `{"input_tokens": int, "output_tokens": int}`.
- `get_context_limit(model_name)` → límite de tokens de contexto del modelo.

### Formato de mensajes

Estilo chat, común a todos los proveedores:

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
]
```

Cada engine lo traduce internamente al formato de su SDK.

### Hiperparámetros normalizados

Usa siempre estas claves; cada engine las mapea a su API:

| Clave común | OpenAI | Anthropic | Bedrock | Gemini |
|-------------|--------|-----------|---------|--------|
| `temperature` | `temperature` | `temperature` | `temperature` | `temperature` |
| `topP` | `top_p` | `top_p` | `topP` | `top_p` |
| `maxTokens` | `max_output_tokens` | `max_tokens` | `maxTokens` | `max_output_tokens` |

Los modelos que no admiten parámetros de muestreo (modelos de razonamiento) los
descartan automáticamente.

### Reglas de generación y pipeline (`generation/`)

La capa `generation` conecta una *regla de generación* con la llamada al modelo:

- **`GenerationRules`**: definición declarativa (prompt de sistema, prompt de
  *chunk*, reglas de formato por tipo de salida, etiquetas de contexto).
- **`PromptBuilder`**: construye los mensajes finales y los de cada *chunk* con
  delimitadores que separan instrucciones de datos.
- **`count_tokens`** / **`split_text`**: conteo de tokens y *chunking* (con
  fallbacks si no instalas `tiktoken` / `langchain-text-splitters`).
- **`TextGenerator`**: orquesta todo el pipeline
  (presupuesto de contexto → *chunking* → resumen de fragmentos → llamada final)
  y devuelve un `GenerationResult` con texto, recuento de palabras y tokens.

```python
from llm_engines.generation import TextGenerator, GenerationRules

generator = TextGenerator(engine, GenerationRules(system_prompt="..."))
result = generator.generate(texto, model="gpt-4.1-mini", instructions="Resume.")
```

---

## Resiliencia

Todos los engines incluyen reintentos con *backoff* exponencial ante errores
transitorios (rate limits, throttling, errores temporales de API). Los valores
por defecto (3 intentos, factor 2) se definen en `llm_engines/constants.py`.

Bedrock, además, reintenta automáticamente con un *inference profile*
cross-region cuando el modelo lo requiere.

---

## Estructura del proyecto

Los engines (`base`, `openai`, `anthropic`, `bedrock`, `gemini`) son los mismos
que se usan en producción, publicados tal cual (la única diferencia respecto al
origen es de dónde importan `constants`). Encima se añaden una factoría y la
capa de generación (`generation/`) que conecta las reglas con la llamada al modelo.

```
llm-engines/
├── README.md
├── ARCHITECTURE.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── SECURITY.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .env.example
├── src/
│   └── llm_engines/
│       ├── __init__.py
│       ├── constants.py      # MAX_RETRIES, BACKOFF_FACTOR (original)
│       ├── base.py           # Interfaz común LLMEngine (original)
│       ├── openai.py         # OpenAIEngine (original)
│       ├── anthropic.py      # AnthropicEngine (original)
│       ├── bedrock.py        # BedrockEngine (original)
│       ├── gemini.py         # GeminiEngine (original)
│       ├── factory.py        # create_engine(provider, ...)
│       └── generation/       # Pipeline regla -> modelo
│           ├── rules.py          # GenerationRules
│           ├── prompt_builder.py # PromptBuilder
│           ├── tokens.py         # count_tokens
│           ├── chunking.py       # split_text
│           └── pipeline.py       # TextGenerator / GenerationResult
├── examples/
│   ├── basic_openai.py
│   └── pipeline_rules_to_llm.py
└── tests/
    ├── test_base.py
    ├── test_factory.py
    └── test_generation.py
```

---

## Credenciales

Copia `.env.example` a `.env` y rellena tus claves. **Nunca** subas el `.env`
real ni ningún secreto al repositorio. Para producción se recomienda usar un
gestor de secretos (AWS Secrets Manager, GCP Secret Manager, variables de
entorno del runtime, etc.).

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Los tests no realizan llamadas de red: validan la factoría, el mapeo de reglas
y la interfaz base.

---

## Licencia

[MIT](LICENSE).

## Contribuir

Lee [CONTRIBUTING.md](CONTRIBUTING.md). Las contribuciones son bienvenidas.

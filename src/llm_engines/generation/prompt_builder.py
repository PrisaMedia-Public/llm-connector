"""Construye los mensajes que se envían al engine a partir de unas `GenerationRules`.

Reproduce la construcción de prompts del pipeline original (delimitadores
<<<START_...>>> para separar instrucciones de datos y reducir el riesgo de que
el contenido se interprete como instrucciones), de forma genérica.
"""

from typing import Any, Mapping, Optional

from llm_engines.generation.rules import GenerationRules


class PromptBuilder:
    """Convierte reglas + datos de la petición en mensajes estilo chat."""

    def __init__(self, rules: Optional[GenerationRules] = None):
        self.rules = rules or GenerationRules()

    def format_context(self, context: Optional[Mapping[str, Any]]) -> str:
        """Metadatos del texto como lista '- Etiqueta: valor'. Omite los vacíos."""
        if not context:
            return ""

        labels = self.rules.context_labels
        fields = labels.keys() if labels else context.keys()

        lines = []
        for name in fields:
            value = context.get(name)
            if not value:
                continue
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(v) for v in value)
            lines.append(f"- {labels.get(name, name)}: {value}")
        return "\n".join(lines)

    def build_chunk_messages(self, chunk_text: str) -> list:
        """Mensajes para resumir un fragmento de un texto largo."""
        content = "\n\n".join([
            f"INSTRUCCIONES:\n{self.rules.chunk_prompt}",
            f"REGLAS DE FORMATO:\n{self.rules.chunk_format_rule}",
            f"FRAGMENTO:\n<<<START_CHUNK>>>\n{chunk_text}\n<<<END_CHUNK>>>",
        ])
        return [{"role": "user", "content": content}]

    def build_final_messages(
        self,
        text: str,
        instructions: str,
        *,
        system_prompt: Optional[str] = None,
        output_type: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        output_format: str = "",
    ) -> list:
        """
        Mensajes para la generación final.

        Args:
            text: texto de entrada (o la unión de resúmenes de fragmentos).
            instructions: instrucciones específicas de la petición.
            system_prompt: prompt de sistema; por defecto, `rules.system_prompt`.
            output_type: clave de `rules.format_rules` (p. ej. 'short_summary'). Opcional.
            context: metadatos del texto (título, autores...).
            output_format: formato de salida libre (p. ej. un esquema JSON).
        """
        sections = [f"INSTRUCCIONES:\n{instructions}"]

        if output_type:
            if output_type not in self.rules.format_rules:
                available = ", ".join(self.rules.format_rules) or "(ninguno)"
                raise ValueError(f"output_type desconocido: '{output_type}'. Disponibles: {available}")
            sections.append(
                f"TIPO DE SALIDA REQUERIDO:\n{output_type}\n\n"
                f"REGLAS DE FORMATO:\n{self.rules.format_rules[output_type]}"
            )

        context_block = self.format_context(context)
        if context_block:
            sections.append(f"CONTEXTO:\n<<<START_CONTEXT>>>\n{context_block}\n<<<END_CONTEXT>>>")

        if output_format:
            sections.append(f"FORMATO DE SALIDA REQUERIDO:\n{output_format}")

        sections.append(f"TEXTO:\n<<<START_TEXT>>>\n{text}\n<<<END_TEXT>>>")

        messages = []
        system = system_prompt if system_prompt is not None else self.rules.system_prompt
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": "\n\n".join(sections)})
        return messages

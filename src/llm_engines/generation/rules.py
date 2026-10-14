"""Reglas de generación de texto: qué se le dice al modelo y cómo.

`GenerationRules` es la definición declarativa e independiente del proveedor que
el pipeline (`TextGenerator`) traduce a llamadas concretas al engine. Los valores
por defecto son genéricos; sustitúyelos según tu caso de uso.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class GenerationRules:
    """
    Reglas de generación.

    Attributes:
        system_prompt: prompt de sistema por defecto si la llamada no aporta uno.
        chunk_prompt: instrucciones para resumir cada fragmento de un texto largo.
        chunk_format_rule: regla de formato para los resúmenes de fragmento.
        format_rules: tipo de salida -> regla de formato (extensión, estructura...).
        context_labels: metadato -> etiqueta en el prompt. Si está vacío, se
            incluyen todos los metadatos recibidos con su nombre original.
        prompt_version: identificador que se devuelve en el resultado para trazabilidad.
    """

    system_prompt: str = ""
    chunk_prompt: str = (
        "Resume el siguiente fragmento de texto de forma concisa, "
        "conservando la información esencial."
    )
    chunk_format_rule: str = "Devuelve únicamente un resumen breve en un párrafo de texto."
    format_rules: Mapping[str, str] = field(default_factory=dict)
    context_labels: Mapping[str, str] = field(default_factory=dict)
    prompt_version: str = "v1"

    def with_overrides(self, **changes: Any) -> "GenerationRules":
        """Copia de las reglas con algunos campos sustituidos."""
        return dataclasses.replace(self, **changes)

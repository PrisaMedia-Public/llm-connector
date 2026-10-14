# Contribuir

¡Gracias por tu interés en contribuir! Este proyecto busca fomentar la
colaboración abierta en la integración de LLMs.

## Flujo de trabajo

1. Haz un *fork* del repositorio y crea una rama descriptiva
   (`feat/nuevo-engine`, `fix/bedrock-retry`...).
2. Instala el entorno de desarrollo:
   ```bash
   pip install -e ".[all,dev]"
   ```
3. Asegúrate de que los tests pasan:
   ```bash
   pytest
   ```
4. Abre un *Pull Request* describiendo el cambio y su motivación.

## Estilo de código

- Python 3.10+.
- Sigue el estilo existente (nombres claros, *docstrings* breves en español,
  logging con `logging` del módulo).
- No introduzcas dependencias obligatorias nuevas sin discutirlo; los SDKs de
  proveedor deben ser *extras* opcionales.

## Añadir un proveedor

Consulta la sección "Cómo añadir un nuevo proveedor" en
[ARCHITECTURE.md](ARCHITECTURE.md).

## Seguridad

- No incluyas claves, tokens ni credenciales en el código ni en los tests.
- Reporta vulnerabilidades de forma privada a los mantenedores antes de
  divulgarlas públicamente.

## Licencia

Al contribuir, aceptas que tu aportación se publique bajo la licencia
[MIT](LICENSE).

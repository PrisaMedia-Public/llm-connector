# Política de seguridad

## Gestión de credenciales

- La librería **no** lee ni almacena secretos por su cuenta: las claves de API y
  credenciales se inyectan al construir cada engine (`create_engine(...)`).
- **Nunca** subas claves, tokens ni ficheros de credenciales al repositorio.
  Usa variables de entorno o un gestor de secretos (AWS Secrets Manager,
  GCP Secret Manager, etc.). El fichero `.env` está en `.gitignore`.
- Para Amazon Bedrock se recomienda usar roles IAM en lugar de claves estáticas.

## Buenas prácticas de prompting

- Los textos de entrada deben tratarse como datos, no como instrucciones.
  Valida y delimita el contenido de usuario antes de incrustarlo en los prompts
  para reducir el riesgo de *prompt injection*.

## Reporte de vulnerabilidades

Si detectas una vulnerabilidad, repórtala de forma privada a los mantenedores
antes de divulgarla públicamente. No abras un *issue* público con detalles de la
vulnerabilidad.

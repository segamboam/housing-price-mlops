# Uso de Herramientas de IA

Este documento describe las herramientas de inteligencia artificial utilizadas durante el desarrollo de este proyecto MLOps.

## Herramientas Utilizadas

### Google AI Studio
**Propósito:** Planificación inicial y análisis arquitectónico

- Análisis y comprensión del dataset Boston Housing
- Diseño de la arquitectura general del sistema
- Planificación de patrones de diseño (Strategy, Factory)
- Evaluación de trade-offs entre diferentes enfoques

### Claude Code (Anthropic)
**Propósito:** Implementación y desarrollo principal

- Implementación de todas las funcionalidades core
- Desarrollo de la API REST con FastAPI
- Creación del pipeline de entrenamiento
- Integración con MLflow (tracking, registry, artifacts)
- Configuración de Docker y docker-compose
- Pipeline CI/CD con GitHub Actions
- Suite de tests con pytest
- Documentación técnica

### Cursor
**Propósito:** Refinamiento y desarrollo asistido

- Refinamiento de código e iteraciones rápidas
- Correcciones puntuales y ajustes menores
- Desarrollo asistido por IDE con contexto del proyecto

## Principios Aplicados

1. **Revisión Humana**: Todo código generado por IA fue revisado manualmente antes de ser integrado
2. **Testing**: Todas las funcionalidades fueron validadas con tests automatizados
3. **Supervisión Arquitectónica**: Las decisiones de arquitectura fueron tomadas con criterio humano
4. **Responsabilidad**: El desarrollador mantiene responsabilidad total sobre la calidad y seguridad del código final

## Transparencia

Esta declaración sigue las mejores prácticas de desarrollo asistido por IA, reconociendo las herramientas utilizadas mientras se mantiene plena responsabilidad sobre el producto final.

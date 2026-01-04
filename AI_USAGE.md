# Uso de Herramientas de IA

Este documento describe cómo utilicé herramientas de IA en el desarrollo de este proyecto.

## Enfoque de Desarrollo

Desarrollé este proyecto con un enfoque de **desarrollo asistido por IA**, donde las herramientas funcionaron como aceleradores de productividad bajo mi dirección y supervisión.

## Herramientas Utilizadas

- **Google AI Studio**: Análisis inicial del problema y planificación
- **Claude (Anthropic)**: Generación y refinamiento de código
- **Cursor**: IDE con asistencia contextual

## Mi Contribución Directa

### Decisiones de Arquitectura
- **Stack tecnológico**: Elegí MLflow por mi experiencia previa en proyectos ML, FastAPI porque es mi herramienta de uso diario (asincronía, Pydantic), y Docker Compose para orquestación
- **Patrón Strategy**: Analicé el requerimiento de intercambiabilidad de modelos y diseñé la solución
- **Artifact Bundle**: Diseñé el empaquetado para centralizar modelo y preprocesador como unidad atómica
- **Infraestructura robusta**: Decidí usar PostgreSQL + MinI para lograr un setup production-ready

### Implementaciones y Funcionalidades
- Pedí específicamente el sistema de CLI y experiments (no sugerido por IA)
- Creé el script de seed para facilitar la evaluación del proyecto
- Configuré Ruff y los estándares de código

### Iteraciones y Resolución de Problemas
- Iteré múltiples veces en la integración MLflow con fuentes externas
- Ajusté el sistema de guardado de modelo + preprocesador
- Debuggeé y corregí código generado

## Principios que Apliqué

1. **Dirección propia**: Tomé las decisiones de arquitectura y diseño con criterio basado en mi experiencia
2. **Supervisión activa**: Revisé y ajusté todo el código generado según las necesidades del proyecto
3. **Testing**: Supervisé el desarrollo de la suite de tests para validar funcionalidad
4. **Comprensión completa**: Puedo explicar y defender cada decisión técnica del proyecto

## Transparencia

Reconozco el uso de herramientas de IA como aceleradores de desarrollo. Mi valor agregado está en la dirección técnica, las decisiones arquitectónicas y la capacidad de construir, mantener y evolucionar el sistema.

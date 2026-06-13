# Guía de la Entrega Final: Máster en Inteligencia Artificial, Cloud Computing y DevOps

## 1. Introducción

Esta guía describe la evaluación final de la asignatura **Large Language Models**. El trabajo consiste en crear un **asistente turístico** utilizando un Google Gemini a través de su API. El producto final será un notebook que reúna:

- **RAG** sobre una guía turística facilitada.
- **Diálogo multiturno** que mantenga el contexto de la conversación.
- Al menos una **llamada de función externa** para la predicción del tiempo (`get_weather(fecha)`) con gestión de errores.

## 2. Objetivo General

Desarrollar un **prototipo conversacional reproducible** que combine búsqueda semántica, mantenimiento de contexto y llamadas a servicios externos desde un único notebook.

## 3. Requisitos Mínimos

1.  **Conexión con un LLM comercial:** Uso de variables de entorno o gestor seguro para la API key y exposición clara de parámetros (temperatura, top-p, tokens, etc.).
2.  **Implementación RAG:**
    - División de la guía en _chunks_ y generación de _embeddings_.
    - Uso de un _vector store_ (FAISS, Chroma u otro).
    - Respuestas con **citación de fuentes**.
3.  **Diálogo multiturno:** Uso de memoria/contenedor de historial y control de límites de tokens.
4.  **Function Call obligatoria:**
    - Definición mediante JSON Schema o Pydantic de la función `get_weather`.
    - Ejecución (real o simulada) con manejo de errores y registro de intentos en un **log**.

## 4. Entregables

- **Repositorio:** Estructura clara, README detallado, `requirements.txt` o `pyproject.toml` y `.gitignore`.
- **Notebook principal:** Celdas ordenadas (carga, indexación, conversación, pruebas, análisis).
- **Informe final:** Diseño de la solución, decisiones técnicas, resultados, limitaciones y mejoras futuras.

## 5. Criterios de Evaluación (Pesos)

- **RAG y conexión con el LLM:** 30%.
- **Diálogo multiturno y function calls:** 25%.
- **Evaluación y análisis crítico:** 15%.
- **Calidad del código y documentación:** 20%.
- **Experiencia de usuario:** 10%.
- **Bonus:** Hasta +20% (despliegue, streaming, agentes, multimodalidad, observabilidad).

## 6. Detalles de los Criterios de Evaluación

### 7.1 RAG y conexión con el LLM (30%)

- Estrategia de _chunking_ y _embeddings_.
- Eficiencia del _vector store_ y calidad de recuperación.

### 7.2 Diálogo multiturno y function calls (25%)

- Coherencia entre turnos.
- Definición y serialización correcta de la llamada.
- Mínimo **tres invocaciones correctas** con manejo de fallos.

### 7.3 Evaluación y visualización (15%)

- Conjunto reproducible de _prompts_.
- Presentación de métricas y gráficos.
- Discusión de casos límite y limitaciones.

### 7.4 Calidad del código y documentación (20%)

- Cumplimiento de **PEP 8** y _docstrings_ claros.
- Código modular y reutilizable con instrucciones de ejecución.

### 7.5 Experiencia de usuario (10%)

- Interfaz intuitiva, mensajes de error útiles y **citaciones claras**.
- Ejemplos que guíen al usuario.

## 8. Bonificaciones Técnicas (Hasta +20%)

- **Despliegue web:** Interfaz con Streamlit, Gradio, etc.
- **Streaming de tokens:** Respuestas en tiempo real.
- **Agentes:** Coordinación de varias herramientas.
- **Multimodalidad:** Soporte de imágenes o audio.

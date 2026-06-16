# Informe Final — Asistente Turístico de Tenerife

## 1. Diseño de la solución

### ¿Qué queríamos construir?

El objetivo era crear un asistente conversacional para turistas que visitan Tenerife. El asistente tiene que ser capaz de responder preguntas sobre la isla (qué ver, dónde comer, zonas...) usando un documento PDF como fuente de conocimiento, y además dar información meteorológica en tiempo real.

Para eso, la solución combina tres capacidades:

1. **RAG (Retrieval-Augmented Generation):** en vez de depender solo del conocimiento interno del LLM (que puede estar desactualizado o ser genérico), extraemos información de un PDF específico sobre Tenerife y la inyectamos en el prompt como contexto. El LLM responde basándose en esa información concreta.

2. **Function Calling:** cuando el usuario pregunta por el tiempo, el LLM no inventa datos meteorológicos. En su lugar, solicita ejecutar una función Python real (`get_weather`) que consulta la API de Open-Meteo y devuelve el pronóstico actualizado.

3. **Diálogo multiturno:** el asistente mantiene memoria de la conversación durante varios turnos, lo que permite preguntas de seguimiento naturales ("¿y mañana?", "¿y en el Norte?").

### Arquitectura general

```
Usuario (Streamlit)
        │
        ▼
  chat_with_tools()
        │
        ├── 1. RAG: busca chunks relevantes en ChromaDB
        │          └── GeminiEmbeddings → búsqueda semántica
        │
        ├── 2. Primera llamada al LLM (Gemini 2.5 Flash)
        │          └── ¿necesita datos del tiempo?
        │                    │
        │          SÍ ───────┴──────── NO
        │          │                   │
        │    get_weather()        responde directo
        │    Open-Meteo API       con contexto RAG
        │          │
        │    Segunda llamada al LLM
        │    (con datos del tiempo)
        │
        ▼
  Respuesta con citación de fuentes → Usuario
```

### El flujo de dos llamadas al LLM

Este concepto se trabajó bastante en clase, así que quedó claro desde el principio. Cuando hay function calling, el LLM no puede responder directamente porque necesita datos externos. El flujo es:

1. **Primera llamada:** el LLM recibe la pregunta y decide que necesita el tiempo. En vez de responder texto, devuelve una _petición de ejecución_ con la herramienta que quiere usar y los argumentos.
2. **Python ejecuta `get_weather`** con esos argumentos y obtiene el pronóstico real de Open-Meteo.
3. **Segunda llamada:** el LLM recibe el resultado de la herramienta y ahora sí redacta la respuesta final para el usuario.

Esto significa que cada pregunta sobre el tiempo hace **dos llamadas a la API de Gemini**, no una.

---

## 2. Decisiones técnicas

### 2.1 Chunking con metadatos enriquecidos

El PDF se divide en fragmentos usando `RecursiveCharacterTextSplitter` con separadores adaptados al formato del documento (`\n\n`, `\n•`, `▪`, etc.). El tamaño elegido fue de **1000 caracteres con 200 de solapamiento**. Se probaron estos valores como punto de partida razonable: chunks demasiado pequeños pierden contexto dentro del fragmento, y chunks demasiado grandes reducen la precisión de la búsqueda semántica. El solapamiento evita que una idea que cae justo en el corte entre dos chunks quede partida y pierda significado.

La decisión fue enriquecer cada chunk con metadatos de **zona** (`Norte`/`Sur`) y **tipo** (`Sitio de interés`/`Restaurante`), detectados mediante una máquina de estados que lee el texto secuencialmente. Esto permite que el LLM cite fuentes con información concreta (`[Sur - Sitio de interés]`) en lugar de solo decir "según el documento".

### 2.2 Embeddings con Gemini y ChromaDB

Para que la búsqueda semántica funcione con ChromaDB, se necesita un objeto que implemente la interfaz `Embeddings` de LangChain. Como la API de Gemini no la implementa directamente, se creó el adaptador `GeminiEmbeddingFunction` que:

- Recibe el cliente de Gemini ya inicializado (en vez de crear uno nuevo cada vez)
- Implementa `embed_documents` (para indexar chunks) y `embed_query` (para vectorizar la pregunta del usuario)

ChromaDB se persiste en disco (`data/chromadb/`), de modo que no hay que regenerar los embeddings en cada arranque de la aplicación.

### 2.3 Definición de la herramienta con `@tool`

En vez de definir la función con JSON Schema puro, se usó el decorador `@tool` de LangChain. Esto tiene una ventaja práctica: la descripción de la función y sus argumentos se escriben directamente en el docstring de Python, y LangChain se encarga de convertirlos al formato que entiende Gemini. Además, el docstring incluye instrucciones explícitas para el LLM sobre cuándo debe usar la herramienta.

### 2.4 Garantía de citación de fuentes en tool calling

Al revisar los resultados de la evaluación, se comprobó que el string `[get_weather]` no aparecía siempre en la respuesta del modelo. Para evitar depender de que el LLM lo incluya de forma consistente, se optó por añadir una lógica determinista que lo garantice:

```python
if "[get_weather]" not in texto_respuesta:
    texto_respuesta += "\n\nFuentes: [get_weather]"
```

Si el modelo ya lo incluyó, no pasa nada. Si no lo incluyó, se añade. Es una solución sencilla pero efectiva para asegurar consistencia en la evaluación.

### 2.5 Control de la ventana de tokens

El historial de conversación crece con cada turno y puede superar el límite de contexto del modelo. La solución implementada (`controlar_ventana_tokens`) mantiene solo los últimos **3 turnos** (6 mensajes), eliminando siempre pares completos para no dejar mensajes vacíos.

Se eligió 3 turnos como compromiso entre memoria y coste en tokens. En una aplicación real habría que ajustar este valor según el caso de uso.

### 2.6 Logging de invocaciones

Cada vez que el LLM invoca `get_weather`, se registra en `logs/tool_calls.log` con timestamp y nivel de severidad. El logger se configura con `propagate = False` para evitar que los mensajes se dupliquen en la consola, y con `logging.getLogger("httpx").setLevel(WARNING)` para silenciar el ruido de las llamadas HTTP internas de LangChain.

### 2.7 Streamlit: decisiones de arquitectura

Para la interfaz web se tomaron varias decisiones concretas:

- **`@st.cache_resource`** en la inicialización del vector store y del modelo: garantiza que solo se ejecutan una vez por sesión, evitando regenerar embeddings o crear múltiples clientes de API en cada recarga.
- **`Path(__file__).parent`** para todas las rutas de ficheros: al ejecutar la app con Streamlit aparecían errores de ruta que no tenían sentido a primera vista. El problema era que Streamlit no ejecuta el script desde el mismo directorio que el archivo, así que las rutas relativas fallaban. La solución fue construir todas las rutas a partir de la ubicación del propio archivo.
- **`st.session_state.historial`** en vez de una variable global: Streamlit re-ejecuta el script completo en cada interacción, así que el historial debe vivir en el estado de sesión para persistir entre mensajes.
- **Streaming real con generadores**: la función `chat_with_tools` devuelve un generador cuando `stream=True`. La primera llamada al LLM (detección de tool calls) siempre es bloqueante; solo la llamada final se streamea. Esto permite mostrar el spinner mientras se procesa la lógica y luego hacer aparecer el texto token a token.

---

## 3. Resultados de la evaluación

Se evaluaron 9 prompts representativos contra un ground truth con los valores esperados para dos métricas:

| Métrica            | Descripción                                            | Resultado           |
| ------------------ | ------------------------------------------------------ | ------------------- |
| `contiene_fuentes` | La respuesta incluye citación de fuentes               | **100%** de acierto |
| `uso_tool_calling` | La herramienta `get_weather` fue invocada cuando debía | **100%** de acierto |

Los prompts cubiertos incluían preguntas de información general (RAG), preguntas meteorológicas (tool calling) y preguntas mixtas. La latencia media por respuesta se situó en torno a los 3-6 segundos, con picos en las respuestas que requieren tool calling (dos llamadas a la API).

La visualización con Plotly permite comparar de forma visual el comportamiento real vs. esperado, y observar la distribución de tiempos de respuesta por tipo de pregunta.

---

## 4. Limitaciones

### 4.1 Solo dos zonas geográficas

`get_weather` distingue únicamente entre `Norte` y `Sur`. Si el usuario pregunta por una zona específica como "Garachico" o "El Teide", el modelo tiene que inferir a qué zona pertenece. En algunos casos puede devolver un error si no reconoce la zona.

### 4.2 Dependencia del PDF para el conocimiento

El asistente solo sabe lo que está en el PDF. Si un usuario pregunta por un restaurante que no aparece en el documento, el asistente responde que no tiene esa información, aunque el lugar exista. El conocimiento está acotado al contenido indexado.

### 4.3 Sin memoria entre sesiones

El historial de conversación vive en `st.session_state` y desaparece al cerrar el navegador o recargar la página. No hay persistencia entre sesiones distintas, lo que impide que el asistente recuerde preferencias de usuarios anteriores.

### 4.4 El LLM puede no activar el tool en preguntas ambiguas

Si el usuario pregunta "¿hace buen tiempo en Tenerife?", el LLM puede responder con información general del PDF en vez de invocar `get_weather`. La herramienta se activa de forma más fiable con preguntas directas como "¿qué temperatura hay hoy en el Sur?". Esto es un comportamiento esperado del function calling reactivo: el modelo decide de forma autónoma cuándo usar la herramienta.

### 4.5 Ventana de historial corta

Mantener solo 3 turnos en memoria es suficiente para conversaciones simples, pero puede romper el hilo en conversaciones más largas donde el usuario hace referencia a algo dicho hace tres o cuatro mensajes.

---

## 5. Mejoras futuras

### 5.1 Más herramientas

Ampliar el sistema de function calling con herramientas adicionales:

- `buscar_restaurantes(zona, tipo_cocina)` — integración con Google Places o TripAdvisor
- `get_transport(origen, destino)` — información de líneas de bus o taxis en Tenerife
- `check_availability(actividad, fecha)` — consulta de disponibilidad de excursiones

Con varias herramientas el sistema se convierte en un agente real que puede combinar múltiples fuentes de información en una sola respuesta.

### 5.2 Persistencia de sesiones

Guardar el historial de cada usuario en una base de datos (SQLite, Redis) indexado por un ID de sesión. Esto permitiría retomar conversaciones y personalizar respuestas según preferencias previas.

### 5.3 Detección de zona más robusta

Añadir un diccionario de municipios y puntos de interés mapeados a sus zonas para que `get_weather` no dependa de que el modelo infiera la zona correctamente.


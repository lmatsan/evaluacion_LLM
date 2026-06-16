# Asistente Turístico de Tenerife con LLMs

Asistente conversacional para viajeros que quieren información sobre Tenerife. Combina recuperación de información (RAG) sobre un documento PDF con consultas meteorológicas en tiempo real mediante function calling a la API de Open-Meteo.

## Arquitectura

```
Usuario
  │
  ▼
chat_with_tools()
  ├── RAG: ChromaDB + Gemini Embeddings → contexto del PDF
  └── Tool Calling: get_weather() → Open-Meteo API
        │
        ▼
  Gemini 2.5 Flash (LangChain)
        │
        ▼
  Respuesta con citación de fuentes
```

**Flujo de function calling (dos llamadas):**
1. El LLM recibe la pregunta y decide si necesita datos meteorológicos
2. Si llama a `get_weather`, Python ejecuta la función y devuelve el pronóstico
3. El LLM recibe el resultado y redacta la respuesta final

## Estructura del proyecto

```
Evaluacion/
├── notebooks/
│   └── main.ipynb          # Notebook principal (secciones 1-9)
├── data/
│   └── TENERIFE.pdf        # Documento fuente para el RAG
├── logs/
│   └── tool_calls.log      # Registro de invocaciones a get_weather
├── src/                    # Módulos extraídos (para Streamlit)
│   ├── chat.py
│   ├── rag.py
│   ├── tools.py
│   └── evaluation.py
├── app.py                  # Interfaz Streamlit (bonus)
├── requirements.txt        # Dependencias del proyecto
└── .env                    # Variables de entorno (no incluido en git)
```

## Requisitos previos

- Python 3.11+
- Clave de API de Google Generative AI ([Google AI Studio](https://aistudio.google.com/))

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd Evaluacion

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar el entorno virtual
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

## Configuración

Crea un archivo `.env` en la raíz del proyecto con tu clave de API:

```
GOOGLE_API_KEY=tu_clave_aqui
```

## Ejecución

### Notebook (flujo completo)

```bash
jupyter notebook notebooks/main.ipynb
```

Ejecuta las celdas en orden. El notebook está estructurado en 9 secciones:

| Sección | Contenido |
|---------|-----------|
| 1 | Setup, carga de variables de entorno y verificación de conexión |
| 2 | Lectura y extracción de texto del PDF |
| 3 | Chunking con `RecursiveCharacterTextSplitter` y enriquecimiento de metadatos |
| 4 | Generación de embeddings con Gemini y almacenamiento en ChromaDB |
| 5 | Búsqueda semántica (RAG) e inyección de contexto en el prompt |
| 6 | Diálogo multiturno con control de ventana de tokens |
| 7 | Integración con Open-Meteo API (`get_weather`) |
| 8 | Agente conversacional con function calling (LangChain + `@tool`) |
| 9 | Evaluación automática: latencia, citación de fuentes y detección de tool calling |

## Modelos utilizados

| Uso | Modelo |
|-----|--------|
| Generación de respuestas | `gemini-2.5-flash` |
| Embeddings | `models/gemini-embedding-001` |

## Características principales

- **RAG con metadatos**: los chunks del PDF se enriquecen con zona (`Norte`/`Sur`) y tipo (`Sitio de interés`/`Restaurante`), que el modelo cita en cada respuesta
- **Function calling**: el LLM decide de forma autónoma cuándo invocar `get_weather`, sin instrucciones explícitas por turno
- **Historial multiturno**: ventana deslizante de 2 turnos para mantener coherencia sin superar el límite de contexto
- **Logging de herramientas**: cada invocación a `get_weather` queda registrada en `logs/tool_calls.log`
- **Evaluación con ground truth**: conjunto fijo de 9 prompts con métricas de precisión comparadas contra valores esperados

## Dependencias principales

```
google-genai       # Cliente oficial de Google Generative AI
langchain          # Orquestación LLM y function calling
langchain-chroma   # Integración ChromaDB
pdfplumber         # Extracción de texto del PDF
plotly             # Visualización de métricas
streamlit          # Interfaz web (bonus)
```

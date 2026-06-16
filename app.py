import streamlit as st
st.set_page_config(page_title="Asistente Turístico de Tenerife", layout="wide", initial_sidebar_state="expanded")

import logging
import pandas as pd
from src.rag import extraer_texto_pdf, create_chunks, clasificar_chunks_por_estructura, GeminiEmbeddingFunction
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pathlib import Path
from langchain.chat_models import init_chat_model
from src.tools import get_weather
from src.chat import chat_with_tools
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os
from google import genai

# Configuramos el formato del log para que muestre la hora, el nivel de severidad y el mensaje
logger = logging.getLogger("tool_calls")
logger.setLevel(logging.INFO)
logger.propagate = False  # evita que suba al root logger y salga por pantalla
LOG_PATH = Path(__file__).parent / "logs" / "tool_calls.log"
handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

logging.getLogger("httpx").setLevel(logging.WARNING)

## Configuracion
# Cargar variables de entorno desde el archivo .env
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Falta GOOGLE_API_KEY en el archivo .env")
    st.stop()

# Configuración de modelos
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Parametros modelo para Verificación
VERIFY_MAX_TOKENS = 40
VERIFY_TEMPERATURE = 0.1

# Parametros modelo para Asistente
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.7

# Configuracion del pdf path
PDF_PATH = Path(__file__).parent / "data" / "TENERIFE.pdf"

# Configuracion de la ruta de persistencia para Chroma
CHROMA_PATH = Path(__file__).parent / "data" / "chromadb"

# Creacion del cliente
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# inicializar vector store
@st.cache_resource
def inicializar_vector_store():
    '''Inicializar vector store'''
    embeddings = GeminiEmbeddingFunction(client, model_name=EMBEDDING_MODEL)
    if CHROMA_PATH.exists():
        return Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embeddings)
    
    pdf_text = extraer_texto_pdf(PDF_PATH)
    chunks = create_chunks(pdf_text)
    chunks_con_metadata = clasificar_chunks_por_estructura(chunks)
    return Chroma.from_documents(
        documents=chunks_con_metadata,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

@st.cache_resource
def inicializar_modelo():
    modelo = init_chat_model(GENERATION_MODEL, model_provider="google_genai")
    return modelo.bind_tools([get_weather])

vector_store = inicializar_vector_store()
modelo_con_tools = inicializar_modelo()

PREGUNTAS_RAPIDAS = [
    "¿Qué tiempo hace en el Sur?",
    "¿Dónde comer en el Norte?",
    "¿Qué ver en Santa Cruz?",
    "¿Cuál es el mejor parque acuático?",
]

# Session state
if "historial" not in st.session_state:
    st.session_state.historial = []
if "pregunta_rapida" not in st.session_state:
    st.session_state.pregunta_rapida = None

# Sidebar
with st.sidebar:
    st.title("Tenerife")
    zonas = pd.DataFrame({
        "lat": [28.39, 28.05],
        "lon": [-16.52, -16.71],
    })
    st.map(zonas, zoom=9)
    st.caption("Norte · Sur")
    st.divider()
    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.historial = []
        st.rerun()

# Título
st.title("Asistente Turístico de Tenerife")

# Mostrar mensajes previos
for mensaje in st.session_state.historial:
    rol = "user" if isinstance(mensaje, HumanMessage) else "assistant"
    with st.chat_message(rol):
        st.markdown(mensaje.content)

# Botones de preguntas rápidas (solo cuando el chat está vacío)
if not st.session_state.historial:
    st.markdown("**Prueba a preguntar:**")
    cols = st.columns(len(PREGUNTAS_RAPIDAS))
    for col, sugerida in zip(cols, PREGUNTAS_RAPIDAS):
        if col.button(sugerida, use_container_width=True):
            st.session_state.pregunta_rapida = sugerida
            st.rerun()

# Input del usuario (teclado o botón rápido)
pregunta_rapida = st.session_state.pop("pregunta_rapida", None)
input_usuario = st.chat_input("¿Qué quieres saber sobre Tenerife?")
if pregunta := (pregunta_rapida or input_usuario):
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            gen, historial_parcial = chat_with_tools(
                pregunta, modelo_con_tools, st.session_state.historial, vector_store, stream=True
            )
        texto_completo = st.write_stream(gen)
        historial_parcial.append(AIMessage(content=texto_completo))
        st.session_state.historial = historial_parcial
import logging
from rag import extraer_texto_pdf, create_chunks, clasificar_chunks_por_estructura, GeminiEmbeddingFunction
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pathlib import Path
from langchain.chat_models import init_chat_model
from src.tools import get_weather
import streamlit as st
from chat import chat_with_tools
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Configuramos el formato del log para que muestre la hora, el nivel de severidad y el mensaje
logger = logging.getLogger("tool_calls")
logger.setLevel(logging.INFO)
logger.propagate = False  # evita que suba al root logger y salga por pantalla
handler = logging.FileHandler("../logs/tool_calls.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

logging.getLogger("httpx").setLevel(logging.WARNING)

## Configuracion
# Cargar variables de entorno desde el archivo .env
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Google API key: ")

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
PDF_PATH = Path("../data/TENERIFE.pdf")

# Configuracion de la ruta de persistencia para Chroma
CHROMA_PATH = Path("../data/chromadb/")

# inicializar vector store
@st.cache_resource
def inicializar_vector_store():
    '''Inicializar vector store'''
    pdf_text = extraer_texto_pdf(PDF_PATH)
    chunks = create_chunks(pdf_text)
    chunks_con_metadata = clasificar_chunks_por_estructura(chunks)
    embeddings = GeminiEmbeddingFunction(client, model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks_con_metadata,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vector_store

@st.cache_resource
def inicializar_modelo():
    modelo = init_chat_model(GENERATION_MODEL, model_provider="google_genai")
    return modelo.bind_tools([get_weather])

vector_store = inicializar_vector_store()
modelo_con_tools = inicializar_modelo()

# Session state
if "historial" not in st.session_state:
    st.session_state.historial = []

# Título
st.title("Asistente Turístico de Tenerife")

# Mostrar mensajes previos
for mensaje in st.session_state.historial:
    rol = "user" if isinstance(mensaje, HumanMessage) else "assistant"
    with st.chat_message(rol):
        st.markdown(mensaje.content)

# Input del usuario
if pregunta := st.chat_input("¿Qué quieres saber sobre Tenerife?"):
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            respuesta, st.session_state.historial = chat_with_tools(
                pregunta,
                modelo_con_tools,
                st.session_state.historial,
                vector_store
            )
        st.markdown(respuesta)
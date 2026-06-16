from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List
import pdfplumber
from pathlib import Path

class GeminiEmbeddingFunction(Embeddings):
    """Adaptador entre la API de Gemini y ChromaDB — implementa la interfaz
    EmbeddingFunction para que ChromaDB pueda generar embeddings con Gemini 
    de forma transparente."""
    def __init__(self, api_client: genai.Client, model_name: str = EMBEDDING_MODEL):
        """
        Inicializamos el cliente de Google GenAI.
        """
        # El cliente busca automáticamente os.environ["GEMINI_API_KEY"]
        self.client = api_client
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Método que LangChain llama para vectorizar los chunks"""
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts
        )
        return [embedding.values for embedding in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Método que LangChain llama para vectorizar la pregunta del usuario"""
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return response.embeddings[0].values
    
def create_chunks(text, 
                  separators:list = ["\n\n", "\n•", "\n▪", "•", "▪", "\n", " ", ""], 
                  chunk_size:int = 1000, 
                  overlap:int = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators
    )
    chunks = text_splitter.create_documents([text])
    print(f"Se han creado {len(chunks)} chunks de texto para el embedding.")
    return chunks  



def clasificar_chunks_por_estructura(chunks_puros: List[Document]) -> List[Document]:
    """
    Recorre los chunks secuencialmente y les asigna metadatos de 'zona' y 'tipo'
    basándose en el contenido del texto (máquina de estados).
    """
    # Variables de estado iniciales
    zona_actual = "Desconocida"
    tipo_actual = "Desconocido"
    
    chunks_procesados = []
    
    for chunk in chunks_puros:
        # Convertimos a mayúsculas para que la búsqueda sea insensible a mayúsculas/minúsculas
        texto_clean = chunk.page_content.upper()
        
        # Detectar cambios de ZONA
        if "ZONA NORTE" in texto_clean:
            zona_actual = "Norte"
        elif "ZONA SUR" in texto_clean:
            zona_actual = "Sur"
            
        # Detectar cambios de TIPO
        if "SITIOS QUE VER" in texto_clean:
            tipo_actual = "Sitio de interes"
        elif "RESTAURANTES" in texto_clean:
            tipo_actual = "Restaurante"
            
        # Clonar las metadatos existentes del chunk (por si el PDF traía número de página, etc.)
        # y añadirle nuestras nuevas etiquetas de zona y tipo.
        nuevos_metadatos = chunk.metadata.copy()
        nuevos_metadatos.update({
            "zona": zona_actual,
            "tipo": tipo_actual
        })
        
        # Crear el nuevo documento enriquecido
        doc_etiquetado = Document(
            page_content=chunk.page_content,
            metadata=nuevos_metadatos
        )
        
        chunks_procesados.append(doc_etiquetado)
        
    return chunks_procesados

## Función para buscar los chunks relevantes en ChromaDB 
def buscar_chunks_relevantes(query: str, vector_store, k: int = 3) -> str:
    """
    Busca los k chunks más relevantes en ChromaDB para la query dada.
    Devuelve un string concatenado de los resultados con metadatos de zona y tipo.
    """
    resultados = vector_store.similarity_search(query, k=k)
    
    # Unimos los fragmentos inyectando la zona y el tipo directamente antes del texto
    contexto_concatenado = "\n\n".join([f"[{res.metadata.get('zona')} - {res.metadata.get('tipo')}]: {res.page_content}" for res in resultados])
    
    return contexto_concatenado



def construir_mensajes(pregunta_actual: str, contexto_rag: str, historial_conversacion: list[dict] = None) -> list[dict]:
    """
    Construye la lista de mensajes para el modelo de chat.
    [SystemMessage] + historial_conversacion + [HumanMessage(contexto + pregunta)]    """
    if historial_conversacion is None:
        historial_conversacion = []

    system_prompt = (
    "Eres un asistente virtual de viajes experto y amable para viajeros que quieren saber de Tenerife.\n"
    "Tu misión es responder a la pregunta del usuario utilizando ÚNICAMENTE el contexto proporcionado.\n"
    "CRÍTICO: Al final de tu respuesta cita las fuentes usadas. Ejemplo: 'Fuentes: [Sur - Sitio de interes], [Norte - Restaurante]'\n"
    "Si en el contexto no encuentras la respuesta, di amablemente que no dispones de esa información.\n\n"
    )
    
    # Iniciamos la lista con el SystemMessage
    mensajes = [SystemMessage(content=system_prompt)]
    # Sumamos el historial de conversación
    mensajes += historial_conversacion
    
    # Construimos el ÚLTIMO HumanMessage empaquetando el contexto RAG y la pregunta actual
    contenido_humano_final = (
        f"CONTEXTO:\n{contexto_rag}\n\n"f"Pregunta: {pregunta_actual}"
    )

    # Añadimos el mensaje final a la lista
    mensajes += [HumanMessage(content=contenido_humano_final)]

    return mensajes


def controlar_ventana_tokens(historial_conversacion: list, max_turnos: int = 3) -> list:
    """
    Controla la ventana de tokens del modelo limitando el historial por turnos.
    Garantiza la coherencia eliminando siempre pares completos (HumanMessage + AIMessage)
    desde el inicio del hilo de la conversación.
    """
    # Cada turno de chat (pregunta + respuesta) equivale a 2 mensajes individuales
    max_mensajes = max_turnos * 2
    
    # Si el volumen de mensajes supera el límite de la ventana, recortamos por el principio
    while len(historial_conversacion) > max_mensajes:
        historial_conversacion.pop(0)  # Elimina el HumanMessage más antiguo
        historial_conversacion.pop(0)  # Elimina el AIMessage más antiguo
        
    return historial_conversacion

def extraer_texto_pdf(PDF_PATH: Path) -> str:
    with pdfplumber.open(PDF_PATH) as pdf:
        # pdf.pages es una lista de páginas
        # cada página tiene un método .extract_text()
        content = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                print(f"Texto de la página {i + 1}:\n{text[0:50]}\n")
                content.append(text)
            else:
                print(f"No se pudo extraer texto de la página {i + 1}.")
        print(f"Se extrajo texto de {len(pdf.pages)} páginas del PDF.")

    # Unir y devolver todo el contenido extraído en un solo string
    return "\n".join(content)
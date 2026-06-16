from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from rag import controlar_ventana_tokens, buscar_chunks_relevantes, construir_mensajes
from tools import get_weather
import logging

logger = logging.getLogger("tool_calls")


def chat_with_tools(pregunta: str, model_con_tools, historial: list, vector_store) -> tuple[str, list]:
    """
    Función principal adaptada para soportar Tool Calling de forma reactiva.
    """
 
    # Control de la ventana de tokens (Mantiene el historial limpio antes de procesar)
    historial = controlar_ventana_tokens(historial, max_turnos=2)

    # Obtener contexto del RAG por si la pregunta es sobre información general
    contexto = buscar_chunks_relevantes(pregunta, k=3)
    
    # Construir la lista de mensajes con la estructura correcta
    lista_mensajes = construir_mensajes(
        pregunta_actual=pregunta, 
        contexto_rag=contexto, 
        historial_conversacion=historial
    )
    
    # Primera llamada al modelo (el modelo decidirá si responde o usa la Tool)
    respuesta_llm = model_con_tools.invoke(lista_mensajes)

    # FLUJO CON TOOL CALLING: ¿El modelo ha solicitado usar una herramienta
    texto_respuesta = ""
    if respuesta_llm.tool_calls:
        # Registro de intentos 
        logger.info(f" [TOOL CALL DETECTADA] El LLM llama a la herramienta 'get_weather'")
        
        # El modelo no devuelve texto directo, devuelve una petición de ejecución
        for tool_call in respuesta_llm.tool_calls:
            if tool_call["name"] == "get_weather":
                # Ejecutamos la función real en Python con los argumentos que extrajo Gemini
                argumentos = tool_call["args"]
                resultado_clima = get_weather.invoke({"zone": argumentos.get("zone")})                
                # Para que el flujo continúe, debemos añadir la petición del modelo 
                # y la respuesta de la herramienta a la lista de mensajes
                lista_mensajes.append(respuesta_llm) 
                lista_mensajes.append(
                    ToolMessage(
                        content=str(resultado_clima), 
                        tool_call_id=tool_call["id"]
                    )
                )
                
                # Segunda llamada al modelo: Ahora Gemini lee los datos de Open-Meteo y redacta la respuesta final
                segunda_respuesta = model_con_tools.invoke(lista_mensajes)
                content = segunda_respuesta.content
                texto_respuesta = content[0]["text"] if isinstance(content, list) else content
                if "[get_weather]" not in texto_respuesta:
                    texto_respuesta += "\n\nFuentes: [get_weather]"

    else:
        # El modelo respondió directamente usando solo el contexto del RAG
        content = respuesta_llm.content
        texto_respuesta = content[0]["text"] if isinstance(content, list) else content
    

    # Guardamos la pregunta limpia y la respuesta final (ocultando la Tool)
    # para que los próximos turnos mantengan una coherencia de chat perfecta.
    historial.append(HumanMessage(content=pregunta))
    historial.append(AIMessage(content=texto_respuesta))
    
    return  texto_respuesta, historial
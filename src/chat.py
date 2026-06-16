from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.rag import controlar_ventana_tokens, buscar_chunks_relevantes, construir_mensajes
from src.tools import get_weather
import logging

logger = logging.getLogger("tool_calls")


def chat_with_tools(pregunta: str, model_con_tools, historial: list, vector_store, stream: bool = False):
    """
    Función principal adaptada para soportar Tool Calling de forma reactiva.
    stream=False → devuelve (str, list)
    stream=True  → devuelve (generator, list) donde el generator yield chunks de texto
    """
    historial = controlar_ventana_tokens(historial)
    contexto = buscar_chunks_relevantes(pregunta, vector_store, k=3)
    lista_mensajes = construir_mensajes(
        pregunta_actual=pregunta,
        contexto_rag=contexto,
        historial_conversacion=historial
    )

    # Primera llamada: siempre invoke (necesitamos leer .tool_calls de la respuesta completa)
    respuesta_llm = model_con_tools.invoke(lista_mensajes)

    if respuesta_llm.tool_calls:
        logger.info(" [TOOL CALL DETECTADA] El LLM llama a la herramienta 'get_weather'")

        for tool_call in respuesta_llm.tool_calls:
            if tool_call["name"] == "get_weather":
                argumentos = tool_call["args"]
                resultado_clima = get_weather.invoke({"zone": argumentos.get("zone")})
                lista_mensajes.append(respuesta_llm)
                lista_mensajes.append(
                    ToolMessage(content=str(resultado_clima), tool_call_id=tool_call["id"])
                )

                if stream:
                    # Segunda llamada en modo stream
                    def _gen_tool():
                        acumulado = ""
                        for chunk in model_con_tools.stream(lista_mensajes):
                            content = chunk.content
                            texto = content[0]["text"] if isinstance(content, list) and content else content
                            if texto:
                                acumulado += texto
                                yield texto
                        if "[get_weather]" not in acumulado:
                            yield "\n\nFuentes: [get_weather]"

                    historial.append(HumanMessage(content=pregunta))
                    return _gen_tool(), historial
                else:
                    segunda_respuesta = model_con_tools.invoke(lista_mensajes)
                    content = segunda_respuesta.content
                    texto_respuesta = content[0]["text"] if isinstance(content, list) else content
                    if "[get_weather]" not in texto_respuesta:
                        texto_respuesta += "\n\nFuentes: [get_weather]"
    else:
        if stream:
            # Respuesta directa RAG en modo stream
            def _gen_rag():
                for chunk in model_con_tools.stream(lista_mensajes):
                    content = chunk.content
                    texto = content[0]["text"] if isinstance(content, list) and content else content
                    if texto:
                        yield texto

            historial.append(HumanMessage(content=pregunta))
            return _gen_rag(), historial
        else:
            content = respuesta_llm.content
            texto_respuesta = content[0]["text"] if isinstance(content, list) else content

    historial.append(HumanMessage(content=pregunta))
    historial.append(AIMessage(content=texto_respuesta))
    return texto_respuesta, historial
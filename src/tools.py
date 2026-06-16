import requests
from langchain_core.tools import tool

def api_openmeteo(latitude: float, longitude: float) -> dict:
    """
    Consulta la API de Open-Meteo y devuelve el pronóstico de 7 días 
    estructurado por fechas en formato JSON.
    
    NOTA PARA EL MODELO: Si la respuesta contiene la clave 'error', 
    informa amablemente al usuario de que el servicio externo no está disponible.
    """

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "precipitation_probability"],
        "timezone": "Atlantic/Canary"  # Forzamos la hora local de Tenerife
    }

    try:
        # Petición directa y verificación de errores HTTP (ej: 404, 500)
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Parseamos a JSON puro
        data = response.json()
        
        # Extraemos las listas del JSON
        lista_horas = data["hourly"]["time"]                     # Lista de strings: ["2026-06-15T00:00", ...]
        lista_temps = data["hourly"]["temperature_2m"]           # Lista de floats
        lista_lluvia = data["hourly"]["precipitation_probability"] # Lista de ints
        
        # Agrupamos los datos por día usando un diccionario nativo
        pronostico_semanal = {}

        # Separamos por la 'T' para agrupar los datos horarios bajo una misma clave de fecha (YYYY-MM-DD)
        for hora, temp, lluvia in zip(lista_horas, lista_temps, lista_lluvia):
            fecha_dia = hora.split("T")[0]

            # Si es la primera vez que vemos este día, inicializamos su estructura
            if fecha_dia not in pronostico_semanal:
                pronostico_semanal[fecha_dia] = {
                    "temperaturas": [],
                    "probabilidades_lluvia": []
                }
            
            # Acumulamos los valores de cada hora de ese día
            pronostico_semanal[fecha_dia]["temperaturas"].append(temp)
            pronostico_semanal[fecha_dia]["probabilidades_lluvia"].append(lluvia)

        # Reducimos los datos a las métricas clave (Max, Min, Lluvia) para ahorrar tokens
        resumen_final = {}
        for fecha_dia, datos in pronostico_semanal.items():
            temp_max = max(datos["temperaturas"])
            temp_min = min(datos["temperaturas"])
            lluvia_max = max(datos["probabilidades_lluvia"])
            
            # Reducimos las 24 horas a métricas clave del día para optimizar la ventana de 
            # contexto del LLM y ahorrar tokens
            resumen_final[fecha_dia] = {
                "resumen_del_dia": f"Max: {temp_max:.1f}°C, Min: {temp_min:.1f}°C, Lluvia máx: {lluvia_max}%"
            }
                
        return {
            "unidad_temperatura": "°C",
            "unidad_precipitacion": "%",
            "pronostico_7_dias": resumen_final
            }
    except Exception as e:
        return {"error": f"Fallo al conectar con el servicio meteorológico: {str(e)}"}


@tool
def get_weather(zone: str) -> dict:
    """
    Obtiene el pronóstico meteorológico detallado para los próximos 7 días 
    en una zona específica de la isla de Tenerife.

    Esta función actúa como una herramienta para el modelo de lenguaje (LLM). 
    Debe ser invocada siempre que el usuario pregunte por el clima actual, 
    la predicción para mañana, las condiciones generales de la semana o si 
    busca recomendaciones de actividades basadas en el tiempo en Tenerife.

    Args:
        zona (str): Región geográfica de Tenerife para la consulta. 
            Debe ser estrictamente uno de los siguientes valores:
            - "Norte": Puerto de la Cruz, La Orotava, Icod, etc.
            - "Sur": Los Cristianos, Las Américas, Costa Adeje, etc.
    Returns:
        dict: Un diccionario JSON con el pronóstico de 7 días. Cada clave es una 
        fecha (DD-MM-YYYY) y su valor contiene un resumen con temperaturas (Max/Min) 
        y la probabilidad máxima de precipitación.
        
        Si ocurre un error de red o la zona no es válida, devuelve un diccionario 
        con la clave 'error' y la descripción del fallo.

    Instrucciones para el LLM:
        1. Analiza el JSON devuelto y mapea de forma autónoma la fecha que el 
           usuario solicita (hoy, mañana, fin de semana) con las claves del diccionario.
        2. Si la respuesta contiene la clave 'error', no inventes datos climáticos; 
           informa al usuario con amabilidad de que el servicio meteorológico no está disponible.
    """
    coordenadas = {
    "norte": {"lat": 28.39, "lon": -16.52},
    "sur": {"lat": 28.05, "lon": -16.71},
    }
    if zone.lower() in coordenadas:
        lat = coordenadas[zone.lower()]["lat"]
        lon = coordenadas[zone.lower()]["lon"]
    else:
         return { "error": f"La zona '{zone}' no está registrada. "
                   f"Las zonas válidas son: {list(coordenadas.keys())}." }
        

    resultado_clima = api_openmeteo(
        latitude=lat, 
        longitude=lon
        )
    
    return resultado_clima
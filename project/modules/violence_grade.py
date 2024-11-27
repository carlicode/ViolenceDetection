import pandas as pd
import numpy as np

def calculate_violence_degree(row, weights, epsilon=0.001):
    """
    Calcula el grado de violencia para un fragmento dado.
    
    Args:
        row (pd.Series): Fila de un DataFrame con datos del fragmento.
        weights (dict): Pesos asignados a cada etiqueta.
        epsilon (float): Valor pequeño para evitar divisiones por cero.
    
    Returns:
        float: Grado de violencia calculado.
    """
    violence_score = 0
    for label, weight in weights.items():
        probability = row[label]
        
        # Asegurar que la columna 'Distancia' sea de tipo float
        distance = row["Distancia"] if row["Distancia"] != "" else 1.0  # Máxima distancia si está vacía
        distance = float(distance)  # Convertir la distancia a float
        
        # Asegurarse de que la distancia sea positiva y dentro de un rango razonable
        distance = abs(distance)  # Convertir a valor absoluto
        
        # Reemplazar valores extremadamente altos con un valor razonable, por ejemplo, 1.0
        if distance > 100:  # Umbral arbitrario para detectar valores atípicos
            distance = 1.0

        # Calcular el puntaje de violencia
        weight_distance = 1 / (distance + epsilon)
        violence_score += weight * probability * weight_distance
    return violence_score

def process_violence_data(file_path, weights):
    """
    Procesa un archivo Excel para calcular el grado de violencia.
    
    Args:
        file_path (str): Ruta del archivo Excel.
        weights (dict): Pesos asignados a las etiquetas.
    
    Returns:
        pd.DataFrame: DataFrame con los datos procesados y el grado de violencia calculado.
    """
    # Cargar datos del archivo
    df = pd.read_excel(file_path).fillna("")

    # Asegurarse de que la columna 'Distancia' sea tipo float
    df["Distancia"] = pd.to_numeric(df["Distancia"], errors="coerce").fillna(1.0)  # Convertir a float y reemplazar NaN por 1.0
    
    # Limitar los valores atípicos de 'Distancia'
    df["Distancia"] = df["Distancia"].apply(lambda x: abs(x) if x > 0 else 1.0)  # Garantizar que 'Distancia' sea positiva y reemplazar valores erróneos
    
    # Calcular grado de violencia para cada fragmento
    df["Grado de Violencia"] = df.apply(lambda row: calculate_violence_degree(row, weights), axis=1)
    
    return df

# Función para predecir la evolución de la violencia
def predict_violence_evolution(df, label_weights):
    """
    Predice la evolución de la violencia basándose en los datos procesados.

    Args:
        df (pd.DataFrame): Datos procesados con el grado de violencia.
        label_weights (dict): Pesos para las etiquetas.

    Returns:
        tuple: (Resultado de la predicción, desglose del cálculo)
    """
    # Convertir el "Tiempo del Fragmento" a segundos (asumimos que está en formato "X,Y")
    df["Tiempo (segundos)"] = df["Tiempo del Fragmento"].apply(lambda x: float(x.split(',')[0]))
    
    # Obtener los últimos fragmentos correspondientes a los últimos 10 segundos
    last_10_seconds = df[df["Tiempo (segundos)"] >= df["Tiempo (segundos)"].max() - 10]
    
    if len(last_10_seconds) >= 5:
        recent_fragments = last_10_seconds["Grado de Violencia"].tail(10)  # Usamos los últimos 10 segundos
    elif len(last_10_seconds) >= 5:
        recent_fragments = last_10_seconds["Grado de Violencia"].tail(5)  # Usamos los últimos 5 si hay menos de 10
    else:
        return "Datos insuficientes para realizar la predicción.", {}

    # Calcular la tendencia
    x = np.arange(len(recent_fragments))
    y = recent_fragments.values
    slope, _ = np.polyfit(x, y, 1)

    # Promedio reciente del grado de violencia
    avg_recent_violence = recent_fragments.mean()

    # Evaluar eventos críticos
    critical_events = df[["gun_shot", "screams", "glass_breaking"]].tail(5)
    high_critical_events = (critical_events > 50).sum().sum()

    # Ponderar factores
    alpha, beta, gamma = 0.5, 0.4, 0.2
    evolution_score = (
        alpha * avg_recent_violence +
        beta * slope +
        gamma * high_critical_events)/100

    # Clasificar evolución
    result = ""
    if evolution_score > 0.5:
        result = "Es probable que la violencia evolucione."
    elif 0.2 <= evolution_score <= 0.5:
        result = "La violencia parece mantenerse estable."
    else:
        result = "Es probable que la violencia disminuya."

    # Desglose del cálculo
    breakdown = {
        "Promedio Grado de Violencia Reciente": avg_recent_violence,
        "Pendiente de la Tendencia": slope,
        "Eventos Críticos Altos": high_critical_events,
        "Puntaje de Evolución": evolution_score,
        "Clasificación Final": result
    }

    return result, breakdown

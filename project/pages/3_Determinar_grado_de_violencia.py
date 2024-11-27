import streamlit as st
import os
import matplotlib.pyplot as plt
from modules.violence_grade import process_violence_data, predict_violence_evolution

# Configuración del directorio de histórico
HISTORIC_DIR = '/histórico'

# Pesos para las etiquetas
LABEL_WEIGHTS = {
    "crying": 0.2,
    "glass_breaking": 0.5,
    "gun_shot": 1.0,
    "people_talking": 0.1,
    "screams": 0.7
}

# Título de la página
st.title("Determinación del Grado de Violencia")

# Explicación del cálculo del grado de violencia
st.write("""
### ¿Cómo se calcula el grado de violencia?
El grado de violencia se calcula utilizando las probabilidades de las etiquetas detectadas, la distancia entre los embeddings de los fragmentos y los pesos asignados a cada etiqueta. Estos elementos se combinan en la fórmula para generar un valor numérico que representa el grado de violencia en un fragmento de audio o video.
""")

st.latex(r"""
Grado\ de\ Violencia_{episodio} = \sum_{i=1}^{n} \left( Peso_{i} \cdot Probabilidad_{i} \cdot \frac{1}{Distancia_{i} + \epsilon} \right)
""")

st.write("""
### Escala de Grado de Violencia
- **0-50**: Baja violencia (verde).
- **50-200**: Moderada violencia (amarillo).
- **200-500**: Alta violencia (rojo).
""")

st.write("""
**1. Fórmula del grado de violencia**  
Esta fórmula calcula el grado de violencia como una suma ponderada de los eventos detectados.  
- **Peso**: Representa la relevancia o severidad de cada etiqueta (por ejemplo, "gun_shot" tiene un peso mayor que "people_talking").  
- **Probabilidad**: Es la probabilidad de que un evento específico esté presente en el fragmento, según el modelo de detección.  
- **Distancia**: La distancia entre los embeddings del fragmento y un punto de referencia. Valores más cercanos indican mayor similitud. Para evitar divisiones por cero, se suma un pequeño valor **epsilon**.

**2. Promedio Grado de Violencia Reciente**  
Este valor es el promedio del grado de violencia en los últimos 5 fragmentos procesados. Refleja la intensidad promedio reciente.  

**3. Pendiente de la Tendencia**  
Se calcula utilizando regresión lineal para identificar si el grado de violencia está aumentando o disminuyendo. Una pendiente positiva sugiere que la violencia está evolucionando, mientras que una pendiente negativa indica una disminución.  

**4. Eventos Críticos Altos**  
Este indicador cuenta cuántos eventos críticos como disparos, gritos o vidrios rotos superan un umbral de probabilidad del 50% en los últimos 5 fragmentos.  

**5. Puntaje de Evolución**  
El puntaje combina todas las métricas anteriores utilizando una fórmula ponderada:  
\[
Puntaje\ de\ Evolución = \alpha \cdot Promedio\ Grado\ de\ Violencia + \beta \cdot Pendiente\ de\ la\ Tendencia + \gamma \cdot Eventos\ Críticos\ Altos
\]
Donde **α, β y γ** son pesos asignados para balancear la importancia de cada métrica.

**6. Clasificación Final**  
Basado en el puntaje de evolución, se determina si la violencia probablemente evoluciona, se mantiene estable o disminuye.
""")

st.write("""
### Predicción de violencia
""")

st.latex(r"""
Puntaje\ de\ Evolución = \frac{\alpha \cdot Promedio + \beta \cdot Pendiente + \gamma \cdot Eventos\ Críticos}{100}
""")

st.write("""
El **Puntaje de Evolución** se calcula como una combinación ponderada de tres métricas clave: 
el promedio reciente del grado de violencia, la pendiente de la tendencia de la violencia y el número de eventos críticos detectados. 
Esta combinación se divide entre 100 para normalizar el resultado. La fórmula es:

""")
st.write("""
Donde:
- **Promedio**: Es la media del grado de violencia calculado en los últimos fragmentos.
- **Pendiente**: Representa la dirección y magnitud del cambio en el grado de violencia reciente.
- **Eventos Críticos**: Es el número de eventos graves (como disparos o gritos) detectados en los fragmentos recientes, ponderados por su impacto.
- **α, β, γ**: Son los pesos asignados a cada componente, ajustados para reflejar su importancia relativa en la evaluación.

El resultado final clasifica la evolución de la violencia en tres categorías:
- Si el puntaje es mayor a 0.5: **Es probable que la violencia evolucione.**
- Si el puntaje está entre 0.2 y 0.5: **La violencia parece mantenerse estable.**
- Si el puntaje es menor a 0.2: **Es probable que la violencia disminuya.**
""")

# Clasificar los valores de 'Grado de Violencia' en los rangos definidos
def categorize_violence(degree):
    if degree <= 50:
        return "Baja Violencia"
    elif degree <= 200:
        return "Violencia Moderada"
    elif degree <= 500:
        return "Alta Violencia"
    else:
        return "Fuera de Rango"
    
# Verificar si el directorio de histórico contiene archivos
if not os.path.exists(HISTORIC_DIR):
    st.error(f"La carpeta de histórico no existe: {HISTORIC_DIR}")
else:
    # Obtener la lista de archivos XLSX en el directorio
    xlsx_files = [file for file in os.listdir(HISTORIC_DIR) if file.endswith(".xlsx")]

    if len(xlsx_files) == 0:
        st.warning("No hay archivos de histórico disponibles en la carpeta.")
    else:
        # Selector de archivo
        selected_file = st.selectbox("Selecciona un archivo de histórico:", xlsx_files)

        # Mostrar el contenido del archivo seleccionado
        if selected_file:
            file_path = os.path.join(HISTORIC_DIR, selected_file)
            try:
                # Procesar los datos para calcular el grado de violencia
                df = process_violence_data(file_path, LABEL_WEIGHTS)

                # Mostrar los datos procesados
                st.write(f"Mostrando datos procesados para: **{selected_file}**")
                st.dataframe(df)

                # Graficar el grado de violencia a través del tiempo
                time_intervals = df["Tiempo del Fragmento"].str.split(",", expand=True).astype(float)
                plt.figure(figsize=(10, 6))

                # Colorear el fondo del gráfico según los niveles
                plt.axhspan(0, 50, color="green", alpha=0.1, label="Baja Violencia")
                plt.axhspan(50, 200, color="yellow", alpha=0.1, label="Violencia Moderada")
                plt.axhspan(200, 500, color="red", alpha=0.1, label="Alta Violencia")

                # Trazar el grado de violencia
                plt.plot(time_intervals[0], df["Grado de Violencia"], color="blue", label="Grado de Violencia")

                # Líneas horizontales para los límites
                plt.axhline(50, color="green", linestyle="--", linewidth=1)
                plt.axhline(200, color="orange", linestyle="--", linewidth=1)

                # Configuración del gráfico
                plt.title("Grado de Violencia a través del Tiempo")
                plt.xlabel("Tiempo (segundos)")
                plt.ylabel("Grado de Violencia")
                plt.xticks(range(0, int(time_intervals[0].max()) + 1, 2))  # Intervalos de 2 en 2
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)
                
                # Aplicar la clasificación al DataFrame
                df['Categoría de Violencia'] = df['Grado de Violencia'].apply(categorize_violence)

                # Contar las frecuencias de cada categoría
                violence_category_counts = df['Categoría de Violencia'].value_counts()

                # Mostrar la categoría más frecuente dentro de cada rango
                st.write("Frecuencia de las categorías de violencia:")
                st.write(violence_category_counts)

                # Calcular predicción de la evolución de la violencia
                result, breakdown = predict_violence_evolution(df, LABEL_WEIGHTS)

                # Mostrar el resultado de la predicción
                st.write("### Resultado de la Predicción:")
                st.write(result)

                # Mostrar desglose del cálculo
                st.write("### Desglose del Cálculo:")
                st.json(breakdown)

            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")

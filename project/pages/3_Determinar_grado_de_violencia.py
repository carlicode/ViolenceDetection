import streamlit as st
import os
import matplotlib.pyplot as plt
from modules.violence_grade import process_violence_data, predict_violence_evolution

# Configuración del directorio de histórico
HISTORIC_DIR = '/Users/carli.code/Desktop/ViolenceDetection/histórico'

# Pesos para las etiquetas
LABEL_WEIGHTS = {
    "crying": 0.2,
    "glass_breaking": 0.5,
    "gun_shot": 1.0,
    "people_talking": 0.1,
    "screams": 0.7
}

# Título de la página
st.title("Determinar Grado de Violencia")

# Explicación del cálculo del grado de violencia
st.write("""
### ¿Cómo se calcula el grado de violencia?
El grado de violencia se calcula considerando las probabilidades de las etiquetas detectadas, la distancia de los embeddings y los pesos asignados a cada etiqueta.
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
### Cómo interpretar el desglose del cálculo
El desglose incluye métricas clave que explican cómo se evalúa la evolución de la violencia:
- **Promedio Grado de Violencia Reciente**: Promedio del grado de violencia en los últimos 5 fragmentos. Refleja la intensidad promedio de los eventos recientes.
- **Pendiente de la Tendencia**: Representa la dirección del cambio en el grado de violencia. Un valor positivo indica que la violencia está aumentando, mientras que un valor negativo sugiere que está disminuyendo.
- **Eventos Críticos Altos**: Número de eventos graves (por ejemplo, disparos o gritos) que superan un umbral de probabilidad en los últimos 5 fragmentos.
- **Puntaje de Evolución**: Resultado final de combinar las métricas anteriores utilizando una fórmula ponderada.
- **Clasificación Final**: Predicción basada en el puntaje de evolución, que puede indicar:
  - "Es probable que la violencia evolucione."
  - "La violencia parece mantenerse estable."
  - "Es probable que la violencia disminuya."
""")

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

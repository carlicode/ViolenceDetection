import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuración del directorio de histórico
HISTORIC_DIR = '/Users/carli.code/Desktop/ViolenceDetection/histórico'

# Título de la página
st.title("Ver Histórico")

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
                # Cargar el archivo Excel
                df = pd.read_excel(file_path)

                # Reemplazar valores None por cadenas vacías
                df = df.fillna("")

                # Mostrar el archivo
                st.write(f"Mostrando el contenido de: **{selected_file}**")
                st.dataframe(df)

                # Filtrar las columnas de interés
                columns_of_interest = ["crying", "glass_breaking", "gun_shot", "people_talking", "screams"]
                if all(col in df.columns for col in columns_of_interest):
                    df_filtered = df[columns_of_interest]
                    
                    # Crear una columna de tiempo a partir del índice
                    time_intervals = df["Tiempo del Fragmento"].str.split(",", expand=True).astype(float)
                    df_filtered["Tiempo"] = time_intervals[0]  # Usar la columna inicial del tiempo
                    
                    # Gráfico 1: Todas las etiquetas respecto al tiempo
                    st.subheader("Distribución de todas las etiquetas respecto al tiempo")
                    plt.figure(figsize=(10, 6))
                    for col in columns_of_interest:
                        plt.plot(df_filtered["Tiempo"], df_filtered[col], label=col)

                    plt.title("Distribución de etiquetas respecto al tiempo")
                    plt.xlabel("Tiempo (segundos)")
                    plt.ylabel("Porcentaje (%)")
                    plt.xticks(range(0, int(df_filtered["Tiempo"].max()) + 1, 2))  # Intervalos de 2 en 2
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)

                    # Filtrar solo las etiquetas mayores al 20%
                    st.subheader("Etiquetas mayores al 20%")
                    plt.figure(figsize=(10, 6))
                    for col in columns_of_interest:
                        filtered_df = df_filtered[df_filtered[col] > 20]
                        if not filtered_df.empty:
                            plt.plot(filtered_df["Tiempo"], filtered_df[col], label=col)

                    plt.title("Etiquetas mayores al 20% respecto al tiempo")
                    plt.xlabel("Tiempo (segundos)")
                    plt.ylabel("Porcentaje (%)")
                    plt.xticks(range(0, int(df_filtered["Tiempo"].max()) + 1, 2))  # Intervalos de 2 en 2
                    plt.legend()
                    plt.grid(True)
                    st.pyplot(plt)

                    # Selector de umbral para el gráfico de distancia
                    st.subheader("Gráfico 3: Puntos con distancia menor al umbral seleccionado")
                    threshold = st.selectbox(
                        "Selecciona un umbral para la distancia:",
                        [0.15, 0.20, 0.25, 0.3],
                        index=1  # Umbral por defecto: 0.20
                    )

                    # Gráfico 3: Puntos donde la distancia del embedding es menor al umbral seleccionado
                    if "Distancia" in df.columns:
                        df["Distancia"] = pd.to_numeric(df["Distancia"], errors="coerce")
                        df_distance_filtered = df[df["Distancia"] < threshold]

                        plt.figure(figsize=(10, 6))
                        if not df_distance_filtered.empty:
                            plt.scatter(
                                df_distance_filtered["Tiempo del Fragmento"].str.split(",", expand=True)[0].astype(float),
                                df_distance_filtered["Distancia"],
                                color="red",
                                label=f"Distancia < {threshold}"
                            )
                            plt.title(f"Puntos con Distancia menor a {threshold} respecto al Tiempo")
                            plt.xlabel("Tiempo (segundos)")
                            plt.ylabel("Distancia")
                            plt.xticks(range(0, int(df_filtered["Tiempo"].max()) + 1, 2))  # Intervalos de 2 en 2
                            plt.legend()
                            plt.grid(True)
                            st.pyplot(plt)

                            # Mostrar la tabla de textos y distancias que cumplen el umbral
                            st.subheader("Textos y distancias que cumplen el umbral")
                            df_texts = df_distance_filtered[["Texto", "Embedding Asociado", "Distancia"]].copy()
                            df_texts["Distancia"] = df_texts["Distancia"].round(4)  # Redondear distancias
                            st.dataframe(df_texts)
                        else:
                            st.info(f"No hay puntos que cumplan con el umbral de distancia {threshold}.")
                            plt.figure(figsize=(10, 6))
                            plt.title(f"Puntos con Distancia menor a {threshold} respecto al Tiempo")
                            plt.xlabel("Tiempo (segundos)")
                            plt.ylabel("Distancia")
                            plt.xticks(range(0, int(df_filtered["Tiempo"].max()) + 1, 2))  # Intervalos de 2 en 2
                            plt.grid(True)
                            st.pyplot(plt)
                    else:
                        st.warning("La columna 'Distancia' no está disponible en el archivo.")

                else:
                    st.warning("El archivo no contiene las columnas necesarias para generar las gráficas.")

            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")

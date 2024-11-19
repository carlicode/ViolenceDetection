import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import os  # Para manejar la eliminación de archivos temporales
from keras.models import load_model
import pandas as pd
import speech_recognition as speech_recog  # Renombrar alias para evitar conflictos

# Configuración inicial
target_sampling_rate = 16000
n_mels = 128
fmax = 8000
labels = ["crying", "glass_breaking", "gun_shot", "people_talking", "screams"]

# Opciones de modelos disponibles
models = {
    "Audios originales": "/Users/carli.code/Desktop/ViolenceDetection/models/original_dataset.h5",
    "Gaussian Noise con 30 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/audio_classification_model_30.h5",
    "Gaussian Noise con 40 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/audio_classification_model_40.h5"
}

# Configuración de la app
st.title("Clasificador de fragmentos de audio")
st.write("Seleccione un modelo, cargue un archivo de audio y el modelo clasificará cada fragmento de 2 segundos.")

# Selector de modelo
model_choice = st.selectbox("Selecciona el modelo de predicción:", list(models.keys()))

# Cargar el modelo seleccionado
model_path = models[model_choice]
model = load_model(model_path)
st.write(f"Modelo seleccionado: **{model_choice}**")

# Función para dividir un audio en fragmentos de 2 segundos
def split_audio(audio, sr, duration=2):
    fragment_length = duration * sr
    num_fragments = len(audio) // fragment_length
    fragments = [audio[i * fragment_length: (i + 1) * fragment_length] for i in range(num_fragments)]
    return fragments

# Función para procesar un fragmento y generar el Mel-espectrograma
def process_fragment(fragment, sr):
    spectrogram = librosa.feature.melspectrogram(y=fragment, sr=sr, n_mels=n_mels, fmax=fmax)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram

# Función para realizar predicciones
def predict_episode(episodes, model):
    predictions = []
    for episode in episodes:
        spectrogram = process_fragment(episode, target_sampling_rate)
        spectrogram = np.expand_dims(spectrogram, axis=-1)  # Expandir dimensión para el modelo
        spectrogram = np.expand_dims(spectrogram, axis=0)   # Agregar batch size
        pred = model.predict(spectrogram)
        predictions.append(pred[0])  # Guardar la predicción
    return predictions

# Función para convertir audio a texto usando Speech-to-Text
def audio_to_text(audio_path):
    recognizer = speech_recog.Recognizer()
    try:
        with speech_recog.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language="es-ES")  # Cambiar a "en-US" para inglés
    except speech_recog.UnknownValueError:
        text = "No se pudo reconocer el audio."
    except speech_recog.RequestError as e:
        text = f"Error en el servicio de reconocimiento: {e}"
    return text

# Subir archivo de audio
uploaded_file = st.file_uploader("Sube un archivo de audio", type=["wav"])

if uploaded_file is not None:
    # Cargar y procesar el audio
    audio, sr = librosa.load(uploaded_file, sr=target_sampling_rate, mono=True)
    episodes = split_audio(audio, sr)

    # Mostrar los fragmentos procesados
    st.write(f"Se detectaron {len(episodes)} fragmentos de 2 segundos.")

    # Generar predicciones para cada fragmento
    predictions = predict_episode(episodes, model)

    # Procesar cada fragmento
    for i, (episode, pred) in enumerate(zip(episodes, predictions)):
        # Guardar el fragmento en un archivo temporal
        fragment_filename = f"fragment_{i + 1}.wav"
        sf.write(fragment_filename, episode, target_sampling_rate)

        # Reproducir el fragmento en Streamlit
        st.audio(fragment_filename, format="audio/wav")

        # Convertir el fragmento a texto
        recognized_text = audio_to_text(fragment_filename)

        # Convertir las probabilidades a porcentaje con dos decimales
        pred_percentage = [round(p * 100, 2) for p in pred]

        # Mostrar las probabilidades para cada etiqueta como tabla
        st.write(f"Fragmento {i + 1} - Distribución de probabilidad (en porcentaje):")
        pred_df = pd.DataFrame({'Label': labels, 'Probabilidad (%)': pred_percentage})
        st.table(pred_df)  # Mostrar como tabla

        # Mostrar el texto identificado
        st.write(f"Texto identificado en el fragmento {i + 1}: **{recognized_text}**")

        # Eliminar el archivo del fragmento
        os.remove(fragment_filename)

import streamlit as st
import librosa
import soundfile as sf
import os
import pandas as pd
import speech_recognition as speech_recog
from modules.chroma_query import load_chroma_db, query_chroma_db
from modules.audio_prediction import load_audio_model, split_audio, predict_episode

# Configuración del directorio para guardar el histórico
HISTORIC_DIR = '/Users/carli.code/Desktop/ViolenceDetection/histórico'
os.makedirs(HISTORIC_DIR, exist_ok=True)

# Etiquetas para las predicciones
labels = ["crying", "glass_breaking", "gun_shot", "people_talking", "screams"]

# Opciones de modelos disponibles
models = {
    "Audios originales con 20 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/original_dataset_20.h5",
    "Audios originales con 30 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/original_dataset_30.h5",
    "Audios originales con 40 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/original_dataset_40.h5",
    "Audios originales con 100 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/original_dataset_100.h5",
    "Gaussian Noise con 20 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/gaussian noise_20.h5",
    "Gaussian Noise con 30 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/gaussian noise_30.h5",
    "Gaussian Noise con 40 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/gaussian noise_40.h5",
    "Gaussian Noise con 100 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/gaussian_noise_100.h5"
}

# Cargar la base vectorial de ChromaDB
db = load_chroma_db()

# Página de Streamlit
st.title("Clasificador de Fragmentos de Audio")

# Selector de modelo
model_choice = st.selectbox("Selecciona el modelo de predicción:", list(models.keys()))
model = load_audio_model(models[model_choice])
st.write(f"Modelo seleccionado: **{model_choice}**")

# Función para convertir audio a texto usando Speech-to-Text
def audio_to_text(audio_path):
    recognizer = speech_recog.Recognizer()
    try:
        with speech_recog.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio, language="es-ES")
    except (speech_recog.UnknownValueError, speech_recog.RequestError):
        return ""

# Subir archivo de audio
uploaded_file = st.file_uploader("Sube un archivo de audio", type=["wav"])

if uploaded_file:
    audio, sr = librosa.load(uploaded_file, sr=16000, mono=True)
    episodes = split_audio(audio, sr)
    historical_data = []

    st.write(f"Se detectaron {len(episodes)} fragmentos de 2 segundos.")

    predictions = predict_episode(episodes, model)
    for i, (episode, pred) in enumerate(zip(episodes, predictions)):
        fragment_filename = f"fragment_{i + 1}.wav"
        sf.write(fragment_filename, episode, 16000)

        st.audio(fragment_filename, format="audio/wav")

        recognized_text = audio_to_text(fragment_filename)
        document, distance = query_chroma_db(db, recognized_text) if recognized_text else (None, None)

        pred_percentage = [round(p * 100, 2) for p in pred]
        fragment_time = f"{i * 2},{(i + 1) * 2}"
        historical_data.append({
            'Número de Fragmento': i + 1,
            'Tiempo del Fragmento': fragment_time,
            'Texto': recognized_text or "",
            'Embedding Asociado': document or "",
            'Distancia': f"{distance:.4f}" if distance else "",
            **{label: percentage for label, percentage in zip(labels, pred_percentage)}
        })

        st.write(f"Fragmento {i + 1}: Probabilidades")
        st.table(pd.DataFrame({'Label': labels, 'Probabilidad (%)': pred_percentage}))

        st.write(f"Fragmento {i + 1}: Texto Identificado")
        st.table(pd.DataFrame({
            'Texto': [recognized_text or ""],
            'Embedding Asociado': [document or ""],
            'Distancia': [f"{distance:.4f}" if distance else ""]
        }))

        os.remove(fragment_filename)

    # Aquí se realiza el cambio para que el archivo se guarde con el nombre del dataset y el modelo seleccionado
    dataset_name = os.path.splitext(uploaded_file.name)[0]  # El nombre base del archivo de audio
    model_name = model_choice.replace(" ", "_").replace("con", "").lower()  # Formatear el nombre del modelo
    output_filename = f"{dataset_name}_{model_name}_historico.xlsx"  # Crear el nombre del archivo
    output_path = os.path.join(HISTORIC_DIR, output_filename)
    
    pd.DataFrame(historical_data).to_excel(output_path, index=False)
    st.success(f"Histórico guardado en: {output_path}")
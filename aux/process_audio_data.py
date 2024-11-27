import os
import librosa
import pandas as pd
import soundfile as sf
from openai import OpenAI
from modules.audio_prediction import load_audio_model, split_audio, predict_episode
from modules.chroma_query import load_chroma_db, query_chroma_db

# Configuración de directorios
AUDIO_DIR = '/Users/carli.code/Desktop/ViolenceDetection/audio test'
HISTORIC_DIR = '/Users/carli.code/Desktop/ViolenceDetection/histórico'
os.makedirs(HISTORIC_DIR, exist_ok=True)

# Modelos disponibles
models = {
    
    "Gaussian Noise con 40 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/gaussian noise_40.h5",
    "Gaussian Noise con 100 epochs": "/Users/carli.code/Desktop/ViolenceDetection/models/gaussian_noise_100.h5"
}

# Cargar la base vectorial de ChromaDB
db = load_chroma_db()

#OPENAI_API_KEY = ''
client = OpenAI(api_key = OPENAI_API_KEY)

# Función para convertir audio a texto usando Whisper
def audio_to_text(audio_path):
    audio_file = open(audio_path, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file)
    return transcription.text  # Devolver el texto transcrito

# Función para procesar y generar la tabla de predicciones
def process_audio_files():
    # Recorrer todos los archivos en la carpeta de audios
    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith(".wav"):
            file_path = os.path.join(AUDIO_DIR, filename)
            print(f"Procesando archivo: {filename}")
            
            # Cargar el audio
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            episodes = split_audio(audio, sr)  # Dividir el audio en fragmentos de 2 segundos
            historical_data = []

            # Seleccionar el modelo para la predicción (por ejemplo, "Audios originales con 40 epochs")
            model_choice = "Gaussian Noise con 100 epochs"  # Usamos el modelo por defecto
            model = load_audio_model(models[model_choice])

            # Procesar los fragmentos de audio
            predictions = predict_episode(episodes, model)
            for i, (episode, pred) in enumerate(zip(episodes, predictions)):
                fragment_filename = f"fragment_{i + 1}.wav"
                sf.write(fragment_filename, episode, 16000)

                # Obtener la transcripción de cada fragmento
                recognized_text = audio_to_text(fragment_filename)  # Usamos la función de Whisper
                document, distance = query_chroma_db(db, recognized_text) if recognized_text else (None, None)

                # Guardar los datos en la lista
                pred_percentage = [round(p * 100, 2) for p in pred]
                fragment_time = f"{i * 2},{(i + 1) * 2}"
                historical_data.append({
                    'Número de Fragmento': i + 1,
                    'Tiempo del Fragmento': fragment_time,
                    'Texto': recognized_text or "",
                    'Embedding Asociado': document or "",
                    'Distancia': f"{distance:.4f}" if distance else "",
                    **{label: percentage for label, percentage in zip(["crying", "glass_breaking", "gun_shot", "people_talking", "screams"], pred_percentage)}
                })

                os.remove(fragment_filename)  # Borrar el archivo temporal

            # Guardar los resultados en un archivo CSV
            dataset_name = os.path.splitext(filename)[0]  # El nombre base del archivo de audio
            model_name = model_choice.replace(" ", "_").replace("con", "").lower()  # Formatear el nombre del modelo
            output_filename = f"{dataset_name}_{model_name}_historico.csv"  # Crear el nombre del archivo
            output_path = os.path.join(HISTORIC_DIR, output_filename)
            
            # Guardar la tabla en CSV
            pd.DataFrame(historical_data).to_csv(output_path, index=False)
            print(f"Histórico guardado en: {output_path}")

if __name__ == "__main__":
    process_audio_files()

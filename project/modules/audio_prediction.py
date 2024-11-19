import numpy as np
import librosa
import os
from keras.models import load_model

# Configuración inicial para el procesamiento de audio
target_sampling_rate = 16000
n_mels = 128
fmax = 8000

def load_audio_model(model_path):
    """
    Carga un modelo preentrenado para predicción de audio.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo no se encuentra en la ruta especificada: {model_path}")
    return load_model(model_path)

def split_audio(audio, sr, duration=2):
    """
    Divide un audio en fragmentos de duración fija.
    """
    fragment_length = duration * sr
    num_fragments = len(audio) // fragment_length
    fragments = [audio[i * fragment_length: (i + 1) * fragment_length] for i in range(num_fragments)]
    return fragments

def process_fragment(fragment, sr, n_mels=n_mels, fmax=fmax):
    """
    Genera un Mel-espectrograma a partir de un fragmento de audio.
    """
    spectrogram = librosa.feature.melspectrogram(y=fragment, sr=sr, n_mels=n_mels, fmax=fmax)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram

def predict_episode(episodes, model):
    """
    Realiza predicciones sobre una lista de fragmentos de audio.
    """
    predictions = []
    for episode in episodes:
        spectrogram = process_fragment(episode, target_sampling_rate)
        spectrogram = np.expand_dims(spectrogram, axis=-1)  # Expandir dimensión para el modelo
        spectrogram = np.expand_dims(spectrogram, axis=0)   # Agregar batch size
        pred = model.predict(spectrogram)
        predictions.append(pred[0])  # Guardar la predicción
    return predictions

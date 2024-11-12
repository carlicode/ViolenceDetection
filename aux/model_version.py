import h5py

# Abre el archivo .h5 en modo de solo lectura
file_path = "models/Ruido_2.h5"
with h5py.File(file_path, 'r') as file:
    # Imprime los atributos de nivel superior, que a veces incluyen la versión de TensorFlow/Keras
    for key, value in file.attrs.items():
        print(f"{key}: {value}")

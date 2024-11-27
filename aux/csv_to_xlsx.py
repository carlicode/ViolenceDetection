import os
import pandas as pd

# Directorios de entrada y salida
input_dir = '/Users/carli.code/Desktop/ViolenceDetection/histórico'
output_dir = '/Users/carli.code/Desktop/ViolenceDetection/histórico_excel'

# Verificar si el directorio de salida existe, si no, crear uno
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obtener la lista de archivos CSV en el directorio de entrada
csv_files = [file for file in os.listdir(input_dir) if file.endswith(".csv")]

if len(csv_files) == 0:
    print("No hay archivos CSV disponibles en la carpeta.")
else:
    # Convertir cada archivo CSV a Excel y guardarlos en la carpeta de salida
    for csv_file in csv_files:
        csv_path = os.path.join(input_dir, csv_file)
        excel_filename = os.path.splitext(csv_file)[0] + '.xlsx'  # Cambiar extensión a .xlsx
        excel_path = os.path.join(output_dir, excel_filename)

        try:
            # Cargar el archivo CSV
            df = pd.read_csv(csv_path)

            # Guardar el DataFrame como un archivo Excel
            df.to_excel(excel_path, index=False)
            print(f"Archivo convertido y guardado: {excel_path}")

        except Exception as e:
            print(f"Error al convertir el archivo {csv_file}: {e}")

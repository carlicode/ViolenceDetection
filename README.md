# Implementación de Modelo de Procesamiento de Lenguaje Natural para Predicción de Violencia Mediante Segmentación de Audio Stream

## Descripción del Proyecto

Este proyecto se centra en la **fragmentación de audio en texto y espectrogramas** para el análisis y predicción de violencia utilizando algoritmos de procesamiento de lenguaje natural (NLP). Combina modelos de aprendizaje profundo y técnicas de NLP para identificar eventos violentos en grabaciones de audio segmentadas.

---

## Objetivo General

Fragmentación de audio en texto y espectrograma para el análisis y predicción de violencia mediante algoritmos de procesamiento de lenguaje natural.

---

## Objetivos Específicos

- Descomposición de audio en texto y señales de audio.
- Identificación de set de datos para el entrenamiento y evaluación del modelo de clasificación.
- Creación, entrenamiento y evaluación del modelo.
- Análisis de texto usando embeddings.
- Identificación de sonidos contextuales.
- Identificación de falsos positivos y falsos negativos.
- Plan de contingencia y mejora del modelo para casos no identificados por el modelo.

---

## Estructura del Proyecto

- **src/**: Código fuente del proyecto.
- **models/**: Modelos pre-entrenados utilizados para la clasificación de audio.
- **data/**: Conjunto de datos de audio utilizados para entrenamiento y evaluación.
- **notebooks/**: Análisis y prototipos de desarrollo en Jupyter Notebooks.
- **results/**: Resultados obtenidos del análisis y entrenamiento.

---

## Dependencias

Este proyecto utiliza las siguientes dependencias:

- `numpy`
- `pandas`
- `keras`
- `tensorflow`
- `librosa`
- `soundfile`
- `speechrecognition`
- `streamlit`

Para instalar todas las dependencias, ejecuta el siguiente comando:

`pip install -r requirements.txt`

## Estructura del proyecto

```project/
│
├── main.py  # Página principal con navegación
├── pages/
│   ├── 1_Realizar_clasificacion_de_audio.py  # Clasificación de audio
│   ├── 2_Ver_historico.py  # Página "En construcción"
│
├── modules/
│   ├── chroma_query.py  # Funciones relacionadas con ChromaDB
│   ├── audio_prediction.py  # Funciones relacionadas con el modelo de audio
│
├── histórico/  # Carpeta donde se guardan los historiales generados
├── README.md
---
```

## Estructura del Proyecto - Módulos

La carpeta `modules/` contiene los módulos principales que soportan las funcionalidades del sistema:

- **chroma_query.py**:
  - Maneja la integración con ChromaDB.
  - Proporciona funciones para cargar la base de datos y realizar consultas utilizando embeddings generados por modelos de lenguaje.

- **audio_prediction.py**:
  - Contiene funciones para el manejo del modelo de audio.
  - Divide los audios en fragmentos, genera espectrogramas y realiza predicciones utilizando modelos preentrenados.

Estos módulos están diseñados para ser reutilizables y simplificar la lógica dentro de los archivos principales del proyecto.

---

## Licencia

**Universidad Mayor de San Simón, Cochabamba, Bolivia**

El contenido de este proyecto es parte del trabajo de tesis titulado *"Implementación de Modelo de Procesamiento de Lenguaje Natural para Predicción de Violencia mediante Segmentación de Audio Stream"*, desarrollado por **Carla Marcela Florida Roman** bajo la supervisión del **Tutor Eduardo Di Santi Grönros, PhD**.

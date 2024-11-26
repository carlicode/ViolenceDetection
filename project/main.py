import streamlit as st

st.set_page_config(
    page_title="Sistema de Clasificación de Audio",
    page_icon="🎧",
    layout="centered"
)

st.title("Sistema de Clasificación de Violencia en Audios 🎧")
st.markdown("""
## Bienvenido al Sistema de Clasificación de Audio

Este sistema permite:
- Clasificar segmentos de audio para detectar eventos relacionados con violencia.
- Generar un reporte detallado de cada fragmento analizado.
- Ver historiales de archivos previamente analizados para hacer seguimiento.

### Selecciona una opción en el menú lateral:
1. **Realizar Clasificación de Audio:**  
   Analiza audios de eventos en tiempo real. Subirás un archivo de audio para que el sistema lo procese y te entregue un análisis detallado, incluyendo la predicción de violencia y los eventos detectados.
   
2. **Ver Histórico:**  
   Consulta los historiales generados con los análisis anteriores. Aquí podrás revisar los resultados pasados y hacer comparaciones entre diferentes fragmentos de audio.

3. **Acerca de:**  
   Aquí encontrarás detalles sobre el funcionamiento del sistema, la metodología utilizada y cómo el sistema de predicción de violencia puede ser aplicado en distintos contextos de seguridad pública y monitoreo de espacios.

---

""")

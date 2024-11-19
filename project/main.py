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

### Selecciona una opción en el menú lateral:
1. **Realizar Clasificación de Audio:** Analiza audios y genera reportes.
2. **Ver Histórico:** (Próximamente) Consulta los historiales generados.

---
""")
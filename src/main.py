from src.chroma_query import load_chroma_db, query_chroma_db

# Cargar la base de datos de Chroma
db = load_chroma_db()

# Lista de textos a consultar
input_texts = ["feliz cumpleaños", "violencia", "violentar"]

# Realizar consultas
query_results = query_chroma_db(db, input_texts)

# Mostrar resultados
for result in query_results:
    print(f"Texto: {result['text']}")
    print(f"Documento asociado: {result['document']}")
    print(f"Distancia: {result['distance']:.4f}" if result['distance'] is not None else "Distancia: No disponible")
    print("-" * 50)

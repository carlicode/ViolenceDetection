import pandas as pd
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Configuración global
CHROMA_PATH = '/Users/carli.code/Desktop/tesis/violence-classification-interface/data_embeddings'
COLLECTION_NAME = 'violence_embeddings_DB'
MODEL_NAME = "all-mpnet-base-v2"

def load_chroma_db(chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME):
    """
    Carga o conecta a una colección existente en ChromaDB.
    """
    client_persistent = PersistentClient(path=chroma_path)
    try:
        # Intentar cargar la colección existente
        db = client_persistent.get_collection(name=collection_name)
        print(f"Colección '{collection_name}' cargada exitosamente.")
    except Exception as e:
        # Crear la colección si no existe
        print(f"La colección no existe, creando una nueva con métrica cosine...")
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
        db = client_persistent.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}  # Configurar distancia cosine
        )
        print(f"Colección '{collection_name}' creada exitosamente.")
    return db

def query_chroma_db(db, input_texts, model_name=MODEL_NAME, n_results=1):
    """
    Consulta una lista de textos en la colección de ChromaDB y devuelve las distancias y documentos asociados.

    Args:
        db: Objeto de la colección de ChromaDB.
        input_texts: Lista de textos a consultar.
        model_name: Nombre del modelo SentenceTransformer a utilizar.
        n_results: Número de resultados a devolver por consulta.

    Returns:
        results_list: Lista de diccionarios con 'text', 'document', y 'distance'.
    """
    # Cargar modelo de embeddings
    model = SentenceTransformer(model_name)
    
    results_list = []
    
    for text in input_texts:
        query_embedding = model.encode(text).tolist()
        results = db.query(query_embeddings=[query_embedding], n_results=n_results)
        
        if results['documents'] and results['distances']:
            for document, distance in zip(results['documents'][0], results['distances'][0]):
                results_list.append({
                    'text': text,
                    'document': document,
                    'distance': distance
                })
        else:
            results_list.append({
                'text': text,
                'document': None,
                'distance': None
            })
    
    return results_list

if __name__ == "__main__":
    # Cargar la colección
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

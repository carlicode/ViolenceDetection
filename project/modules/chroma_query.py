from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Configuración global para ChromaDB
CHROMA_PATH = '/data_embeddings'
COLLECTION_NAME = 'violence_embeddings_DB'
MODEL_NAME = "all-mpnet-base-v2"

def load_chroma_db(chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME):
    """
    Carga o conecta a una colección existente en ChromaDB.
    """
    client_persistent = PersistentClient(path=chroma_path)
    try:
        db = client_persistent.get_collection(name=collection_name)
        print(f"Colección '{collection_name}' cargada exitosamente.")
    except Exception as e:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
        db = client_persistent.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}  # Configurar distancia cosine
        )
        print(f"Colección '{collection_name}' creada exitosamente.")
    return db

def query_chroma_db(db, input_text, model_name=MODEL_NAME, n_results=1):
    """
    Realiza una consulta en la base vectorial de ChromaDB.
    """
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(input_text).tolist()
    results = db.query(query_embeddings=[query_embedding], n_results=n_results)

    if results['documents'] and results['distances']:
        document = results['documents'][0][0]
        distance = results['distances'][0][0]
        return document, distance
    return None, None
import os
import logging
import sqlite3
import pickle
import numpy as np

# 1. Configuración estricta para evitar cualquier aviso o llamada opcional al HF Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"  # Fuerza el modo completamente offline (usa caché local)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer, util


class SQLiteVectorCatalog:
    """Catálogo basado en SQLite y Vectores para persistencia local."""

    def __init__(self, db_name="vector_store.db"):
        self.db_name = db_name
        self._init_db()

    def _init_db(self):
        """Crea la tabla si no existe."""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            conn.commit()

    def insert(self, content: str, vector: np.ndarray):
        """Inserta un nuevo texto y su vector serializado en SQLite."""
        vector_blob = pickle.dumps(vector.astype(np.float32))
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO items (content, embedding) VALUES (?, ?)",
                (content, vector_blob)
            )
            conn.commit()
        return f"✅ Elemento insertado: '{content}'"

    def get_all(self):
        """Recupera todos los registros de la base de datos."""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, embedding FROM items")
            rows = cursor.fetchall()
        
        items = []
        embeddings = []
        for row in rows:
            items.append({"id": row[0], "content": row[1]})
            embeddings.append(pickle.loads(row[2]))
        
        if not embeddings:
            return [], None
            
        return items, np.array(embeddings, dtype=np.float32)

    def search_similar(self, query_vector: np.ndarray, top_k=3):
        """Busca los elementos más similares calculando la similitud de coseno."""
        items, embeddings = self.get_all()
        if not items or embeddings is None:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)

        # Cálculo de Similitud de Coseno con NumPy
        dot_product = np.dot(embeddings, q_vec)
        norm_a = np.linalg.norm(embeddings, axis=1)
        norm_b = np.linalg.norm(q_vec)
        
        # Evitar división por cero
        scores = dot_product / (norm_a * norm_b + 1e-8)

        # Ordenar de mayor a menor similitud
        best_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in best_indices:
            results.append({
                "content": items[idx]["content"],
                "score": round(float(scores[idx]), 4)
            })
        return results


# =====================================================================
# EJECUCIÓN PRINCIPAL CON SQLITE
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" 1. CARGANDO EL MODELO DE SENTENCE-TRANSFORMERS ")
    print("=" * 65)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Inicializamos nuestra base de datos SQLite persistente
    db = SQLiteVectorCatalog("vector_store.db")

    items, _ = db.get_all()
    
    # Si la base de datos está vacía, sembramos datos iniciales (máx 200 registros de prueba)
    if len(items) == 0:
        print("\n📥 Base de datos vacía. Insertando datos de prueba iniciales...")
        datos_iniciales = [
            "Curso completo de desarrollo web con JavaScript y Node.js",
            "Aprende los fundamentos de Python y ciencia de datos desde cero",
            "Receta fácil y rápida para hacer un pastel de chocolate casero",
            "Introducción a la programación orientada a objetos usando Python",
            "Guía de bases de datos relacionales con SQLite y SQL",
            "Fundamentos de inteligencia artificial y modelos de lenguaje LLM"
        ]
        for texto in datos_iniciales:
            vec = model.encode(texto)
            db.insert(texto, vec)
        print(f"✅ Se insertaron {len(datos_iniciales)} registros iniciales.")

    print("\n" + "=" * 65)
    print(" 2. REALIZANDO BÚSQUEDA SEMÁNTICA DESDE SQLITE ")
    print("=" * 65)

    consulta = "necesito aprender a programar en python"
    print(f"Búsqueda: '{consulta}'\n")

    # Generar vector de consulta
    vector_consulta = model.encode(consulta)
    
    # Buscar en SQLite
    resultados = db.search_similar(vector_consulta, top_k=3)

    print("Resultados más relevantes:")
    for i, res in enumerate(resultados, 1):
        print(f" {i}. [{res['score']}] {res['content']}")

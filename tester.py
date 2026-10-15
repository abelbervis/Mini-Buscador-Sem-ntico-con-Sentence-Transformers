import os
import logging
import numpy as np

# 1. Configuración para evitar avisos molestos de Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer, util


class MiniVectorCatalog:
    """Catálogo sencillo en memoria RAM para buscar textos por similitud semántica."""

    def __init__(self):
        self.products = []      # Lista para guardar los nombres/metadatos
        self.embeddings = None  # Matriz donde se acumulan los vectores numéricos

    def insert_product(self, product_id, name, vector):
        """Agrega un elemento y su vector asociado al catálogo."""
        self.products.append({"id": product_id, "name": name})
        
        new_vector = np.array(vector, dtype=np.float32)

        if self.embeddings is None:
            self.embeddings = np.array([new_vector])
        else:
            self.embeddings = np.vstack([self.embeddings, new_vector])

        return f"✅ Producto '{name}' indexado con éxito."

    def search_similar(self, query_vector, top_k=2):
        """Compara un vector de búsqueda con todos los vectores guardados."""
        if self.embeddings is None or len(self.products) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)

        # Cálculo de Similitud de Coseno utilizando álgebra lineal con NumPy
        dot_product = np.dot(self.embeddings, q_vec)
        norm_a = np.linalg.norm(self.embeddings, axis=1)
        norm_b = np.linalg.norm(q_vec)
        scores = dot_product / (norm_a * norm_b)

        # Ordenar de mayor a menor similitud
        best_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in best_indices:
            results.append({
                "product": self.products[idx]["name"],
                "score": round(float(scores[idx]), 4)
            })
        return results


# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================

print("=" * 65)
print(" 1. CARGANDO EL MODELO DE SENTENCE-TRANSFORMERS EN MEMORIA ")
print("=" * 65)
# Esto descarga (la 1ra vez) y carga en memoria el diccionario de conceptos vectoriales
model = SentenceTransformer("all-MiniLM-L6-v2")


print("\n" + "=" * 65)
print(" 2. EJEMPLO DE BÚSQUEDA EN CATÁLOGO VIRTUAL ")
print("=" * 65)

catalog = MiniVectorCatalog()

# Indexamos varios textos convirtiéndolos a vectores con model.encode()
catalog.insert_product(1, "Hamburguesa Doble con Queso", model.encode("Hamburguesa Doble con Queso"))
catalog.insert_product(2, "Ensalada César Fresca", model.encode("Ensalada César Fresca"))
catalog.insert_product(3, "Pizza Pepperoni Mediana", model.encode("Pizza Pepperoni Mediana"))

consulta_catalogo = "quiero algo con carne"
print(f"Búsqueda del usuario: '{consulta_catalogo}'")

# Convertimos la pregunta del usuario en vector y buscamos en el catálogo
vector_pregunta = model.encode(consulta_catalogo)
resultados = catalog.search_similar(vector_pregunta, top_k=2)

print("\nResultados encontrados:")
for res in resultados:
    print(f" -> {res['product']} (Similitud: {res['score']})")


print("\n" + "=" * 65)
print(" 3. EJEMPLO DE COMPARACIÓN DIRECTA CON UMBRAL DE COINCIDENCIA ")
print("=" * 65)

consulta = "programacion"
candidatos = [
    "Curso completo de desarrollo web con JavaScript",
    "Aprende los fundamentos de Python desde cero",
    "Receta fácil para hacer un pastel de chocolate",
    "Introducción a la programación usando Python",
]

print(f"Consulta principal: '{consulta}'\n" + "-" * 50)

# model.encode() convierte los textos en tensores numéricos para comparar todo de un jalón
vector_consulta = model.encode(consulta, convert_to_tensor=True)
vectores_candidatos = model.encode(candidatos, convert_to_tensor=True)

# util.cos_sim calcula matemáticamente qué tan cerca están los significados (0 a 1)
puntuaciones = util.cos_sim(vector_consulta, vectores_candidatos)[0]

UMBRAL_COINCIDENCIA = 0.55

for i, candidato in enumerate(candidatos):
    score = puntuaciones[i].item()
    hubo_coincidencia = score >= UMBRAL_COINCIDENCIA

    estado = "✅ SÍ HUBO COINCIDENCIA" if hubo_coincidencia else "❌ NO HUBO COINCIDENCIA"

    print(f"Texto: '{candidato}'")
    print(f"-> Resultado: {estado} (Similitud: {score:.2f})\n")

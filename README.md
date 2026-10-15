# Mini Buscador Semántico con Sentence-Transformers

Este proyecto es un ejemplo práctico y educativo en Python que demuestra cómo utilizar **`sentence-transformers`** para realizar búsqueda semántica, calcular similitud de coseno y entender los fundamentos detrás de los embeddings de texto (la tecnología que impulsa los buscadores modernos y sistemas de IA).

---

## 🚀 ¿Qué hace este proyecto?

El script principal (`tester.py`) incluye dos ejemplos clave:

1. **Catálogo Vectorial en Memoria (`MiniVectorCatalog`):** Simula una pequeña base de datos donde puedes indexar textos (como productos o artículos) convirtiéndolos en vectores numéricos, y buscar elementos similares por significado (similitud de coseno) en milisegundos.
2. **Coincidencia con Umbral de Confianza:** Compara una consulta de búsqueda contra una lista de candidatos y determina automáticamente si **sí hubo o no coincidencia** superando un límite porcentual de similitud.

---

## 🛠️ Requisitos e Instalación

Asegúrate de tener Python instalado. Luego, instala las dependencias necesarias ejecutando:

```bash
pip install sentence-transformers numpy torch
```

O si prefieres usar un archivo de requerimientos (`requirements.txt`), asegúrate de incluir:
```txt
sentence-transformers
numpy
torch
```

---

## 🏃‍♂️ Cómo Ejecutarlo

Una vez instaladas las dependencias, simplemente ejecuta el script en tu terminal:

```bash
python tester.py
```

---

## 🧠 Conceptos Clave Explicados en el Código

* **`SentenceTransformer("all-MiniLM-L6-v2")`**: Carga un modelo ligero de red neuronal preentrenado que transforma oraciones en listas de números (vectores) que capturan su significado semántico.
* **`model.encode()`**: La función principal que traduce texto humano en vectores matemáticos.
* **Similitud de Coseno (`util.cos_sim` o cálculo manual con NumPy)**: Mide la distancia geométrica entre dos vectores para saber qué tan relacionados están en significado (valores de `0.0` a `1.0`).

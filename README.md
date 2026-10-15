# Mini Buscador Semántico con Sentence-Transformers y SQLite

Este proyecto es una aplicación ligera y práctica en Python que demuestra cómo combinar **`sentence-transformers`** con una base de datos local **SQLite** para realizar búsqueda semántica con persistencia de datos. 

Está diseñada para ser una base lista para usar en agentes o proyectos pequeños (optimizada para manejar hasta cientos de registros de forma eficiente).

---

## 🚀 Características

1. **Persistencia Local con SQLite:** Los textos y sus embeddings vectoriales se guardan en un archivo local `vector_store.db`, por lo que no se pierden al cerrar el script.
2. **Serialización de Vectores (`BLOB`):** Los vectores numéricos se almacenan de forma binaria utilizando `pickle`.
3. **Búsqueda Semántica por Similitud de Coseno:** Calcula mediante álgebra lineal (`NumPy`) qué tan afín es una consulta del usuario respecto a los registros almacenados.
4. **Autosembrado:** Si la base de datos está vacía al iniciar, inserta automáticamente un conjunto de datos iniciales de prueba.

---

## 🛠️ Requisitos e Instalación

Asegúrate de tener Python instalado. Luego, instala las dependencias necesarias ejecutando:

```bash
pip install sentence-transformers numpy torch
```

---

## 🏃‍♂️ Cómo Ejecutar la Aplicación

Ejecuta el archivo principal en tu terminal:

```bash
python main.py
```

Al ejecutarlo por primera vez:
- Creará el archivo `vector_store.db`.
- Insertará los datos de prueba iniciales.
- Ejecutará una búsqueda semántica de ejemplo y mostrará los resultados con su puntuación de similitud.

Las siguientes veces que lo ejecutes, las consultas buscarán directamente sobre la base de datos ya persistida.

---

## 🧠 Estructura del Proyecto

* `main.py`: Código principal que maneja la conexión a SQLite, la indexación y la búsqueda semántica.
* `vector_store.db`: Archivo de base de datos SQLite generado localmente (ignorado por Git).
* `.gitignore`: Configurado para ignorar entornos virtuales, cachés y bases de datos locales.

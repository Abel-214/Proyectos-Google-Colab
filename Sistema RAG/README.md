## README - Sistema RAG para Documentos POA UNL

Este notebook implementa un sistema de Recuperación Aumentada por Generación (RAG) diseñado para responder preguntas basadas en documentos de Planes Operativos Anuales (POA) de la Universidad Nacional de Loja (UNL).

### 1. Preparación del Entorno

*   **Instalación de Librerías**: Se instalaron las librerías necesarias como `langchain`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-chroma`, `pypdf`, `sentence-transformers`, `chromadb`, y `faiss-cpu`.
*   **Configuración**: Se establecieron variables de entorno y se importaron los módulos clave de Langchain y HuggingFace.

### 2. Construcción de la Base Documental

*   **Carga de Documentos**: Se cargaron varios archivos PDF (`poa_2025`, `poa_2024`, `poa_2026`) utilizando `PyPDFLoader`.
*   **Fragmentación**: Los documentos se dividieron en fragmentos más pequeños (chunks) utilizando `RecursiveCharacterTextSplitter` con un tamaño de `chunk_size=800` y `chunk_overlap=100` para optimizar la recuperación de información.

### 3. Generación de Embeddings

*   **Modelo de Embeddings**: Se utilizó el modelo `sentence-transformers/all-MiniLM-L6-v2` de HuggingFace para generar representaciones vectoriales (embeddings) de los fragmentos de texto.

### 4. Construcción de la Base Vectorial

*   **ChromaDB**: Se creó una base de datos vectorial (vectorstore) utilizando ChromaDB, indexando todos los fragmentos generados con sus respectivos embeddings. La colección se nombró `poa_unl`.

### 5. Implementación del Sistema RAG

*   **Retriever**: Se configuró un `retriever` a partir de la base vectorial para buscar los 5 fragmentos más relevantes (`search_kwargs={"k":5}`) en función de la pregunta del usuario.
*   **Modelo de Lenguaje (LLM)**: Se cargó el modelo `Qwen/Qwen2.5-0.5B-Instruct` de HuggingFace para la generación de respuestas.
*   **Prompt Template**: Se definió una plantilla de prompt estricta para guiar al LLM a responder únicamente con la información proporcionada en el contexto recuperado.
*   **Función `preguntar_rag`**: Esta función orquesta el proceso completo: recupera documentos relevantes, construye el prompt con el contexto y la pregunta, y genera la respuesta utilizando el LLM.

### 6. Pruebas del Sistema

Se realizaron varias pruebas con diferentes preguntas para evaluar la capacidad del sistema RAG para extraer y sintetizar información de los documentos.

### 7. Análisis de Resultados

*   **Calidad de las respuestas**: En general, las respuestas fueron coherentes, pero se observaron casos donde el modelo interpretó el contexto de forma parcial o añadió información no explícita, como en la quinta consulta sobre la alineación del POA, donde se omitió el PEDI.
*   **Pertinencia de los documentos recuperados**: Los fragmentos recuperados fueron en su mayoría relevantes, aunque en ocasiones se incluyeron documentos de diferentes años (POA 2024, 2025, 2026), lo que introdujo información adicional no siempre directamente pertinente a la consulta específica.
*   **Posibles errores de recuperación**: La recuperación de fragmentos de diferentes versiones de POA con información similar pero no idéntica pudo llevar a respuestas con datos inconsistentes, enfatizando la importancia de la calidad y especificidad de los documentos indexados.

**Conclusión**: El sistema RAG demuestra ser efectivo para la consulta de documentos POA, pero la precisión de las respuestas está fuertemente ligada a la calidad del contexto recuperado y a la capacidad del LLM para adherirse estrictamente a las reglas del prompt.

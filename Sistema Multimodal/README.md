## README: Análisis de Modelos Visuales con CLIP

Este notebook presenta una investigación y experimentación con el modelo CLIP (Contrastive Language-Image Pre-training) de OpenAI, basado en el paper de Radford et al. (2021).

### Contenido:

1.  **Investigación Bibliográfica:** Resumen del paper "Learning Transferable Visual Models From Natural Language Supervision", destacando su aporte clave y relevancia científica.

2.  **Implementación del Modelo:** Carga y configuración del modelo `CLIPModel` y `CLIPProcessor` de Hugging Face `transformers` para su uso en inferencia.

3.  **Carga y Visualización de Datos:** Descarga y visualización de imágenes de ejemplo que serán utilizadas en los experimentos.

4.  **Función de Inferencia:** Implementación de una función `ejecutar_inferencia` para procesar imágenes y una lista de textos, calculando las probabilidades de coincidencia y el tiempo de inferencia.

5.  **Experimentación:** Serie de experimentos diseñados para evaluar el comportamiento de CLIP bajo diversas condiciones:
    *   **Especificidad del texto:** Variación de la descripción textual para una misma imagen.
    *   **Distractores:** Evaluación de la capacidad de discriminación del modelo con textos ambiguos o irrelevantes.
    *   **Imágenes distintas:** Prueba con diferentes escenarios visuales.
    *   **Sensibilidad al idioma:** Comparación del rendimiento con textos en español e inglés.
    *   **Número de distractores:** Análisis del impacto de un número creciente de opciones en la predicción.
    *   **Robustez:** Evaluación del modelo ante transformaciones de imagen (ruido, rotación, escala de grises).

6.  **Evaluación:** Implementación de métricas de recuperación (Top-1 y Top-5 Accuracy) utilizando la similitud coseno de los embeddings de imagen y texto para cuantificar el rendimiento del modelo en una tarea de recuperación imagen-texto.

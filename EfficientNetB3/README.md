# Proyecto de Clasificación de Razas de Perros con CNN

## Descripción
Este proyecto implementa una red neuronal convolucional (CNN) utilizando la arquitectura EfficientNetB3 pre-entrenada para clasificar cinco razas específicas de perros del dataset Stanford Dogs. El modelo está diseñado para identificar y categorizar imágenes de Chihuahuas, Japanese Spaniels, Malteses, Pekineses y Shih-Tzus.

## Estructura del Proyecto
El notebook está dividido en las siguientes secciones:
1.  **Importación de Librerías**: Carga las bibliotecas necesarias como TensorFlow, Keras y Matplotlib.
2.  **Carga del Dataset**: Descarga y prepara el dataset Stanford Dogs, filtrando las 5 razas seleccionadas y realizando un preprocesamiento inicial.
3.  **Exploración de Datos**: Visualiza ejemplos de imágenes y la distribución de las clases en el dataset.
4.  **Preprocesamiento**: Normaliza las imágenes para el entrenamiento del modelo.
5.  **Construcción de la CNN**: Define la arquitectura del modelo utilizando EfficientNetB3 como base, añadiendo capas de regularización y una capa de salida para la clasificación.
6.  **Visualización de Arquitectura**: Muestra un resumen de las capas del modelo.
7.  **Compilación del Modelo**: Configura el optimizador, la función de pérdida y las métricas para el entrenamiento.
8.  **Entrenamiento del Modelo**: Entrena la CNN utilizando los datos preparados, con callbacks para Early Stopping y reducción del Learning Rate.
9.  **Evaluación de Desempeño**: Calcula la pérdida y la precisión del modelo en el conjunto de prueba.
10. **Matriz de Confusión**: Genera una matriz de confusión para visualizar el rendimiento de clasificación por clase.
11. **Curva de Aprendizaje**: Grafica la precisión de entrenamiento y validación a lo largo de las épocas.
12. **Predicción de Imágenes**: Demuestra el funcionamiento del modelo realizando predicciones sobre un subconjunto de imágenes de prueba.
13. **Predicciones con Imágenes Subidas**: Permite al usuario subir sus propias imágenes para obtener predicciones en tiempo real.

## Cómo Ejecutar el Notebook
1.  **Entorno**: Asegúrate de tener un entorno Python con las siguientes librerías instaladas:
    - `tensorflow`
    - `tensorflow_datasets`
    - `matplotlib`
    - `numpy`
    - `scikit-learn`
    - `Pillow`
    - `requests`

    Puedes instalar las dependencias con `pip`:
    ```bash
    pip install tensorflow tensorflow-datasets matplotlib numpy scikit-learn Pillow requests
    ```
2.  **Ejecución**: Simplemente ejecuta las celdas del notebook en orden, desde el principio hasta el final.

## Resultados
El modelo entrenado alcanzó una precisión de validación del 96.34%. La matriz de confusión y las curvas de aprendizaje proporcionan una visión detallada del rendimiento del modelo y su capacidad para generalizar a nuevas imágenes.

## Limitaciones y Mejoras Futuras
-   **Cantidad de Datos**: El dataset original para estas 5 razas es limitado, lo que puede afectar la robustez del modelo.
-   **Número de Clases**: Actualmente el modelo solo clasifica 5 razas. Expandir a más razas requeriría más datos de entrenamiento.
-   **Overfitting**: Aunque se utilizan técnicas como Dropout y Data Augmentation, con más datos y un entrenamiento más prolongado se podría mejorar la generalización.

---

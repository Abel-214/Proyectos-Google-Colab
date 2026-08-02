# LeNet-5 para la Clasificación de Fashion MNIST

Este notebook demuestra la implementación y el entrenamiento de una red neuronal convolucional (CNN) LeNet-5 para la clasificación de imágenes en el conjunto de datos Fashion MNIST.

## Resumen del Proyecto

El objetivo de este proyecto es clasificar imágenes de prendas de vestir del conjunto de datos Fashion MNIST utilizando una arquitectura LeNet-5 personalizada. El notebook cubre la carga de datos, el preprocesamiento, la definición del modelo, el entrenamiento, la evaluación y la visualización de los resultados.

## Conjunto de Datos

Se utiliza el conjunto de datos [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist), que consta de 70.000 imágenes en escala de grises de 10 categorías de moda. Cada imagen tiene 28x28 píxeles. Para este proyecto, las imágenes se redimensionan a 32x32 píxeles para que coincidan con los requisitos de entrada de la arquitectura LeNet-5.

## Arquitectura del Modelo

El modelo LeNet-5 implementado aquí incluye:
- Dos capas convolucionales (C1 y C3) con activación ReLU y normalización por lotes (Batch Normalization).
- Dos capas de pooling promedio (S2 y S4).
- Una capa convolucional final (C5).
- Una capa Flatten para convertir los mapas de características 3D en vectores 1D.
- Una capa oculta completamente conectada (F6) con activación ReLU.
- Una capa de salida softmax para 10 clases.

## Configuración y Uso

Para ejecutar este notebook:

1.  **Dependencias**: Asegúrate de tener TensorFlow, Keras, NumPy, Matplotlib y Scikit-learn instalados. Puedes instalarlos usando pip:
    ```bash
    pip install tensorflow numpy matplotlib scikit-learn
    ```
2.  **Ejecutar Celdas**: Ejecuta todas las celdas en orden secuencial. El notebook:
    - Cargará y preprocesará el conjunto de datos Fashion MNIST.
    - Construirá y compilará el modelo LeNet-5.
    - Entrenará el modelo con los datos de entrenamiento.
    - Evaluará el rendimiento del modelo con los datos de prueba.
    - Visualizará el historial de entrenamiento y una matriz de confusión.
    - Mostrará una predicción de ejemplo.

## Resultados

Después de entrenar durante 10 épocas, el modelo logró una precisión de aproximadamente **87.54%** en el conjunto de prueba. Las curvas de aprendizaje y la matriz de confusión proporcionan información adicional sobre el rendimiento del modelo y las áreas de mejora.
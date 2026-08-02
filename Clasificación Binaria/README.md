## Implementación y Aplicaciones del Perceptrón

Este notebook demuestra la implementación y aplicación de modelos perceptrón para tareas de clasificación binaria. Cubre dos escenarios principales:

1.  **Predicción de Aprobación de Tarjetas de Crédito**: Se construye un modelo perceptrón básico desde cero para determinar si una persona debe ser aprobada para una tarjeta de crédito basándose en sus puntuaciones de crédito y ahorro.
2.  **Sistema de Recomendación de Películas**: Se implementa un perceptrón con una función de activación tangente hiperbólica (tanh) para clasificar películas como 'recomendadas' o 'no recomendadas' según su calificación y número de reseñas.

### Estructura del Notebook:

*   **Librerías**: Importa las librerías necesarias como NumPy y Matplotlib.
*   **Caso de Aprobación de Tarjetas de Crédito**:
    *   **Generación de Datos**: Define un conjunto de datos de individuos con puntuaciones de crédito y ahorro, junto con su estado de aprobación.
    *   **Visualización**: Grafica los datos para separar visualmente los casos aprobados y denegados.
    *   **Perceptrón desde Cero**: Implementa un modelo perceptrón simple, incluyendo una función de activación y un bucle de entrenamiento para aprender pesos y sesgo.
    *   **Visualización del Límite de Decisión**: Visualiza el límite de decisión aprendido por el perceptrón personalizado.
    *   **Perceptrón de Scikit-learn**: Demuestra cómo lograr lo mismo con `Perceptron` de `sklearn.linear_model`.
*   **Sistema de Recomendación de Películas (Activación Tanh)**:
    *   **Generación de Datos**: Define datos de películas con calificaciones y número de reseñas, junto con una clasificación de 'recomendado' o 'no recomendado'.
    *   **Función de Activación Tanh**: Implementa un perceptrón con una función de activación tangente hiperbólica y su derivada para el entrenamiento.
    *   **Bucle de Entrenamiento**: Entrena el modelo perceptrón con activación tanh.
    *   **Ejemplos de Predicción**: Muestra predicciones para nuevos datos de películas.
    *   **Visualización del Límite de Decisión**: Visualiza el límite de decisión y las regiones de clasificación para el sistema de recomendación de películas utilizando un mapa de calor.
    *   **Perceptrón de Scikit-learn (Tanh)**: Demuestra el uso de `Perceptron` de `sklearn.linear_model` para el caso tanh.

### Cómo Ejecutar:

1.  Abre el notebook en Google Colab o cualquier entorno Jupyter.
2.  Ejecuta todas las celdas secuencialmente para observar el entrenamiento del perceptrón, las visualizaciones y las predicciones para ambos casos.

### Dependencias:

*   `numpy`
*   `matplotlib`
*   `sklearn`

Todas las dependencias pueden instalarse usando `pip`:

```bash
pip install numpy matplotlib scikit-learn
```
```
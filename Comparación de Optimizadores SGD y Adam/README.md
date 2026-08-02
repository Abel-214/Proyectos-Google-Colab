## Optimización en Redes Neuronales y Regularización (Digits Dataset)

Este proyecto explora la optimización y regularización en redes neuronales utilizando el conjunto de datos Digits de scikit-learn. El objetivo es comparar el rendimiento de diferentes optimizadores (SGD y Adam) y evaluar el impacto de la técnica de regularización Dropout en la precisión y la capacidad de generalización del modelo.

## 1. Dataset: Digits

El conjunto de datos Digits consiste en 1797 imágenes de dígitos manuscritos (0-9), cada una de 8x8 píxeles. Las imágenes se representan como vectores de 64 características numéricas. Es un dataset comúnmente utilizado para problemas de clasificación de imágenes.

### Carga y Preparación de Datos

- Los datos se cargan usando `load_digits()` de `sklearn.datasets`.
- Se realiza una división del dataset en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%) utilizando `train_test_split` con estratificación.
- Las características se escalan utilizando `StandardScaler` para normalizar los datos.

**Shapes de los datos después de la preparación:**
- `X_train`: (1257, 64)
- `X_val`: (270, 64)
- `X_test`: (270, 64)

## 2. Implementación del Modelo MLP Base

Se define una red neuronal Perceptrón Multicapa (MLP) con la siguiente arquitectura:

- Una capa de entrada densa con 64 unidades y activación ReLU.
- Una capa de Dropout (opcional).
- Una capa oculta densa con 32 unidades y activación ReLU.
- Otra capa de Dropout (opcional).
- Una capa de salida densa con 10 unidades (para las 10 clases de dígitos) y activación Softmax.

El modelo se compila utilizando `sparse_categorical_crossentropy` como función de pérdida y `accuracy` como métrica.

## 3. Aplicación de Optimizadores

Se entrenan modelos MLP con diferentes configuraciones de optimizadores y Dropout durante 30 épocas y un tamaño de lote de 32.

### Optimizador SGD
- Configuración: `SGD(learning_rate=0.01, momentum=0.9)`
- El modelo SGD mostró una convergencia progresiva y estable, alcanzando alta precisión en entrenamiento, pero la precisión de validación se estancó después de la época 17, indicando posible sobreajuste.

### Optimizador Adam
- Configuración: `Adam(learning_rate=0.01)`
- El optimizador Adam mostró una convergencia significativamente más rápida que SGD, alcanzando altas precisiones en menos épocas. También mostró signos de sobreajuste, con la precisión de entrenamiento llegando al 100% mientras la de validación se estabilizaba.

## 4. Aplicación de Regularización (Dropout)

Se evalúa el impacto de Dropout con una tasa del 0.3, utilizando el optimizador Adam o SGD.

### Adam + Dropout
- Configuración: `Adam(learning_rate=0.001)` con `dropout_rate=0.3`.
- La incorporación de Dropout ralentizó la velocidad de aprendizaje, pero mantuvo las curvas de entrenamiento y validación más cercanas, sugiriendo una mejor generalización y reducción del sobreajuste.

### SGD + Dropout
- Configuración: `SGD(learning_rate=0.01, momentum=0.9)` con `dropout_rate=0.3`.
- Similar al caso de Adam, Dropout con SGD también contribuyó a mantener un equilibrio entre las precisiones de entrenamiento y validación, mitigando el sobreajuste.

## 5. Evaluación de Resultados

Se visualiza el historial de `accuracy` y `loss` para los diferentes modelos y se evalúa su rendimiento en el conjunto de prueba.

**Resultados en el conjunto de prueba:**
- **SGD**: Loss = 0.0507 | Accuracy = 0.9815
- **Adam**: Loss = 0.0489 | Accuracy = 0.9815
- **Adam + Dropout**: Loss = 0.0678 | Accuracy = 0.9815
- **SGD + Dropout**: Loss = 0.0363 | Accuracy = 0.9889

## 6. Conclusiones

1.  **Optimizador con mejor desempeño**: El optimizador Adam presentó el mejor desempeño general, logrando una convergencia más rápida y altos niveles de precisión en menos épocas. En conjunto con Dropout, la combinación de SGD+Dropout fue la que mejor resultado obtuvo en el conjunto de test (98.89%).

2.  **Diferencias entre SGD y Adam**: Adam demostró una velocidad de convergencia superior debido a sus tasas de aprendizaje adaptativas. SGD, por otro lado, mostró un aprendizaje más gradual.

3.  **Afectación de Dropout al rendimiento**: Dropout redujo ligeramente la precisión de entrenamiento y ralentizó la convergencia, pero mejoró la capacidad de generalización al evitar el sobreajuste, manteniendo las curvas de entrenamiento y validación más cercanas.

4.  **Evidencia de Overfitting**: El sobreajuste fue evidente en modelos sin Dropout, donde la precisión de entrenamiento alcanzaba el 100% mientras que la de validación se estancaba o incluso aumentaba ligeramente la pérdida, indicando que el modelo memorizaba los datos de entrenamiento.

5.  **Configuración más adecuada**: La configuración de **SGD + Dropout** se considera la más adecuada, ya que logró el mayor `Accuracy` en el conjunto de prueba (0.9889), presentando un excelente equilibrio entre rendimiento y capacidad de generalización.

## 7. Preguntas de Control

-   **¿Qué es un optimizador en Deep Learning?**
    Un optimizador es un algoritmo que ajusta los pesos de una red neuronal para minimizar la función de pérdida y mejorar la precisión del modelo.

-   **¿Cuál es la diferencia entre SGD y Adam?**
    SGD actualiza los pesos de forma gradual con una tasa de aprendizaje fija, lo que a menudo requiere más épocas para converger. Adam utiliza tasas de aprendizaje adaptativas para cada parámetro, logrando una convergencia más rápida y eficiente.

-   **¿Qué problema busca resolver Dropout?**
    Dropout busca reducir el overfitting desactivando aleatoriamente neuronas durante el entrenamiento, evitando que la red dependa excesivamente de características específicas y mejorando su capacidad de generalización.

-   **¿Qué es overfitting?**
    El overfitting (sobreajuste) ocurre cuando un modelo aprende demasiado bien los datos de entrenamiento, incluyendo ruido, lo que resulta en un pobre rendimiento con datos nuevos o no vistos.

-   **¿Por qué se requiere validación?**
    La validación permite evaluar el desempeño del modelo en datos no vistos durante el entrenamiento, ayudando a detectar sobreajuste y a comparar configuraciones de modelo de manera objetiva.

-   **¿Qué optimizador obtuvo mejores resultados?**
    En esta práctica, la combinación **SGD + Dropout** obtuvo los mejores resultados en el conjunto de prueba, alcanzando una precisión del 98.89%.
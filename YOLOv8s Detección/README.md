# Detección de Razas de Perros (Pug, Boxer, Beagle) con YOLOv8s

## Resumen de la Implementación

Este proyecto implementa un modelo de detección de objetos basado en la arquitectura **YOLOv8s** para identificar y clasificar tres razas específicas de perros: Pug, Boxer y Beagle. Se utilizó aprendizaje por transferencia, preentrenando el modelo con pesos existentes y optimizándolo con el algoritmo AdamW. El desempeño del modelo fue evaluado utilizando métricas estándar como precisión, recall y mAP (mean Average Precision).

## Dataset Utilizado

Se empleó una versión modificada del **Oxford-IIIT Pet Dataset**. Se seleccionaron exclusivamente imágenes de las razas Pug, Boxer y Beagle. Las anotaciones originales se adaptaron al formato YOLO, generando *bounding boxes* para cada objeto detectado. El conjunto de datos fue dividido en subconjuntos de entrenamiento y validación para asegurar una evaluación robusta de la capacidad de generalización del modelo.

## Entrenamiento del Modelo

El modelo YOLOv8s fue entrenado con los siguientes parámetros:

- **Epochs**: 100
- **Tamaño de imagen (imgsz)**: 640
- **Batch size**: 16
- **Optimizador**: AdamW
- **Learning Rate (lr0)**: 0.001
- **Weight Decay**: 0.0005
- **Patience**: 20 (para detención temprana)
- **Preentrenamiento**: True (utilizando pesos 'yolov8s.pt')

## Resultados

El modelo entrenado alcanzó los siguientes resultados sobre el conjunto de validación:

- **Precisión (Precision)**: 91.37%
- **Recall**: 96.42%
- **mAP@50**: 98.06%
- **mAP@50-95**: 83.46%

### Matriz de Confusión Normalizada

Se generó una matriz de confusión para visualizar el rendimiento del modelo en la clasificación de cada raza:

|           | True Pug | True Boxer | True Beagle |
| :-------- | :------- | :--------- | :---------- |
| Predicted Pug   | 1.00     | 0.00       | 0.00        |
| Predicted Boxer | 0.00     | 0.95       | 0.00        |
| Predicted Beagle| 0.00     | 0.00       | 0.95        |

*(Nota: Los valores son representativos y basados en el heatmap generado.)*

## Limitaciones

A pesar de los buenos resultados, es importante considerar las siguientes limitaciones:

*   **Razas Específicas**: El modelo fue entrenado exclusivamente con tres razas, lo que significa que no puede reconocer correctamente otras razas de perros que no fueron parte del conjunto de entrenamiento.
*   **Clasificación de Desconocidos**: Frente a imágenes de razas no conocidas, el modelo tiende a clasificarlas incorrectamente como Pug, Boxer o Beagle, basándose en similitudes visuales.
*   **Tamaño del Dataset**: El tamaño reducido del conjunto de datos podría limitar la capacidad de generalización del modelo en condiciones de iluminación, poses o escenarios significativamente diferentes a los vistos durante el entrenamiento.
*   **Fuente de Imágenes**: La evaluación se realizó con imágenes similares a las del dataset original. El rendimiento podría variar al aplicar el modelo a imágenes de fuentes externas o entornos reales.
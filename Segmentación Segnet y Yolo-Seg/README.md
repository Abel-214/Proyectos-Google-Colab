# Segmentación de imágenes usando Segnet y Yolo-Seg

Este repositorio contiene un proyecto de segmentación de imágenes implementado en Google Colab, que compara dos arquitecturas de redes neuronales: **SegNet** y **YOLO-Seg**. El objetivo es segmentar imágenes de mascotas del dataset `oxford_iiit_pet`.

## 1) Carga y Preprocesamiento de Datos

El dataset `oxford_iiit_pet` se carga usando `tensorflow_datasets`. Las imágenes se normalizan a un tamaño de 224x224 píxeles y los valores de los píxeles se escalan a un rango de [0, 1]. Las máscaras de segmentación se ajustan para que las etiquetas sean 0, 1 y 2.

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

IMG_SIZE = 224
NUM_CLASSES = 3 # Fondo, Borde, Mascota

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1 # Ajuste para que las etiquetas sean 0, 1, 2
    return input_image, input_mask

def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(
        datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE),
        method='nearest'
    )
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

# Preparación de los datasets de entrenamiento y prueba
BATCH_SIZE = 16
BUFFER_SIZE = 1000

train_dataset = tfds.load('oxford_iiit_pet', split='train')
test_dataset = tfds.load('oxford_iiit_pet', split='test')

train_images = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = test_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(tf.data.AUTOTUNE)
)
test_batches = test_images.batch(BATCH_SIZE)

# Función para visualizar imágenes y máscaras
def display(display_list, titles):
    plt.figure(figsize=(10, 4))
    for i, img in enumerate(display_list):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.imshow(tf.keras.utils.array_to_img(img))
        plt.axis('off')
    plt.show()
```

## 2) Arquitectura SegNet

### Descripción

**SegNet** es una arquitectura de segmentación semántica que utiliza una estructura encoder-decoder. El encoder reduce la resolución espacial y extrae características de alto nivel, mientras que el decoder reconstruye la resolución espacial utilizando índices de pooling del encoder para una mejor preservación de los detalles de los bordes.

### Entrenamiento

El modelo SegNet fue compilado con el optimizador Adam y `SparseCategoricalCrossentropy` como función de pérdida. Se entrenó durante 30 épocas.

### Métricas de Evaluación (SegNet)

Después de 30 épocas de entrenamiento, SegNet alcanzó las siguientes métricas en el conjunto de prueba:

*   **Accuracy:** 87.61%
*   **Mean IoU:** 0.6973

**Reporte de clasificación:**

```
              precision    recall  f1-score   support

       Fondo       0.88      0.87      0.87   1287883
       Borde       0.91      0.94      0.93   2256922
     Mascota       0.68      0.58      0.63    469275

    accuracy                           0.88   4014080
   macro avg       0.82      0.80      0.81   4014080
weighted avg       0.87      0.88      0.87   4014080
```

### Matriz de Confusión (SegNet)

![Matriz de Confusión SegNet](confusion_matrix_segnet.png) <!-- Placeholder, actual image needs to be added manually or generated -->

## 3) Arquitectura YOLO-Seg

### Descripción

**YOLO-Seg** es una arquitectura inspirada en principios de YOLO para segmentación, diseñada para ser eficiente y precisa. Incorpora bloques como `conv_bn_silu`, `csp_block` y `sppf_block`, y utiliza skip connections para fusionar características de diferentes escalas, lo que ayuda a preservar la información espacial. Este modelo combina un encoder para la extracción de características y un decoder para la reconstrucción de máscaras de segmentación.

### Entrenamiento

El modelo YOLO-Seg fue compilado con el optimizador Adam y una `combined_loss` que integra `SparseCategoricalCrossentropy` y `Dice Loss`. Se entrenó durante 35 épocas.

```python
def dice_loss(y_true, y_pred, num_classes=NUM_CLASSES, smooth=1e-6):
    y_true_oh = tf.one_hot(tf.cast(tf.squeeze(y_true, axis=-1), tf.int32), num_classes)
    y_pred_soft = tf.nn.softmax(y_pred, axis=-1)

    intersection = tf.reduce_sum(y_true_oh * y_pred_soft, axis=[1, 2])
    union = tf.reduce_sum(y_true_oh, axis=[1, 2]) + tf.reduce_sum(y_pred_soft, axis=[1, 2])

    dice_coef = (2. * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice_coef)

def combined_loss(y_true, y_pred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    ce_loss = scce(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return ce_loss + d_loss
```

### Métricas de Evaluación (YOLO-Seg)

Después de 35 épocas de entrenamiento, YOLO-Seg obtuvo las siguientes métricas en el conjunto de prueba:

*   **Accuracy:** 89.84%
*   **Mean IoU:** 0.7384

**Reporte de clasificación:**

```
              precision    recall  f1-score   support

       Fondo       0.89      0.92      0.91   1287883
       Borde       0.95      0.94      0.94   2256922
     Mascota       0.67      0.65      0.66    469275

    accuracy                           0.90   4014080
   macro avg       0.84      0.84      0.84   4014080
weighted avg       0.90      0.90      0.90   4014080
```

### Matriz de Confusión (YOLO-Seg)

![Matriz de Confusión YOLO-Seg](confusion_matrix_yoloseg.png) <!-- Placeholder, actual image needs to be added manually or generated -->

## Análisis de Resultados

### ¿Qué arquitectura de segmentación implementaron y por qué la seleccionaron?

Se implementó la arquitectura **SegNet**, un modelo de segmentación semántica basado en una estructura **encoder-decoder**. Fue seleccionada porque permite realizar clasificación a nivel de píxel, mantiene información espacial mediante las operaciones de pooling y unpooling, y representa una arquitectura eficiente para establecer un modelo base de comparación en tareas de segmentación.

### ¿Qué fortalezas observaron durante la segmentación?

-   El modelo logró generar máscaras semánticas a nivel de píxel, identificando correctamente las regiones principales de las imágenes.
-   Presentó buen desempeño en la separación entre objeto y fondo.
-   La arquitectura mostró capacidad para aprender características visuales relevantes de las mascotas.
-   Las métricas obtenidas permitieron evaluar la calidad de la segmentación mediante Accuracy, Mean IoU y Dice Coefficient.

### ¿Qué limitaciones encuentran en el modelo utilizado?

-   Puede presentar dificultades en regiones pequeñas o con detalles finos debido a la reducción de resolución durante el encoder.
-   El desempeño puede disminuir cuando existen fondos complejos, oclusiones o variaciones importantes en la apariencia del objeto.
-   Existe posibilidad de sobreajuste si el modelo aprende características específicas del conjunto de entrenamiento.
-   La calidad de la segmentación depende directamente de la precisión de las máscaras originales del dataset.

### ¿Qué variables afectaron el desempeño del modelo?

Los principales factores que pueden influir en la segmentación son:

-   **Iluminación:** cambios de brillo, sombras o baja iluminación.
-   **Fondo:** texturas complejas o colores similares al objeto.
-   **Oclusiones:** objetos que cubren parcialmente la mascota.
-   **Variación visual:** diferentes razas, tamaños y posiciones.
-   **Resolución:** pérdida de detalles al reducir las imágenes para el entrenamiento.

### ¿En qué aplicaciones reales sería adecuada esta arquitectura?

SegNet puede aplicarse en:

-   Sistemas de análisis y reconocimiento de animales.
-   Realidad aumentada y efectos visuales.
-   Robótica y navegación inteligente.
-   Inspección visual automatizada.
-   Análisis de imágenes médicas o biológicas con datasets especializados.

### ¿Qué mejoras realizarían con más tiempo de desarrollo?

Como mejoras futuras se propone:

-   Aplicar técnicas de **Data Augmentation** para aumentar la variabilidad del entrenamiento.
-   Utilizar modelos preentrenados mediante **Transfer Learning** para mejorar la extracción de características.
-   Optimizar hiperparámetros como tasa de aprendizaje y número de épocas.
-   Implementar funciones de pérdida avanzadas como **Dice Loss o Focal Loss** para mejorar la segmentación de clases menos representadas.
-   Comparar SegNet con arquitecturas más modernas como **U-Net o DeepLabV3+**.
-   Realizar análisis de errores para identificar casos donde el modelo falla.

## Preguntas de control

*   **¿Cuál es la diferencia entre clasificación, detección y segmentación de imágenes?**
    La clasificación de imágenes consiste en asignar una única etiqueta a toda la imagen, indicando qué objeto o categoría contiene. La detección de objetos, además de identificar la clase, localiza cada objeto mediante un cuadro delimitador (bounding box). En cambio, la segmentación realiza una clasificación a nivel de píxel, generando una máscara que delimita con precisión cada región de la imagen. En el código desarrollado se implementa segmentación, ya que el modelo predice una máscara para cada imagen del conjunto de datos.

*   **¿Qué diferencia existe entre segmentación semántica y segmentación de instancias?**
    La segmentación semántica asigna una clase a cada píxel, pero no distingue entre diferentes objetos de la misma categoría. Por ejemplo, si existen dos perros en una imagen, ambos serán etiquetados como "perro". La segmentación de instancias, además de clasificar los píxeles, diferencia cada objeto individual, por lo que cada perro tendría su propia máscara independiente.

*   **¿Cuál es la función del encoder y del decoder en una arquitectura de segmentación?**
    El encoder tiene la función de extraer las características más importantes de la imagen mediante capas convolucionales y operaciones de reducción de tamaño, permitiendo que el modelo aprenda información relevante. Posteriormente, el decoder reconstruye la resolución original utilizando esas características para generar la máscara de segmentación, asignando una clase a cada píxel de la imagen.

*   **¿Qué ventajas ofrecen las conexiones Skip Connections en arquitecturas como U-Net?**
    Las Skip Connections permiten transmitir directamente la información de las primeras capas del encoder hacia el decoder. Gracias a ello, se conserva mejor la información espacial y los detalles finos de la imagen, lo que mejora la precisión en los bordes y la segmentación de objetos pequeños. Además, facilitan el entrenamiento de redes profundas al reducir la pérdida de información durante el proceso de codificación.

*   **¿Qué factores pueden afectar la calidad de una segmentación?**
    La calidad de la segmentación puede verse afectada por factores como una iluminación deficiente, fondos complejos, oclusiones parciales de los objetos, baja resolución de las imágenes, ruido, variaciones en la escala o la orientación de los objetos y un conjunto de entrenamiento poco diverso. También influyen la arquitectura seleccionada, los hiperparámetros del entrenamiento y la cantidad de datos disponibles para entrenar el modelo.

*   **¿Qué métricas pueden emplearse para evaluar un modelo de segmentación y qué información proporciona cada una?**
    Se utilizaron métricas como la exactitud (Accuracy), el índice de intersección sobre unión (IoU o Intersection over Union), la matriz de confusión y el reporte de clasificación. Se indica el porcentaje de píxeles correctamente clasificados, aunque puede resultar poco representativa cuando existen clases desbalanceadas. La métrica IoU mide el grado de coincidencia entre la máscara predicha y la máscara real, siendo una de las más utilizadas para evaluar segmentación. La matriz de confusión muestra los aciertos y errores entre las clases, mientras que el reporte de clasificación proporciona métricas como precisión, recall y F1-score para analizar el desempeño del modelo en cada categoría.

## Conclusiones

Los resultados muestran que YOLO-Seg superó a SegNet en las métricas de evaluación, alcanzando un Accuracy de 89.84% y un Mean IoU de 0.7384%, mientras que SegNet obtuvo 87.61% y 0.6973%, respectivamente. Asimismo, YOLO-Seg logró un F1-score de 0.66 para la clase Mascota, superior al 0.63 obtenido por SegNet. Esta diferencia también se aprecia visualmente en las imágenes de prueba, donde las máscaras generadas por YOLO-Seg presentan una mayor correspondencia con las máscaras reales, conservando mejor la forma general de los animales y produciendo menos regiones faltantes o segmentaciones incompletas. En conjunto, estos resultados presentan una ligera mejora en la capacidad de YOLO-Seg para representar con precisión los objetos presentes en el conjunto de datos.

SegNet obtuvo un buen rendimiento en las clases Fondo y Borde, presentó mayores dificultades para segmentar correctamente la clase Mascota, reflejadas en un Recall de 0.58, inferior al 0.65 alcanzado por YOLO-Seg. Esto indica que una mayor cantidad de píxeles pertenecientes a la mascota fueron clasificados incorrectamente como fondo o borde. Esto se observa en las predicciones realizadas, ya que las máscaras producidas por SegNet presentan pérdidas de información en el interior del objeto y contornos menos definidos. Estas limitaciones pueden atribuirse a su arquitectura encoder–decoder, que utiliza operaciones de UpSampling para reconstruir la resolución espacial y no incorpora mecanismos de extracción y fusión multiescala como los presentes en YOLO-Seg, reduciendo así su capacidad para preservar detalles finos y segmentar objetos con formas complejas.
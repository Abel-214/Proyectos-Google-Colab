# Reconocimiento de Comandos de Voz con Redes Convolucionales

Este notebook implementa un sistema de reconocimiento de comandos de voz utilizando redes neuronales convolucionales (CNNs) en TensorFlow/Keras. El objetivo es clasificar audios cortos de comandos simples como 'up', 'down', 'left', 'right', 'go', 'stop', 'yes', y 'no'.

## 1. Importación de Librerías y Carga de Datos

Se importan las librerías necesarias de `tensorflow`, `keras`, `numpy`, `matplotlib`, `sklearn`, y `seaborn`. El dataset utilizado es `mini_speech_commands`, que se descarga automáticamente si no está presente. Este dataset contiene 8000 archivos de audio de 1 segundo, distribuidos equitativamente entre las 8 clases de comandos.

## 2. Preprocesamiento de Datos

Los audios se preprocesan para convertirlos en espectrogramas, que son representaciones visuales de la frecuencia a lo largo del tiempo. Los pasos clave incluyen:

- **Decodificación de audio**: Los archivos `.wav` se decodifican y se les aplica `tf.squeeze` para eliminar dimensiones innecesarias.
- **Obtención de etiquetas**: Las etiquetas se extraen del nombre de la carpeta que contiene cada archivo de audio.
- **Normalización de longitud**: Se asegura que todos los audios tengan una duración de 1 segundo (16000 muestras), rellenando con ceros o truncando si es necesario.
- **Cálculo de espectrogramas**: Se aplica la Transformada de Fourier de Tiempo Corto (STFT) para obtener los espectrogramas, los cuales se expanden con una dimensión de canal para ser compatibles con las capas `Conv2D`.
- **División de datos**: El dataset se divide en conjuntos de entrenamiento (80%), validación (10%) y prueba (10%).
- **Batching y prefetching**: Los datasets se configuran para ser procesados en batches y se utiliza `AUTOTUNE` para optimizar el rendimiento de carga de datos.

## 3. Construcción del Modelo CNN

Se construye un modelo de red neuronal convolucional secuencial (`tf.keras.models.Sequential`) con las siguientes características:

- **Capa de entrada**: `layers.Input` con la forma del espectrograma (124, 129, 1).
- **Redimensionamiento**: Se redimensionan los espectrogramas a (64, 64) para una entrada consistente.
- **Normalización**: `layers.Normalization` adaptada al dataset de entrenamiento para normalizar los datos de entrada.
- **Capas convolucionales**: Bloques de `Conv2D`, `BatchNormalization`, y `MaxPooling2D` para extraer características jerárquicas.
- **Regularización**: `layers.Dropout` se utiliza después de las capas convolucionales y densas para prevenir el sobreajuste.
- **Aplanamiento**: `layers.Flatten` para convertir la salida 2D de las CNNs en un vector 1D.
- **Capas densas**: Dos `layers.Dense`, la última con activación `softmax` para la clasificación multiclase (8 comandos).

El modelo se compila con el optimizador `Adam` (learning rate de 1e-3), función de pérdida `SparseCategoricalCrossentropy` y `accuracy` como métrica.

## 4. Entrenamiento del Modelo

El modelo se entrena durante 30 épocas utilizando `model.fit`. Se implementan `callbacks` para un entrenamiento más robusto:

- `EarlyStopping`: Detiene el entrenamiento si la `val_loss` no mejora durante 5 épocas, restaurando los mejores pesos.
- `ReduceLROnPlateau`: Reduce el learning rate si la `val_loss` no mejora durante 3 épocas.

El entrenamiento muestra una `val_accuracy` alcanzando aproximadamente el 91.25% y una `val_loss` de 0.3016, indicando un buen rendimiento en el conjunto de validación.

## 5. Evaluación del Modelo

El modelo se evalúa en el conjunto de prueba, obteniendo una **precisión del 90.75%**.

Se genera un **reporte de clasificación** detallado (`classification_report`) que muestra la precisión, recall y f1-score para cada una de las 8 clases, así como el promedio macro y ponderado.

Además, se visualiza una **matriz de confusión** (`confusion_matrix`) para entender las predicciones correctas e incorrectas por clase.

## 6. Prueba con Nuevos Comandos (Audios Externos)

Se proporciona una funcionalidad para subir archivos de audio externos (en cualquier formato compatible con `ffmpeg`, como `.ogg`, `.mp3`, `.wav`, `.m4a`) y realizar predicciones en tiempo real.

La función `predecir_audio` realiza los siguientes pasos:

1.  **Conversión a WAV 16kHz**: Utiliza `ffmpeg` para estandarizar el audio a un formato mono de 16kHz, compatible con el modelo entrenado.
2.  **Preprocesamiento**: Decodifica el audio, obtiene el espectrograma, y lo expande para que tenga una dimensión de batch.
3.  **Predicción**: El modelo predice la clase del comando y la confianza asociada.
4.  **Visualización**: Muestra la señal de audio original y un gráfico de barras con las probabilidades de cada clase predicha, destacando la clase con mayor confianza.

Los resultados de las pruebas con audios cargados manualmente (ejemplos: `Pausa down.ogg`, `Left.ogg`, `yes.ogg`, `stop.ogg`, `LEFT pronunciado.ogg`, `YES pronunciado.ogg`, `LEFT bien pronunciado.mp4`) demuestran que el modelo puede predecir con alta confianza para algunas clases, aunque puede haber errores de clasificación en otros casos, especialmente cuando la confianza es más baja.
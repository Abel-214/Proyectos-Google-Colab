# Detección de Enfermedades Cardíacas con Modelos de Machine Learning

## Descripción del Proyecto
Este proyecto tiene como objetivo predecir la presencia de enfermedades cardíacas en pacientes utilizando un conjunto de datos médicos. Se exploran y comparan diferentes modelos de machine learning, incluyendo Perceptrón, Red Neuronal Multicapa (MLP) y Red Neuronal Profunda (DNN), para identificar el enfoque más efectivo en esta tarea de clasificación binaria.

## Conjunto de Datos
El conjunto de datos utilizado, `heart.csv`, contiene información de pacientes con diversas características médicas. Las columnas clave incluyen:
- `age`: Edad del paciente.
- `sex`: Sexo (1 = hombre, 0 = mujer).
- `cp`: Tipo de dolor en el pecho.
- `trestbps`: Presión arterial en reposo.
- `chol`: Colesterol sérico.
- `fbs`: Glucosa en sangre en ayunas (> 120 mg/dl).
- `restecg`: Resultados del electrocardiograma en reposo.
- `thalach`: Frecuencia cardíaca máxima alcanzada.
- `exang`: Angina inducida por el ejercicio.
- `oldpeak`: Depresión del ST inducida por el ejercicio en relación con el reposo.
- `slope`: La pendiente del segmento ST máximo del ejercicio.
- `ca`: Número de vasos principales coloreados por fluoroscopia (0-3).
- `thal`: Tipo de talasemia (1 = normal; 2 = defecto fijo; 3 = defecto reversible).
- `target`: Variable objetivo (0 = ausencia de enfermedad, 1 = presencia de enfermedad).

## Preprocesamiento de Datos
1.  **Carga y Limpieza**: El dataset `heart.csv` se carga y se verifica la ausencia de valores nulos.
2.  **Conversión a XLSX**: Se guarda una copia limpia del dataset en formato `heart_clean.xlsx`.
3.  **Análisis de Nulos**: Se confirma que no hay valores nulos en el dataset limpio.
4.  **Codificación One-Hot**: Las características categóricas (`cp`, `restecg`, `slope`, `thal`) se transforman utilizando codificación One-Hot para convertirlas en un formato adecuado para los modelos de machine learning.
5.  **División de Datos**: El dataset se divide en características (X) y la variable objetivo (y).
6.  **Normalización**: Las características (X) se escalan utilizando `StandardScaler` para asegurar que todas las características contribuyan de manera equitativa al modelo.
7.  **Separación de Conjuntos**: Los datos se dividen en conjuntos de entrenamiento (80%) y prueba (20%) utilizando `train_test_split` con estratificación para mantener la proporción de clases en la variable objetivo.

## Modelos Implementados

### 1. Perceptrón
Un modelo lineal simple que clasifica entradas en una de dos categorías. Se entrena con los datos escalados y se evalúa su rendimiento.

### 2. Red Neuronal Multicapa (MLP)
Una red neuronal con múltiples capas ocultas. Este modelo utiliza capas `Dense` con activación `relu`, `Dropout` para regularización y `sigmoid` en la capa de salida para clasificación binaria. Se compila con el optimizador 'adam' y la función de pérdida 'binary_crossentropy'. Se utiliza `EarlyStopping` para evitar el sobreajuste.

### 3. Red Neuronal Profunda (DNN)
Una arquitectura de red neuronal más profunda que la MLP, con más capas ocultas y `Dropout` entre ellas para mejorar la capacidad de aprendizaje y la generalización. Similar al MLP, utiliza 'adam' y 'binary_crossentropy', con `EarlyStopping`.

## Resultados y Evaluación
Se evalúa el rendimiento de cada modelo utilizando métricas clave como Accuracy, Precision, Recall y F1-Score, así como matrices de confusión.

### Perceptrón
-   **Accuracy**: 0.7902
-   **Precision**: 0.7422
-   **Recall**: 0.9048
-   **F1-Score**: 0.8155

### MLP
-   **Accuracy**: 0.8976
-   **Precision**: 0.8818
-   **Recall**: 0.9238
-   **F1-Score**: 0.9023

### DNN
-   **Accuracy**: 0.9902
-   **Precision**: 1.0000
-   **Recall**: 0.9810
-   **F1-Score**: 0.9904

Los gráficos de pérdida y precisión durante el entrenamiento de MLP y DNN también se proporcionan para visualizar el rendimiento del modelo a lo largo de las épocas.

## Conclusión
El modelo DNN demostró el mejor rendimiento en este conjunto de datos, logrando una precisión y F1-Score significativamente más altos en comparación con el Perceptrón y el MLP.
# README: Comparación de Modelos de Clasificación (Iris Dataset)

## Introducción
Este notebook presenta una comparación entre dos modelos populares de clasificación: la Regresión Logística y el Árbol de Decisión. El objetivo es clasificar las especies de la flor Iris utilizando sus características morfológicas, evaluando el rendimiento de cada modelo y destacando sus principales diferencias y características.

## Dataset
Se utiliza el dataset estándar de Iris, que contiene 150 muestras de flores de Iris, cada una con cuatro características medidas (longitud y ancho del sépalo, longitud y ancho del pétalo) y una de tres posibles especies objetivo: setosa, versicolor o virginica.

### Preparación de Datos
- El dataset se carga utilizando `sklearn.datasets.load_iris` y se convierte en un `pandas.DataFrame`.
- Se dividen los datos en conjuntos de entrenamiento y prueba (80/20) con estratificación para asegurar una representación equitativa de las clases.
- Las características para el modelo de Regresión Logística se escalan utilizando `StandardScaler`.

## Modelos Implementados

### 1. Regresión Logística
- **Descripción:** Un modelo lineal que predice la probabilidad de una instancia de pertenecer a una clase específica. Ideal para clasificaciones binarias y multiclase (con estrategias como One-vs-Rest o multinomial).
- **Configuración:** `LogisticRegression` con `max_iter=200`, `random_state=42`, y `multi_class='multinomial'`.
- **Resultados:**
  - Accuracy: 0.9333 (93.33%)
  - El reporte de clasificación y la matriz de confusión muestran un buen rendimiento general, con algunas confusiones entre `versicolor` y `virginica`.
- **Funcionamiento:** Aprende fronteras de decisión lineales, calculando probabilidades de pertenencia a cada clase mediante la función sigmoide y asignando la clase con mayor probabilidad.

### 2. Árbol de Decisión
- **Descripción:** Un modelo no paramétrico que divide el espacio de características en regiones rectangulares para clasificar las instancias. Es altamente interpretable.
- **Configuración:** `DecisionTreeClassifier` con `max_depth=4`, `random_state=42`, y `criterion='gini'`.
- **Resultados:**
  - Accuracy: 0.9333 (93.33%)
  - Similar a la Regresión Logística, presenta un buen rendimiento, con confusiones en las mismas clases.
  - La visualización del árbol muestra las reglas de decisión en cada nodo.
- **Funcionamiento:** Divide el espacio de características con umbrales en cada nodo, maximizando la pureza de los nodos hijos. La clasificación se realiza siguiendo un camino de reglas hasta un nodo hoja.
- **Importancia de Características:** `petal length (cm)` y `petal width (cm)` son las características más influyentes para este modelo.

## Comparación y Conclusiones

| Modelo                  | Accuracy | Errores | Frontera             | Escalado      | Interpretabilidad          |
|:------------------------|:---------|:--------|:---------------------|:--------------|:---------------------------|
| Regresión Logística     | 0.9333   | 1       | Lineal               | Requerido     | Media (coeficientes)       |
| Árbol de Decisión       | 0.9333   | 1       | No lineal (umbrales) | No requerido  | Alta (reglas visibles)     |

Ambos modelos lograron el mismo *accuracy* en este dataset particular. Sin embargo, sus mecanismos subyacentes y requisitos son diferentes:

- **Regresión Logística** requiere escalado de características y aprende fronteras de decisión lineales. Su interpretabilidad se basa en los coeficientes.
- **Árbol de Decisión** no requiere escalado de características y puede aprender fronteras de decisión no lineales, siendo altamente interpretable a través de sus reglas visuales.

La elección entre estos modelos dependerá de la naturaleza del problema, la interpretabilidad deseada y la complejidad de las relaciones en los datos.
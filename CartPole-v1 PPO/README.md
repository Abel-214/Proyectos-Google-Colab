## ¿Qué es Reinforcement Learning?
El Reinforcement Learning (RL) o Aprendizaje por Refuerzo es una rama de la inteligencia artificial donde un agente aprende a tomar decisiones interactuando con un entorno. No se le dice directamente cuál es la acción correcta, sino que aprende mediante prueba y error para obtener mejores resultados con el tiempo, maximizando la suma de recompensas.

## Configuración del Entorno y Librerías
Este proyecto utiliza las siguientes librerías y un entorno de Gym para simular el problema del CartPole:

```python
!pip install stable-baselines3[extra] -q
!pip install gymnasium -q

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
```

### Creación del Entorno (CartPole-v1)
Se utiliza el entorno `CartPole-v1` de Gymnasium, donde el agente debe mantener un péndulo invertido balanceándose sobre un carro. El entorno se envuelve en `Monitor` para registrar métricas de entrenamiento.

**Espacio de Estados (Observación):**
`Box([-4.8 -inf -0.41887903 -inf], [4.8 inf 0.41887903 inf], (4,), float32)`
Contiene 4 valores: posición del carro, velocidad del carro, ángulo del palo, velocidad angular del palo.

**Espacio de Acciones:**
`Discrete(2)`
Indica 2 posibles acciones: mover el carro a la izquierda o a la derecha.

## Configuración y Entrenamiento del Modelo PPO
Se utiliza el algoritmo Proximal Policy Optimization (PPO) de `stable-baselines3`.

### Parámetros del Modelo PPO:
```python
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    gamma=0.99,
    n_steps=1024,
    batch_size=64,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1
)
```

### Entrenamiento:
El modelo se entrena por un total de `10000` `timesteps`.
```python
model.learn(
    total_timesteps=10000
)

print("\nEntrenamiento finalizado")
```

## Visualización del Aprendizaje
Se genera un gráfico que muestra la evolución de las recompensas durante el entrenamiento para entender cómo el agente mejora con el tiempo.

![Evolución de recompensas durante el entrenamiento PPO](ppo_rewards_plot.png) <!-- Placeholder, as direct image display is not possible here. The notebook cell generates the plot directly. -->

## Evaluación del Agente
El modelo entrenado se evalúa durante 20 episodios para obtener una medida de su rendimiento promedio.

**Resultados de Evaluación:**
- `Reward promedio: 409.95`
- `Desviación estándar: 91.67`

## Prueba del Agente
Se ejecuta el agente entrenado en un nuevo entorno para observar su comportamiento sin renderización visual (para evitar errores en entornos sin GUI).

**Resultados de la Prueba:**
- `Pasos ejecutados: 306`
- `Reward obtenida: 306.0`

## Animación PPO en Colab
Para una visualización del agente en acción, se genera un GIF a partir de la renderización del entorno.

![Animación PPO CartPole](ppo_cartpole.gif) <!-- Placeholder, as direct image display is not possible here. The notebook cell generates the GIF. -->

**Resultados de la Animación:**
- `Reward obtenida: 434.0`

## Experimentos con Hiperparámetros
Se realizaron experimentos modificando el `learning_rate` (LR) y `gamma` (G) para observar su impacto en la recompensa promedio.

### Configuraciones Probadas:
1.  `LR=0.001`, `G=0.95`
2.  `LR=0.0003`, `G=0.99` (Configuración inicial)
3.  `LR=0.0001`, `G=0.999`

### Resultados de los Experimentos:
-   `Learning Rate: 0.001`, `Gamma: 0.95`, `Reward: 266.6`
-   `Learning Rate: 0.0003`, `Gamma: 0.99`, `Reward: 341.6`
-   `Learning Rate: 0.0001`, `Gamma: 0.999`, `Reward: 173.5`

### Visualización de la Comparación:
Se genera un gráfico de barras para comparar visualmente las recompensas promedio obtenidas con cada configuración de hiperparámetros.

![Comparación PPO](ppo_comparison_plot.png) <!-- Placeholder, as direct image display is not possible here. The notebook cell generates the plot directly. -->

## Preguntas y Respuestas

**- ¿Qué diferencia existe entre RL y aprendizaje supervisado?**
La principal diferencia entre el aprendizaje por refuerzo (RL) y el aprendizaje supervisado es la forma en que aprenden los modelos. En el aprendizaje supervisado, el modelo entrena con datos etiquetados donde ya se conoce la respuesta correcta, mientras que en RL el agente aprende interactuando con un entorno y tomando decisiones para obtener mejores resultados con el tiempo. En RL no se le dice directamente cuál es la acción correcta, sino que aprende mediante prueba y error.

**- ¿Qué es una recompensa?**
Una recompensa es un valor numérico que el entorno entrega al agente después de realizar una acción. Esta recompensa indica si la acción fue buena o mala para alcanzar el objetivo. El agente intenta maximizar la suma de recompensas obtenidas, por lo que las utiliza como guía para aprender qué decisiones son más convenientes.

**- ¿Qué representa un estado?**
Un estado representa la situación actual del entorno en un momento específico. Contiene la información necesaria para que el agente pueda decidir qué acción tomar. Por ejemplo, en un videojuego, el estado podría incluir la posición del personaje, enemigos y objetos disponibles en pantalla.

**- ¿Qué función cumple la política?**
La política es la estrategia que utiliza el agente para decidir qué acción realizar en cada estado. En otras palabras, define el comportamiento del agente dentro del entorno. La política puede ser simple, como elegir acciones al azar, o compleja, utilizando redes neuronales para tomar decisiones más inteligentes.

**- ¿Por qué el agente necesita explorar?**
El agente necesita explorar porque al inicio no conoce cuáles acciones producen mejores resultados. Si solo repitiera las acciones conocidas, podría perder oportunidades de encontrar estrategias más efectivas. La exploración permite descubrir nuevas acciones y mejorar el aprendizaje, aunque algunas decisiones iniciales puedan ser incorrectas.

**- ¿Qué parámetros afectan el entrenamiento?**
Los parámetros que afectan el entrenamiento incluyen la tasa de aprendizaje, el número de episodios, el factor de descuento y el nivel de exploración. Estos parámetros influyen en la velocidad con la que aprende el agente, cuánto valora las recompensas futuras y el equilibrio entre explorar nuevas acciones o aprovechar las ya conocidas.
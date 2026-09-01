# MiniGPT desde cero con PyTorch

Manual autocontenido, ejercicios y solucionario para construir un modelo de lenguaje pequeño desde los fundamentos de Machine Learning hasta un Transformer causal entrenable.

```text
texto → tokens → embeddings → Transformer → logits → probabilidades → siguiente token
```

El objetivo no es competir con modelos comerciales. El objetivo es entender qué sucede internamente y poder explicar y programar cada componente sin utilizar modelos preentrenados ni Hugging Face.

> Este README ya contiene toda la teoría, todos los ejercicios, las soluciones y el código final. No funciona como diario de progreso y no necesita ser modificado al completar cada ejercicio. La recomendación es intentar cada enunciado antes de abrir su solución.

## Índice

1. [Metodología](#metodología)
2. [Mapa completo](#mapa-completo)
3. [Fase 0 — Fundamentos de Machine Learning](#fase-0--fundamentos-de-machine-learning)
4. [Fase 1 — De escalares a vectores y redes](#fase-1--de-escalares-a-vectores-y-redes)
5. [Fase 2 — Fundamentos de PyTorch](#fase-2--fundamentos-de-pytorch)
6. [Fase 3 — Texto, tokens y batches](#fase-3--texto-tokens-y-batches)
7. [Fase 4 — Primer modelo de lenguaje: bigrama](#fase-4--primer-modelo-de-lenguaje-bigrama)
8. [Fase 5 — Embeddings y posición](#fase-5--embeddings-y-posición)
9. [Fase 6 — Self-attention causal](#fase-6--self-attention-causal)
10. [Fase 7 — Transformer y MiniGPT](#fase-7--transformer-y-minigpt)
11. [Fase 8 — Entrenamiento, evaluación e inferencia](#fase-8--entrenamiento-evaluación-e-inferencia)
12. [Solución final completa](#solución-final-completa-minigptpy)
13. [Pruebas y errores frecuentes](#pruebas-y-errores-frecuentes)
14. [Glosario](#glosario)

## Metodología

Para cada concepto nuevo:

1. Entender el problema que resuelve.
2. Resolver un ejemplo mínimo a mano.
3. Implementarlo sin abstracciones innecesarias.
4. Comprobar el resultado con casos conocidos.
5. Experimentar cambiando una sola variable cada vez.
6. Revisar el código y la comprensión conceptual.

La regla de trabajo es:

```text
una operación a mano → implementación propia → prueba automática → experimento
```

No merece la pena calcular 25 epochs a mano. Sí merece la pena calcular una actualización para saber qué debería hacer el programa.

## Mapa completo

| Fase | Resultado |
|---|---|
| 0 | Entrenar una regresión con fuerza bruta y gradient descent |
| 1 | Entender vectores, matrices, capas, activaciones y redes |
| 2 | Entrenar modelos con tensores, autograd y `nn.Module` |
| 3 | Convertir texto en ejemplos numéricos para un modelo |
| 4 | Entrenar un modelo de lenguaje bigrama |
| 5 | Representar significado y posición mediante embeddings |
| 6 | Implementar una cabeza de self-attention causal |
| 7 | Construir bloques Transformer y el MiniGPT |
| 8 | Entrenar, validar, guardar y generar texto |

# Fase 0 — Fundamentos de Machine Learning

## 0.1 Qué es Machine Learning

En programación tradicional, el programador escribe las reglas:

```text
entrada + reglas programadas → salida
```

En Machine Learning se define una estructura y se ajustan sus parámetros usando ejemplos:

```text
entradas + respuestas correctas → entrenamiento → parámetros aprendidos
```

El modelo no reescribe el programa. El entrenamiento modifica valores numéricos internos para reducir el error.

## 0.2 Vocabulario fundamental

| Concepto | Significado |
|---|---|
| **Feature** | Dato de entrada utilizado para predecir. |
| **Label o target** | Respuesta correcta asociada a una entrada. |
| **Dataset** | Colección de ejemplos. |
| **Modelo** | Función o estructura que transforma entradas en predicciones. |
| **Predicción** | Salida estimada, normalmente escrita como `ŷ`. |
| **Parámetro** | Valor interno que aprende el modelo, como `w`. |
| **Hiperparámetro** | Valor elegido por el ingeniero, como el learning rate. |
| **Loss** | Número que mide el error del modelo. |
| **Gradiente** | Indica cómo cambia la loss al modificar los parámetros. |
| **Entrenamiento** | Proceso de actualizar parámetros para reducir la loss. |
| **Inferencia** | Uso del modelo entrenado sin modificar sus parámetros. |

Un modelo de regresión produce una cantidad continua. No tiene por qué producir una probabilidad.

## 0.3 Primer modelo: regresión con un parámetro

El modelo es:

$$
\hat y = wx
$$

- `x`: entrada.
- `w`: parámetro entrenable.
- `ŷ`: predicción.
- `y`: target real.

La estructura `ŷ = wx` no cambia durante el entrenamiento. Lo que cambia es `w`.

```python
dataset = [
    (5, 15),
    (7, 21),
    (10, 30),
]
```

En estos datos ideales, `w = 3` reproduce todos los ejemplos.

## 0.4 Función de pérdida

El error cuadrático de un ejemplo es:

$$
(\hat y-y)^2
$$

El cuadrado evita que errores positivos y negativos se cancelen y penaliza más los errores grandes.

El error cuadrático medio o MSE es:

$$
L(w)=\frac{1}{N}\sum_{i=1}^{N}(wx_i-y_i)^2
$$

- MSE alta: malas predicciones.
- MSE baja: predicciones próximas a los targets.
- MSE cero en entrenamiento: reproduce esos datos, pero no demuestra que generalice.

## 0.5 Búsqueda por fuerza bruta

La primera estrategia consiste en probar muchos valores de `w`, calcular su MSE y conservar el mejor. Reducir el salto mejora la precisión, pero aumenta el número de evaluaciones. Esto no escala a millones de parámetros.

Con el ejemplo ruidoso `(8, 25)`, ningún `w` acierta todos los puntos. El óptimo continuo es:

$$
w^*=\frac{\sum x_i y_i}{\sum x_i^2}=3.033613445\ldots
$$

### Ejercicio 0.1

Implementar `predict`, `squared_error`, `mse` y `find_best_w`. Comparar saltos `0.1`, `0.01` y `0.001` en el intervalo `[0, 5]`.

<details>
<summary>Solución 0.1</summary>

```python
def predict(x, w):
    return w * x


def squared_error(prediction, target):
    return (prediction - target) ** 2


def mse(dataset, w):
    squared_errors = []
    for x, target in dataset:
        prediction = predict(x, w)
        squared_errors.append(squared_error(prediction, target))
    return sum(squared_errors) / len(squared_errors)


def find_best_w(dataset, step_size, minimum=0.0, maximum=5.0):
    min_mse = float("inf")
    best_w = None
    number_of_steps = int((maximum - minimum) / step_size) + 1

    for step in range(number_of_steps):
        w = minimum + step * step_size
        current_mse = mse(dataset, w)
        if current_mse < min_mse:
            min_mse = current_mse
            best_w = w

    return best_w, min_mse


dataset = [(5, 15), (7, 21), (10, 30), (8, 25)]

for step_size in [0.1, 0.01, 0.001]:
    best_w, min_loss = find_best_w(dataset, step_size)
    print(step_size, best_w, min_loss)
```

Resultados aproximados:

| Salto | Mejor `w` | MSE |
|---:|---:|---:|
| `0.1` | `3.0` | `0.25` |
| `0.01` | `3.03` | `0.183` |
| `0.001` | `3.034` | `0.1828` |

</details>

## 0.6 Gradiente y gradient descent

Para la MSE del modelo `ŷ = wx`:

$$
\frac{\partial L}{\partial w}
=
\frac{1}{N}\sum_{i=1}^{N}2x_i(wx_i-y_i)
$$

- Gradiente positivo: al aumentar `w`, aumenta la loss; reducimos `w`.
- Gradiente negativo: al aumentar `w`, baja la loss; aumentamos `w`.
- Gradiente próximo a cero: estamos cerca de un punto estacionario.

Gradient descent actualiza en dirección contraria al gradiente:

$$
w_{nuevo}=w_{actual}-\eta\frac{\partial L}{\partial w}
$$

`η` es el learning rate:

- Muy pequeño: aprendizaje estable pero lento.
- Adecuado: convergencia rápida.
- Muy grande: oscilación o divergencia.

Es comparable a la ganancia de un controlador: una ganancia excesiva puede volver inestable el sistema.

### Ejemplo manual resuelto

Para `x = 2`, `y = 6`, `w = 1` y `learning_rate = 0.1`:

| Operación | Resultado |
|---|---:|
| Predicción `wx` | `2` |
| Error `ŷ-y` | `-4` |
| Loss | `16` |
| Gradiente `2x(ŷ-y)` | `-16` |
| Nuevo `w` | `2.6` |
| Nueva predicción | `5.2` |
| Nueva loss | `0.64` |

La loss baja de `16` a `0.64`.

## 0.7 Epoch, batch e iteración

- **Epoch:** una pasada completa por el dataset de entrenamiento.
- **Batch:** subconjunto de ejemplos procesados juntos.
- **Iteración o step:** una actualización de los parámetros.

Si el dataset tiene 100 ejemplos y el batch size es 20:

```text
5 batches por epoch → 5 actualizaciones por epoch
```

En full-batch gradient descent, todo el dataset forma un único batch, por lo que hay una actualización por epoch.

`initial_w` es la configuración inicial. Dentro de `train`, `w` es el estado actual y cambia:

```text
initial_w = 0.0
w: 0.0 → 1.805 → 2.536025 → ...
```

### Ejercicio 0.2

Implementar el gradiente y entrenar durante 25 epochs con learning rates `0.0001`, `0.005` y `0.02`.

<details>
<summary>Solución 0.2</summary>

```python
def mse_gradient(dataset, w):
    gradients = []
    for x, target in dataset:
        prediction = predict(x, w)
        gradients.append(2 * x * (prediction - target))
    return sum(gradients) / len(gradients)


def train(dataset, initial_w, learning_rate, epochs):
    w = initial_w

    for epoch in range(epochs):
        current_loss = mse(dataset, w)
        current_gradient = mse_gradient(dataset, w)

        print(
            f"Epoch = {epoch}, "
            f"w = {w:.6f}, "
            f"loss = {current_loss:.6f}, "
            f"gradient = {current_gradient:.6f}"
        )

        w = w - learning_rate * current_gradient

    return w


for learning_rate in [0.0001, 0.005, 0.02]:
    final_w = train(dataset, 0.0, learning_rate, 25)
    print(
        f"lr={learning_rate}, "
        f"final_w={final_w:.6f}, "
        f"final_loss={mse(dataset, final_w):.6f}"
    )
```

Resultados tras 25 actualizaciones:

| Learning rate | Comportamiento | `w` final | Loss final |
|---:|---|---:|---:|
| `0.0001` | Estable, demasiado lento | `0.784648` | `301.124502` |
| `0.005` | Converge | `3.033613` | `0.182773` |
| `0.02` | Oscila y diverge | `9529.621987` | `5,399,975,219.279538` |

Primeras líneas para `0.005`:

```text
Epoch = 0, w = 0.000000, loss = 547.750000, gradient = -361.000000
Epoch = 1, w = 1.805000, loss = 89.997487, gradient = -146.205000
Epoch = 2, w = 2.536025, loss = 14.914632, gradient = -59.213025
```

Errores corregidos durante el ejercicio:

- `range(epochs + 1)` ejecuta 26 iteraciones si `epochs = 25`; debe ser `range(epochs)`.
- La lista correcta es `[0.0001, 0.005, 0.02]`, no repetir `0.0001`.
- `train` debe devolver `w`.
- La loss, el gradiente y el `w` impresos deben pertenecer al mismo estado, antes de actualizar.

</details>

## 0.8 Mini-batches y barajado

Procesar todo el dataset puede ser caro. Los mini-batches permiten actualizar más frecuentemente y utilizar memoria limitada. Barajar evita que el orden fijo introduzca patrones indeseados.

- Full batch: gradiente exacto del dataset, caro.
- Mini-batch: estimación eficiente y algo ruidosa.
- Stochastic gradient descent estricto: batch size igual a 1.

### Ejercicio 0.3

1. Crear batches de tamaño 2.
2. Barajar los datos una vez por epoch.
3. Actualizar `w` después de cada batch.
4. Contar el número total de updates.

<details>
<summary>Solución 0.3</summary>

```python
import random


def make_batches(dataset, batch_size, rng):
    shuffled = list(dataset)
    rng.shuffle(shuffled)
    return [
        shuffled[start:start + batch_size]
        for start in range(0, len(shuffled), batch_size)
    ]


def train_mini_batch(
    dataset,
    initial_w,
    learning_rate,
    epochs,
    batch_size,
    seed=42,
):
    w = initial_w
    updates = 0
    rng = random.Random(seed)

    for epoch in range(epochs):
        batches = make_batches(dataset, batch_size, rng)

        for batch in batches:
            gradient = mse_gradient(batch, w)
            w = w - learning_rate * gradient
            updates += 1

        print(
            f"epoch={epoch}, w={w:.6f}, "
            f"dataset_loss={mse(dataset, w):.6f}"
        )

    return w, updates


final_w, updates = train_mini_batch(
    dataset=dataset,
    initial_w=0.0,
    learning_rate=0.005,
    epochs=25,
    batch_size=2,
)

assert updates == 50  # 2 batches × 25 epochs
print(final_w, mse(dataset, final_w), updates)
```

Con solo cuatro datos, el resultado exacto depende del orden de los mini-batches. Lo importante es que la loss tienda a disminuir y se produzcan 50 actualizaciones.

</details>

## 0.9 Train, validation y test

- **Train:** ajusta los parámetros.
- **Validation:** elige arquitectura e hiperparámetros.
- **Test:** estima el rendimiento final una sola vez.

No se debe entrenar con validation o test. Si tomamos muchas decisiones mirando test, lo convertimos indirectamente en validation.

### Ejercicio 0.4

Dividir un dataset de forma reproducible en train, validation y test, sin perder ni duplicar ejemplos.

<details>
<summary>Solución 0.4</summary>

```python
import random


def split_dataset(dataset, train_ratio=0.7, validation_ratio=0.15, seed=42):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio debe estar entre 0 y 1")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio debe estar entre 0 y 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Debe quedar espacio para test")

    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    validation_end = train_end + int(len(shuffled) * validation_ratio)

    train_data = shuffled[:train_end]
    validation_data = shuffled[train_end:validation_end]
    test_data = shuffled[validation_end:]

    assert len(train_data) + len(validation_data) + len(test_data) == len(dataset)
    return train_data, validation_data, test_data
```

En un dataset real se debe vigilar además el orden temporal, los grupos relacionados y la distribución de clases para evitar data leakage.

</details>

## 0.10 Underfitting, overfitting y generalización

- **Underfitting:** el modelo es demasiado simple o está poco entrenado; falla en train y validation.
- **Overfitting:** memoriza peculiaridades de train; train mejora mientras validation empeora.
- **Generalización:** funciona bien con datos nuevos de la misma distribución.
- **Data leakage:** información que no debería estar disponible entra en el entrenamiento.

| Train loss | Validation loss | Diagnóstico probable |
|---:|---:|---|
| Alta | Alta | Underfitting |
| Muy baja | Mucho más alta | Overfitting o cambio de distribución |
| Baja | Baja y próxima | Buena generalización |

Medidas habituales contra overfitting: más datos, modelo más pequeño, regularización, dropout, data augmentation y early stopping.

# Fase 1 — De escalares a vectores y redes

## 1.1 Escalar, vector, matriz y tensor

- Escalar: un número, forma `()`.
- Vector: lista unidimensional, forma `(n,)`.
- Matriz: tabla bidimensional, forma `(filas, columnas)`.
- Tensor: generalización a cualquier número de dimensiones.

En una capa lineal:

$$
Y=XW+b
$$

Si `X` contiene `B` ejemplos con `D_in` features y la capa produce `D_out` features:

| Tensor | Shape |
|---|---|
| `X` | `(B, D_in)` |
| `W` | `(D_in, D_out)` |
| `b` | `(D_out,)` |
| `Y` | `(B, D_out)` |

Las dimensiones interiores de la multiplicación deben coincidir.

## 1.2 Neurona y capa lineal

Una neurona combina entradas mediante pesos y bias:

$$
z=w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

El bias permite desplazar la función. Sin bias, una regresión lineal está obligada a pasar por el origen.

## 1.3 Funciones de activación

Apilar capas lineales sin activación sigue siendo equivalente a una única transformación lineal. Una activación introduce no linealidad.

- ReLU: `max(0, x)`.
- GELU: activación suave usada habitualmente en Transformers.
- Sigmoid: comprime a `(0, 1)`, útil en algunas salidas binarias.
- Tanh: comprime a `(-1, 1)`.

## 1.4 Forward, backward y backpropagation

- **Forward pass:** calcula predicciones y loss.
- **Backward pass:** aplica la regla de la cadena para obtener gradientes.
- **Optimizer step:** actualiza parámetros.
- **Zero grad:** limpia gradientes anteriores.

Backpropagation no es el optimizador. Backprop calcula gradientes; SGD o AdamW utilizan esos gradientes para actualizar los parámetros.

### Ejercicio 1.1

Calcular la salida de una capa con dos entradas y dos neuronas.

```text
x = [2, 3]
W = [[1, -1],
     [2,  4]]
b = [0.5, -0.5]
```

<details>
<summary>Solución 1.1</summary>

Interpretando las columnas de `W` como neuronas:

$$
y_1=2(1)+3(2)+0.5=8.5
$$

$$
y_2=2(-1)+3(4)-0.5=9.5
$$

```python
import numpy as np

x = np.array([2.0, 3.0])
W = np.array([[1.0, -1.0], [2.0, 4.0]])
b = np.array([0.5, -0.5])
y = x @ W + b

assert np.allclose(y, np.array([8.5, 9.5]))
```

</details>

### Ejercicio 1.2

Implementar una red mínima `2 → 3 → 1` con NumPy y ReLU, solo para el forward pass.

<details>
<summary>Solución 1.2</summary>

```python
import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def forward_mlp(x, W1, b1, W2, b2):
    hidden_pre_activation = x @ W1 + b1
    hidden = relu(hidden_pre_activation)
    output = hidden @ W2 + b2
    return output


x = np.array([[1.0, 2.0], [3.0, 4.0]])
W1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
b1 = np.zeros(3)
W2 = np.array([[0.7], [0.8], [0.9]])
b2 = np.zeros(1)

output = forward_mlp(x, W1, b1, W2, b2)
assert output.shape == (2, 1)
```

</details>

# Fase 2 — Fundamentos de PyTorch

## 2.1 Instalación y ejecución en VS Code

Desde la terminal del proyecto:

```bash
python -m venv .venv
```

Activación en PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activación en Linux o macOS:

```bash
source .venv/bin/activate
```

Instalación:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy
```

En VS Code, seleccionar el intérprete de `.venv` y ejecutar:

```bash
python minigpt.py
```

## 2.2 Tensores, dtype y device

Un tensor contiene datos numéricos y metadatos:

- `shape`: dimensiones.
- `dtype`: tipo numérico.
- `device`: CPU, CUDA o MPS.
- `requires_grad`: si autograd debe seguir las operaciones.

Los índices de tokens deben ser normalmente `torch.long`; los pesos y activaciones suelen ser `float32`.

## 2.3 Autograd

PyTorch construye dinámicamente un grafo de operaciones. `loss.backward()` recorre ese grafo en sentido inverso y acumula gradientes en `.grad`.

```python
import torch

w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)
y = torch.tensor(6.0)

prediction = w * x
loss = (prediction - y) ** 2
loss.backward()

assert w.grad.item() == -16.0
```

El resultado coincide con el cálculo manual `2x(wx-y)`.

## 2.4 Bucle de entrenamiento estándar

El orden correcto es:

```python
optimizer.zero_grad()
prediction = model(x)
loss = loss_function(prediction, y)
loss.backward()
optimizer.step()
```

Por defecto, PyTorch acumula gradientes. Olvidar `zero_grad()` mezcla los gradientes de varias iteraciones.

### Ejercicio 2.1

Entrenar la regresión anterior con un `nn.Linear`.

<details>
<summary>Solución 2.1</summary>

```python
import torch
from torch import nn

torch.manual_seed(42)

x = torch.tensor([[5.0], [7.0], [10.0], [8.0]])
y = torch.tensor([[15.0], [21.0], [30.0], [25.0]])

model = nn.Linear(in_features=1, out_features=1, bias=False)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

for epoch in range(100):
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_function(prediction, y)
    loss.backward()
    optimizer.step()

print(model.weight.item(), loss.item())
```

El peso debe acercarse a `3.0336` y la loss a `0.1828`.

</details>

## 2.5 `nn.Module`, modos y checkpoints

- Las clases de modelo heredan de `nn.Module`.
- `model.train()` activa comportamiento de entrenamiento, como dropout.
- `model.eval()` activa comportamiento de evaluación.
- `torch.no_grad()` evita construir el grafo durante evaluación o inferencia.
- `state_dict()` contiene parámetros y buffers persistentes.

Guardar:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "checkpoint.pt")
```

Cargar:

```python
checkpoint = torch.load("checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

# Fase 3 — Texto, tokens y batches

## 3.1 Tokenización

Un modelo no recibe texto directamente. El tokenizer transforma texto en IDs enteros.

Este proyecto comienza con tokenización por caracteres:

```text
"casa" → [id_c, id_a, id_s, id_a]
```

Ventajas: simple, vocabulario pequeño y sin tokens desconocidos si el vocabulario se construye con todo el corpus. Inconvenientes: secuencias largas y menor eficiencia semántica.

Los modelos reales suelen utilizar subwords: fragmentos frecuentes de palabras.

### Ejercicio 3.1

Construir `encode` y `decode` y comprobar el round trip.

<details>
<summary>Solución 3.1</summary>

```python
text = "hola mundo"
characters = sorted(set(text))
stoi = {character: index for index, character in enumerate(characters)}
itos = {index: character for character, index in stoi.items()}


def encode(value):
    return [stoi[character] for character in value]


def decode(token_ids):
    return "".join(itos[token_id] for token_id in token_ids)


assert decode(encode(text)) == text
```

</details>

## 3.2 Contexto y target desplazado

Un modelo autoregresivo predice el siguiente token utilizando solo tokens anteriores.

Para `tokens = [10, 20, 30, 40, 50]`:

```text
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
```

Cada posición proporciona un ejemplo:

```text
[10]             → 20
[10, 20]         → 30
[10, 20, 30]     → 40
[10, 20, 30, 40] → 50
```

`block_size` es la longitud máxima del contexto.

## 3.3 Batch de lenguaje

En este proyecto:

- `B`: batch size.
- `T`: longitud temporal o de contexto.
- `C`: canales o dimensión del embedding.

Los índices de entrada tienen shape `(B, T)`. Tras el embedding pasan a `(B, T, C)`.

### Ejercicio 3.2

Crear batches aleatorios `x, y` con el target desplazado una posición.

<details>
<summary>Solución 3.2</summary>

```python
import torch


def get_batch(source, batch_size, block_size, device="cpu"):
    starts = torch.randint(
        low=0,
        high=len(source) - block_size,
        size=(batch_size,),
    )
    x = torch.stack([source[i:i + block_size] for i in starts])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


source = torch.arange(100, dtype=torch.long)
x, y = get_batch(source, batch_size=4, block_size=8)

assert x.shape == (4, 8)
assert y.shape == (4, 8)
assert torch.equal(x[:, 1:], y[:, :-1])
```

</details>

# Fase 4 — Primer modelo de lenguaje: bigrama

## 4.1 Qué aprende un bigrama

Un bigrama predice el siguiente token usando únicamente el token actual. Su tabla tiene shape `(vocab_size, vocab_size)`.

Al indexarla con el token actual obtenemos un vector de logits: una puntuación sin normalizar para cada posible token siguiente.

## 4.2 Logits, softmax y probabilidades

Si los logits son `z`, softmax calcula:

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

- Los logits pueden ser negativos y no suman 1.
- Las probabilidades están entre 0 y 1 y suman 1.
- Sumar la misma constante a todos los logits no cambia softmax.

## 4.3 Cross-entropy

Para el token correcto `y`:

$$
L=-\log p(y)
$$

Asignar alta probabilidad al token correcto reduce la loss. `torch.nn.functional.cross_entropy` recibe logits, no probabilidades; internamente aplica una versión numéricamente estable de log-softmax.

### Ejercicio 4.1

Implementar el bigrama y generar tokens.

<details>
<summary>Solución 4.1</summary>

```python
import torch
from torch import nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)
        loss = None

        if targets is not None:
            batch_size, time, channels = logits.shape
            flat_logits = logits.reshape(batch_size * time, channels)
            flat_targets = targets.reshape(batch_size * time)
            loss = F.cross_entropy(flat_logits, flat_targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            last_logits = logits[:, -1, :]
            probabilities = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
```

</details>

# Fase 5 — Embeddings y posición

## 5.1 Token embedding

Un embedding es una tabla aprendida que asigna a cada token un vector denso de longitud `n_embd`:

```text
token ID → vector aprendido
```

Los tokens con funciones parecidas pueden acabar cerca en el espacio vectorial porque el entrenamiento ajusta sus representaciones para resolver la predicción.

`nn.Embedding(vocab_size, n_embd)` recibe índices `(B, T)` y devuelve `(B, T, C)`.

## 5.2 Positional embedding

Self-attention por sí sola no conoce el orden. Sumamos un vector de posición al vector del token:

$$
x=embedding(token)+embedding(posición)
$$

Ambos deben tener la misma dimensión `C` para poder sumarlos.

### Ejercicio 5.1

Combinar embeddings de token y posición.

<details>
<summary>Solución 5.1</summary>

```python
import torch
from torch import nn

B, T = 4, 8
vocab_size = 30
n_embd = 32

idx = torch.randint(vocab_size, (B, T))
token_embedding = nn.Embedding(vocab_size, n_embd)
position_embedding = nn.Embedding(T, n_embd)

token_vectors = token_embedding(idx)                         # (B, T, C)
position_vectors = position_embedding(torch.arange(T))      # (T, C)
x = token_vectors + position_vectors                        # broadcast en B

assert x.shape == (B, T, n_embd)
```

</details>

# Fase 6 — Self-attention causal

## 6.1 Intuición

Cada token crea tres representaciones:

- **Query:** qué información busca.
- **Key:** qué información ofrece.
- **Value:** contenido que entrega si resulta relevante.

Las similitudes entre queries y keys producen pesos de atención. Esos pesos mezclan values.

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}+mask\right)V
$$

La división por `√d_k` evita que productos grandes saturen softmax.

## 6.2 Máscara causal

Al predecir la posición `t`, el modelo no puede ver posiciones futuras. La máscara triangular superior convierte sus scores en `-∞`; softmax les asigna probabilidad cero.

Sin máscara habría data leakage: durante entrenamiento el modelo vería la respuesta que intenta predecir.

## 6.3 Shapes

| Operación | Shape |
|---|---|
| Entrada `x` | `(B, T, C)` |
| `q`, `k`, `v` | `(B, T, head_size)` |
| `q @ k.transpose(-2, -1)` | `(B, T, T)` |
| Pesos tras softmax | `(B, T, T)` |
| Pesos `@ v` | `(B, T, head_size)` |

### Ejercicio 6.1

Implementar una cabeza de atención causal.

<details>
<summary>Solución 6.1</summary>

```python
import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, time, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        attention = query @ key.transpose(-2, -1)
        attention = attention * key.shape[-1] ** -0.5
        attention = attention.masked_fill(
            self.tril[:time, :time] == 0,
            float("-inf"),
        )
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        return attention @ value
```

</details>

## 6.4 Multi-head attention

Varias cabezas pueden aprender relaciones distintas en paralelo. Sus salidas se concatenan y se proyectan de nuevo a `n_embd`.

Si `n_embd = 128` y `n_head = 4`, cada cabeza utiliza normalmente `head_size = 32`. Debe cumplirse `n_embd % n_head == 0`.

<details>
<summary>Solución multi-head attention</summary>

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd debe ser divisible por n_head")

        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(n_head)
        ])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        concatenated = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(concatenated))
```

</details>

# Fase 7 — Transformer y MiniGPT

## 7.1 Feed-forward network

Attention comunica posiciones. La feed-forward procesa cada posición de forma independiente usando la misma pequeña MLP:

```text
n_embd → 4 × n_embd → GELU → n_embd
```

## 7.2 Residual connections

En vez de reemplazar `x`, cada subcapa aprende una corrección:

$$
x_{nuevo}=x+f(x)
$$

Esto facilita el flujo de gradientes y permite redes profundas.

## 7.3 Layer normalization

LayerNorm normaliza las features de cada posición. En arquitectura pre-norm:

```text
x = x + attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

## 7.4 Dropout

Durante entrenamiento, dropout anula aleatoriamente activaciones y ayuda a regularizar. En `model.eval()` queda desactivado.

### Ejercicio 7.1

Implementar la feed-forward y un bloque Transformer pre-norm.

<details>
<summary>Solución 7.1</summary>

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            n_embd, n_head, block_size, dropout
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
```

</details>

## 7.5 Arquitectura completa

```text
token IDs
  ↓
token embeddings + position embeddings
  ↓
N × Transformer block
  ↓
final LayerNorm
  ↓
linear projection to vocabulary
  ↓
logits
```

Durante entrenamiento se comparan los logits con los targets mediante cross-entropy. Durante generación se toma la última posición, se aplica softmax y se muestrea el próximo token.

# Fase 8 — Entrenamiento, evaluación e inferencia

## 8.1 AdamW

SGD utiliza directamente el gradiente. AdamW adapta el paso por parámetro usando medias móviles y aplica weight decay de forma desacoplada. Es una opción estándar para Transformers pequeños.

El optimizador mejora el procedimiento de actualización; no sustituye la loss ni backpropagation.

## 8.2 Evaluación correcta

La evaluación debe:

1. Usar `model.eval()`.
2. Ejecutarse bajo `torch.no_grad()`.
3. Medir train y validation por separado.
4. Restaurar `model.train()` antes de continuar.

## 8.3 Generación autoregresiva

La generación repite:

```text
contexto → logits de la última posición → probabilidades
→ muestrear token → añadir al contexto → repetir
```

Solo se entregan al modelo los últimos `block_size` tokens porque su embedding posicional y su máscara tienen esa longitud máxima.

### Temperatura y top-k

- Temperatura `< 1`: distribución más concentrada y texto conservador.
- Temperatura `> 1`: más diversidad y más errores.
- Top-k: conserva solo los `k` logits mayores antes de muestrear.

## 8.4 Reproducibilidad

`torch.manual_seed` ayuda, pero GPU, kernels no deterministas, versiones y orden de operaciones pueden producir diferencias. Igual seed no siempre significa reproducibilidad bit a bit en todos los dispositivos.

## 8.5 Qué significa que el MiniGPT funcione

No se evalúa por producir conocimiento factual. Con un corpus pequeño debe:

- Reducir train loss.
- Mantener validation loss razonablemente próxima.
- Generar secuencias con patrones del corpus.
- Respetar las shapes y la causalidad.
- Guardarse y cargarse sin cambiar sus logits para una misma entrada.

# Solución final completa: `minigpt.py`

Esta solución reúne tokenizer por caracteres, dataset, batches, atención causal, Transformer, evaluación, checkpoint y generación. Requiere un archivo `input.txt` con suficiente texto en la misma carpeta.

```python
from dataclasses import dataclass, asdict
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    batch_size: int = 32
    block_size: int = 64
    max_steps: int = 3_000
    eval_interval: int = 300
    eval_iterations: int = 100
    learning_rate: float = 3e-4
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2
    seed: int = 42


CONFIG = Config()


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = choose_device()
torch.manual_seed(CONFIG.seed)
random.seed(CONFIG.seed)


class CharacterTokenizer:
    def __init__(self, text):
        self.characters = sorted(set(text))
        self.stoi = {
            character: index
            for index, character in enumerate(self.characters)
        }
        self.itos = {
            index: character
            for character, index in self.stoi.items()
        }

    @property
    def vocab_size(self):
        return len(self.characters)

    def encode(self, text):
        try:
            return [self.stoi[character] for character in text]
        except KeyError as error:
            raise ValueError(
                f"Carácter fuera del vocabulario: {error.args[0]!r}"
            ) from error

    def decode(self, token_ids):
        return "".join(self.itos[int(token_id)] for token_id in token_ids)


def load_corpus(path):
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("input.txt está vacío")
    if len(text) <= CONFIG.block_size + 1:
        raise ValueError("El corpus debe superar block_size + 1 caracteres")
    return text


def split_token_data(data, train_ratio=0.9):
    split_index = int(train_ratio * len(data))
    train_data = data[:split_index]
    validation_data = data[split_index:]

    minimum_length = CONFIG.block_size + 1
    if len(train_data) < minimum_length or len(validation_data) < minimum_length:
        raise ValueError(
            "Train y validation necesitan al menos block_size + 1 tokens. "
            "Usa un corpus mayor o reduce block_size."
        )
    return train_data, validation_data


def get_batch(source):
    starts = torch.randint(
        low=0,
        high=len(source) - CONFIG.block_size,
        size=(CONFIG.batch_size,),
    )
    x = torch.stack([
        source[start:start + CONFIG.block_size]
        for start in starts
    ])
    y = torch.stack([
        source[start + 1:start + CONFIG.block_size + 1]
        for start in starts
    ])
    return x.to(DEVICE), y.to(DEVICE)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(CONFIG.n_embd, head_size, bias=False)
        self.query = nn.Linear(CONFIG.n_embd, head_size, bias=False)
        self.value = nn.Linear(CONFIG.n_embd, head_size, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(CONFIG.block_size, CONFIG.block_size)),
        )
        self.dropout = nn.Dropout(CONFIG.dropout)

    def forward(self, x):
        _, time, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        attention = query @ key.transpose(-2, -1)
        attention = attention * key.shape[-1] ** -0.5
        attention = attention.masked_fill(
            self.causal_mask[:time, :time] == 0,
            float("-inf"),
        )
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        return attention @ value


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        if CONFIG.n_embd % CONFIG.n_head != 0:
            raise ValueError("n_embd debe ser divisible por n_head")

        head_size = CONFIG.n_embd // CONFIG.n_head
        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(CONFIG.n_head)
        ])
        self.projection = nn.Linear(CONFIG.n_embd, CONFIG.n_embd)
        self.dropout = nn.Dropout(CONFIG.dropout)

    def forward(self, x):
        concatenated = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(concatenated))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(CONFIG.n_embd, 4 * CONFIG.n_embd),
            nn.GELU(),
            nn.Linear(4 * CONFIG.n_embd, CONFIG.n_embd),
            nn.Dropout(CONFIG.dropout),
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(CONFIG.n_embd)
        self.layer_norm_2 = nn.LayerNorm(CONFIG.n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, CONFIG.n_embd)
        self.position_embedding = nn.Embedding(
            CONFIG.block_size,
            CONFIG.n_embd,
        )
        self.blocks = nn.Sequential(*[
            Block() for _ in range(CONFIG.n_layer)
        ])
        self.final_layer_norm = nn.LayerNorm(CONFIG.n_embd)
        self.language_model_head = nn.Linear(CONFIG.n_embd, vocab_size)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch_size, time = idx.shape
        if time > CONFIG.block_size:
            raise ValueError("La secuencia supera block_size")

        token_vectors = self.token_embedding(idx)
        positions = torch.arange(time, device=idx.device)
        position_vectors = self.position_embedding(positions)
        x = token_vectors + position_vectors
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.language_model_head(x)

        loss = None
        if targets is not None:
            logits_for_loss = logits.reshape(
                batch_size * time,
                self.vocab_size,
            )
            targets_for_loss = targets.reshape(batch_size * time)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if temperature <= 0:
            raise ValueError("temperature debe ser mayor que cero")

        for _ in range(max_new_tokens):
            context = idx[:, -CONFIG.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                threshold = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx


@torch.no_grad()
def estimate_loss(model, datasets):
    result = {}
    model.eval()

    for split_name, source in datasets.items():
        losses = torch.zeros(CONFIG.eval_iterations)
        for iteration in range(CONFIG.eval_iterations):
            x, y = get_batch(source)
            _, loss = model(x, y)
            losses[iteration] = loss.item()
        result[split_name] = losses.mean().item()

    model.train()
    return result


def save_checkpoint(path, model, optimizer, tokenizer, step):
    torch.save(
        {
            "step": step,
            "config": asdict(CONFIG),
            "characters": tokenizer.characters,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def run_smoke_tests(model, tokenizer, train_data):
    if all(character in tokenizer.stoi for character in "hola"):
        assert tokenizer.decode(tokenizer.encode("hola")) == "hola"

    x, y = get_batch(train_data)
    assert x.shape == (CONFIG.batch_size, CONFIG.block_size)
    assert y.shape == (CONFIG.batch_size, CONFIG.block_size)

    logits, loss = model(x, y)
    assert logits.shape == (
        CONFIG.batch_size,
        CONFIG.block_size,
        tokenizer.vocab_size,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def main():
    text = load_corpus("input.txt")
    tokenizer = CharacterTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, validation_data = split_token_data(data)
    datasets = {"train": train_data, "validation": validation_data}

    model = MiniGPT(tokenizer.vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.learning_rate,
    )

    run_smoke_tests(model, tokenizer, train_data)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={DEVICE}, vocab={tokenizer.vocab_size}, "
        f"parameters={parameter_count:,}"
    )

    for step in range(CONFIG.max_steps):
        if step % CONFIG.eval_interval == 0 or step == CONFIG.max_steps - 1:
            losses = estimate_loss(model, datasets)
            print(
                f"step={step}, "
                f"train_loss={losses['train']:.4f}, "
                f"validation_loss={losses['validation']:.4f}"
            )

        x, y = get_batch(train_data)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    save_checkpoint(
        path="minigpt_checkpoint.pt",
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        step=CONFIG.max_steps,
    )

    model.eval()
    initial_context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated = model.generate(
        initial_context,
        max_new_tokens=500,
        temperature=0.9,
        top_k=20,
    )[0].tolist()
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
```

## Cómo cargar el checkpoint final

Debe reconstruirse exactamente la misma configuración y el mismo vocabulario antes de cargar los pesos.

```python
checkpoint = torch.load("minigpt_checkpoint.pt", map_location=DEVICE)
characters = checkpoint["characters"]
tokenizer = CharacterTokenizer("".join(characters))

model = MiniGPT(tokenizer.vocab_size).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Si se cambia `n_embd`, `n_head`, `n_layer`, `block_size` o el vocabulario, las shapes ya no coincidirán.

# Pruebas y errores frecuentes

## Pruebas mínimas

Antes de entrenar durante minutos u horas:

1. `decode(encode(text)) == text`.
2. `x.shape == y.shape == (B, T)`.
3. `logits.shape == (B, T, vocab_size)`.
4. La loss inicial es finita.
5. El modelo sobreajusta un batch pequeño si se entrena repetidamente con él.
6. En modo `eval`, el mismo input produce los mismos logits.
7. Guardar y cargar conserva los logits.

Prueba de checkpoint:

```python
model.eval()
sample_x, _ = get_batch(train_data)

with torch.no_grad():
    before, _ = model(sample_x)

torch.save(model.state_dict(), "test_state.pt")
clone = MiniGPT(tokenizer.vocab_size).to(DEVICE)
clone.load_state_dict(torch.load("test_state.pt", map_location=DEVICE))
clone.eval()

with torch.no_grad():
    after, _ = clone(sample_x)

assert torch.allclose(before, after)
```

## Tabla de diagnóstico

| Síntoma | Causa probable | Comprobación |
|---|---|---|
| Loss es `nan` | Learning rate alto, logits inestables o datos inválidos | Reducir LR y comprobar `torch.isfinite` |
| Loss no baja | Gradientes nulos, targets mal desplazados o LR inadecuado | Inspeccionar `.grad`, `x` e `y` |
| Error de matrix multiplication | Shapes incompatibles | Imprimir `(B, T, C)` en cada frontera |
| Error en embedding | Índice fuera de `[0, vocab_size)` o dtype incorrecto | Usar `torch.long` y revisar tokenizer |
| GPU/CPU mismatch | Tensores en devices distintos | Mover modelo, inputs y targets al mismo device |
| Validation mucho peor | Overfitting o distribución distinta | Menos capacidad, más datos, dropout |
| Texto incoherente | Corpus pequeño, poco entrenamiento o modelo limitado | Comparar pérdidas antes de aumentar tamaño |
| Generación falla al superar contexto | No recortar a `block_size` | Usar `idx[:, -block_size:]` |
| Gradientes crecen cada step | Falta `zero_grad()` | Limpiar antes de `backward()` |
| Resultados cambian en evaluación | Falta `model.eval()` o hay muestreo | Desactivar dropout y distinguir logits de sampling |

## Experimentos recomendados

Cambiar una sola variable cada vez y registrar train loss, validation loss, velocidad y calidad subjetiva:

1. `block_size`: 32, 64, 128.
2. `n_embd`: 64, 128, 256.
3. `n_layer`: 2, 4, 6.
4. `n_head`: 2, 4, 8 manteniendo divisibilidad.
5. `dropout`: 0.0, 0.1, 0.2.
6. Learning rate: `1e-4`, `3e-4`, `1e-3`.
7. Temperatura: 0.6, 0.9, 1.2.
8. Bigram vs Transformer usando el mismo corpus.

No se debe concluir que un modelo es mejor mirando únicamente train loss. La comparación principal utiliza validation loss y coste de cómputo.

# Glosario

| Término | Definición breve |
|---|---|
| **Autograd** | Sistema que calcula gradientes automáticamente. |
| **Batch size** | Número de secuencias procesadas simultáneamente. |
| **Block size** | Longitud máxima del contexto. |
| **Checkpoint** | Estado guardado del modelo y, normalmente, del optimizador. |
| **Corpus** | Texto utilizado como datos. |
| **Cross-entropy** | Loss para clasificación o predicción de tokens. |
| **Data leakage** | Uso de información que no debería estar disponible. |
| **Embedding** | Vector aprendido que representa un ID discreto. |
| **Epoch** | Pasada completa por train. |
| **Generalización** | Rendimiento con datos no usados para ajustar parámetros. |
| **Gradiente** | Derivada de la loss respecto a cada parámetro. |
| **Inference** | Uso del modelo sin entrenamiento. |
| **Learning rate** | Escala de la actualización de parámetros. |
| **Logit** | Puntuación sin normalizar previa a softmax. |
| **Loss** | Medida numérica del error. |
| **Mask causal** | Impide atender a tokens futuros. |
| **Optimizer** | Algoritmo que actualiza parámetros usando gradientes. |
| **Overfitting** | Ajuste excesivo a train con mala generalización. |
| **Parameter** | Valor aprendido por el modelo. |
| **Perplexity** | `exp(cross_entropy)`; incertidumbre media del modelo. |
| **Residual connection** | Suma la entrada a la salida de una subcapa. |
| **Self-attention** | Mezcla información entre posiciones de una secuencia. |
| **Softmax** | Convierte logits en una distribución de probabilidad. |
| **Target** | Respuesta correcta que debe predecirse. |
| **Tensor** | Array n-dimensional con dtype y device. |
| **Token** | Unidad discreta de texto representada por un ID. |
| **Transformer** | Arquitectura basada en attention, MLP, normalización y residuales. |
| **Underfitting** | Modelo demasiado simple o insuficientemente entrenado. |

## Resumen mental final

```text
1. El tokenizer convierte texto en IDs.
2. Los embeddings convierten IDs en vectores.
3. Se añade información de posición.
4. Attention permite que cada token consulte su contexto anterior.
5. La máscara causal impide mirar el futuro.
6. Las MLP procesan las representaciones posición por posición.
7. Residuales y LayerNorm estabilizan una red profunda.
8. La proyección final produce un logit por token posible.
9. Cross-entropy compara logits con el siguiente token real.
10. Backprop calcula gradientes y AdamW actualiza parámetros.
11. En inferencia, softmax y sampling eligen un token y el ciclo se repite.
```

Ese es el flujo completo de un GPT pequeño, desde texto hasta generación autoregresiva.

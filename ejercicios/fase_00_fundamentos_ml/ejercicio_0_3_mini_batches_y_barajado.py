"""Ejercicio 0.3: entrenar usando mini-batches.

Tareas:
1. Crear batches de tamaño 2.
2. Barajar los datos una vez por epoch.
3. Actualizar w después de cada batch.
4. Contar el número total de actualizaciones.
"""

import random


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


def mse_gradient(dataset, w):
    gradients = []
    for x, target in dataset:
        prediction = predict(x, w)
        gradients.append(2 * x * (prediction - target))

    return sum(gradients) / len(gradients)


def make_batches(dataset, batch_size, rng):
    shuffled = list(dataset)
    rng.shuffle(shuffled)
    return [
        shuffled[start:start + batch_size]
        for start in range(0, len(shuffled), batch_size)
    ]
    


def train_mini_batch(dataset, initial_w, learning_rate, epochs, batch_size, seed=42):
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


dataset = [(5, 15), (7, 21), (10, 30), (8, 25)]

final_w, updates = train_mini_batch(dataset=dataset,initial_w=0.0,learning_rate=0.005,epochs=25,batch_size=2,)

assert updates == 50  # 2 batches × 25 epochs
print(final_w, mse(dataset, final_w), updates)

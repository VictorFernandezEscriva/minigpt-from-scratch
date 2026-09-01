"""Ejercicio 1.2: implementar el forward de una red 2 -> 3 -> 1 con ReLU."""

import numpy as np


def relu(x):
    # TODO: devuelve max(0, x) elemento a elemento.
    raise NotImplementedError


def forward_mlp(x, W1, b1, W2, b2):
    # TODO: capa lineal, ReLU y segunda capa lineal.
    raise NotImplementedError


x = np.array([[1.0, 2.0], [3.0, 4.0]])
W1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
b1 = np.zeros(3)
W2 = np.array([[0.7], [0.8], [0.9]])
b2 = np.zeros(1)

# TODO: ejecuta el forward y comprueba que la salida tiene shape (2, 1).

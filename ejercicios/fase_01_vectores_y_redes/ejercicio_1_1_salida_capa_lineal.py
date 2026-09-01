"""Ejercicio 1.1: calcular la salida de una capa con dos neuronas.

x = [2, 3]
W = [[1, -1],
     [2,  4]]
b = [0.5, -0.5]
"""

import numpy as np


x = np.array([2.0, 3.0])
W = np.array([[1.0, -1.0], [2.0, 4.0]])
b = np.array([0.5, -0.5])

y = x @ W + b

print(f"x shape: {x.shape}")
print(f"W shape: {W.shape}")
print(f"b shape: {b.shape}")
print(f"y shape: {y.shape}")
print(f"y: {y}")

expected_y = np.array([8.5, 9.5])

assert y.shape == (2,)
assert np.allclose(y, expected_y)

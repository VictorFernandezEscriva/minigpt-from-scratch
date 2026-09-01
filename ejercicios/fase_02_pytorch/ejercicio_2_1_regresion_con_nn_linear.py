"""Ejercicio 2.1: entrenar la regresión anterior con torch.nn.Linear."""

import torch
from torch import nn


torch.manual_seed(42)
x = torch.tensor([[5.0], [7.0], [10.0], [8.0]])
y = torch.tensor([[15.0], [21.0], [30.0], [25.0]])

# TODO: crea nn.Linear sin bias, nn.MSELoss y un optimizador SGD.
# TODO: entrena durante 100 epochs siguiendo este orden:
# zero_grad -> forward -> loss -> backward -> step.
# TODO: comprueba que el peso se aproxima a 3.0336.

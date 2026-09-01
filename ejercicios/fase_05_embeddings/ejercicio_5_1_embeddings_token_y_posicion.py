"""Ejercicio 5.1: combinar embeddings de token y de posición."""

import torch
from torch import nn


B, T = 4, 8
vocab_size = 30
n_embd = 32
idx = torch.randint(vocab_size, (B, T))

# TODO: crea ambos nn.Embedding.
# TODO: obtiene los vectores de token y posición y súmalos.
# TODO: comprueba que el resultado tiene shape (B, T, n_embd).

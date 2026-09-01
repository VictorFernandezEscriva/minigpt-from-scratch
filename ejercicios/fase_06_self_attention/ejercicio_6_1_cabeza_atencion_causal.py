"""Ejercicio 6.1: implementar una cabeza de self-attention causal."""

import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        # TODO: crea las proyecciones key, query y value sin bias.
        # TODO: registra una máscara triangular inferior como buffer.
        # TODO: crea la capa de dropout.

    def forward(self, x):
        # TODO: calcula atención escalada, aplica la máscara y softmax,
        # después mezcla los values.
        raise NotImplementedError


# TODO: prueba la clase con un tensor (B, T, C) y verifica la shape de salida.

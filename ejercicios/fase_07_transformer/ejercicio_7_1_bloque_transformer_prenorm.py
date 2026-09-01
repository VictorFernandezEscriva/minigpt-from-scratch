"""Ejercicio 7.1: implementar feed-forward y un bloque Transformer pre-norm.

Para completar Block necesitarás tu implementación de MultiHeadAttention de la
fase anterior o una versión equivalente creada en este archivo.
"""

from torch import nn


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        # TODO: Linear(C, 4C) -> GELU -> Linear(4C, C) -> Dropout.

    def forward(self, x):
        # TODO: devuelve el resultado de la red.
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        # TODO: crea multi-head attention, feed-forward y dos LayerNorm.

    def forward(self, x):
        # TODO: aplica pre-norm y las dos conexiones residuales.
        raise NotImplementedError

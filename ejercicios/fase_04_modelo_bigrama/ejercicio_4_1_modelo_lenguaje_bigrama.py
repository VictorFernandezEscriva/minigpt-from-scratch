"""Ejercicio 4.1: implementar un modelo de lenguaje bigrama y generar tokens."""

import torch
from torch import nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # TODO: crea una tabla de embedding vocab_size x vocab_size.

    def forward(self, idx, targets=None):
        # TODO: calcula logits y, si hay targets, la cross-entropy.
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # TODO: muestrea y concatena un token en cada iteración.
        raise NotImplementedError

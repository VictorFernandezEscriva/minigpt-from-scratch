"""Ejercicio 3.2: crear batches aleatorios con el target desplazado."""

import torch


def get_batch(source, batch_size, block_size, device="cpu"):
    # TODO: elige inicios aleatorios y crea x e y desplazado una posición.
    raise NotImplementedError


source = torch.arange(100, dtype=torch.long)

# TODO: crea un batch con B=4 y T=8.
# TODO: comprueba shapes y que x[:, 1:] sea igual a y[:, :-1].

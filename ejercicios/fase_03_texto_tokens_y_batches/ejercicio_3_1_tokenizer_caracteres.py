"""Ejercicio 3.1: construir encode/decode y comprobar el round trip."""


text = "hola mundo"
characters = sorted(set(text))

# TODO: crea los diccionarios carácter->ID e ID->carácter.
stoi = {}
itos = {}


def encode(value):
    # TODO: convierte un texto en una lista de IDs.
    raise NotImplementedError


def decode(token_ids):
    # TODO: convierte una lista de IDs en texto.
    raise NotImplementedError


# TODO: comprueba que decode(encode(text)) == text.

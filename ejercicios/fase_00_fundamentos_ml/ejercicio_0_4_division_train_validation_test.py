"""Ejercicio 0.4: dividir datos sin perder ni duplicar ejemplos."""

import random


def split_dataset(dataset, train_ratio=0.7, validation_ratio=0.15, seed=42):
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio debe estar entre 0 y 1")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio debe estar entre 0 y 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Debe quedar espacio para test")

    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    validation_end = train_end + int(len(shuffled) * validation_ratio)

    train_data = shuffled[:train_end]
    validation_data = shuffled[train_end:validation_end]
    test_data = shuffled[validation_end:]

    assert len(train_data) + len(validation_data) + len(test_data) == len(dataset)
    return train_data, validation_data, test_data


dataset = [
    (x, 3 * x + (1 if x % 4 == 0 else 0))
    for x in range(1, 21)
]

train_data, validation_data, test_data = split_dataset(dataset)

assert len(train_data) == 14
assert len(validation_data) == 3
assert len(test_data) == 3

assert len(train_data) + len(validation_data) + len(test_data) == len(dataset)
print(f"Train: {len(train_data)} ejemplos")
print(f"Validation: {len(validation_data)} ejemplos")
print(f"Test: {len(test_data)} ejemplos")

print("\nTrain:", train_data)
print("\nValidation:", validation_data)
print("\nTest:", test_data)
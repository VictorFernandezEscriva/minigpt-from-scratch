"""Ejercicio 0.1: comparar distintos saltos al buscar el mejor valor de w."""


def predict(x, w):
    return w * x


def squared_error(prediction, target):
    return (prediction - target) ** 2


def mse(dataset, w):
    squared_error_vector = []
    for x, target in dataset:
        prediction = predict(x, w)
        squared_error_vector.append(squared_error(prediction, target))

    return sum(squared_error_vector) / len(squared_error_vector)


def find_best_w(dataset, step_size):
    min_mse = float("inf")
    best_w = None
    number_of_steps = int(5.0 / step_size) + 1

    for step in range(number_of_steps):
        w = step * step_size
        current_mse = mse(dataset, w)
        if current_mse < min_mse:
            min_mse = current_mse
            best_w = w

    return min_mse, best_w


dataset = [(5, 15), (7, 21), (10, 30), (8, 25)]
step_size_dataset = [0.1, 0.01, 0.001]

for step_size in step_size_dataset:
    min_mse, best_w = find_best_w(dataset, step_size)
    print(
        f"Step size = {step_size}, "
        f"Minimum MSE = {min_mse:.4f}, "
        f"best W = {best_w:.4f}"
    )

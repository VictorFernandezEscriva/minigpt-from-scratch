"""Práctica inicial: buscar por fuerza bruta el mejor valor de w."""


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


dataset = [
    (5, 15),
    (7, 21),
    (10, 30),
]
min_mse = float("inf")
best_w = None

assert predict(5, 2) == 10
assert squared_error(10, 15) == 25
assert mse(dataset, 2) == 58

for step in range(51):
    w = step / 10
    current_mse = mse(dataset, w)
    if current_mse < min_mse:
        min_mse = current_mse
        best_w = w

print(f"Minimum MSE = {min_mse} and best W = {best_w}")

output_predicted_with_input_8 = predict(8, best_w)
print(f"Output predicted = {output_predicted_with_input_8}")

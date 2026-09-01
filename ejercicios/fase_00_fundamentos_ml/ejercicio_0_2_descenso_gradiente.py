"""Ejercicio 0.2: entrenar w con gradient descent y tres learning rates."""


def predict(x, w):
    return w * x


def squared_error(prediction, target):
    return (prediction - target) ** 2


def mse(dataset, w):
    squared_errors = []
    for x, target in dataset:
        prediction = predict(x, w)
        squared_errors.append(squared_error(prediction, target))

    return sum(squared_errors) / len(squared_errors)


def mse_gradient(dataset, w):
    gradients = []
    for x, target in dataset:
        prediction = predict(x, w)
        gradients.append(2 * x * (prediction - target))

    return sum(gradients) / len(gradients)


def train(dataset, initial_w, learning_rate, epochs):
    w = initial_w
    for epoch in range(epochs):
        current_mse = mse(dataset, w)
        current_gradient = mse_gradient(dataset, w)

        print(
            f"Epoch = {epoch}, "
            f"w = {w:.6f}, "
            f"loss = {current_mse:.6f}, "
            f"gradient = {current_gradient:.6f}"
        )
        w = w - (learning_rate * current_gradient)

    return w


dataset = [(5, 15), (7, 21), (10, 30), (8, 25)]
initial_w = 0.0
epochs = 25
learning_rates = [0.0001, 0.005, 0.02]

for learning_rate in learning_rates:
    final_w = train(dataset, initial_w, learning_rate, epochs)
    print(
        f"Learning rate = {learning_rate}, "
        f"final w = {final_w:.6f}, "
        f"final loss = {mse(dataset, final_w):.6f}"
    )

import numpy as np

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

y = np.array([[0], [1], [1], [0]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


input_size = 3
hidden_size = 4
output_size = 1
learning_rate = 0.1
num_epochs = 10000

np.random.seed(1)
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

for epoch in range(num_epochs):
    hidden_layer = sigmoid(np.dot(X, W1))
    output_layer = sigmoid(np.dot(hidden_layer, W2))

    loss = y - output_layer

    output_delta = loss * sigmoid_derivative(output_layer)
    hidden_delta = output_delta.dot(W2.T) * sigmoid_derivative(hidden_layer)

    W2 += hidden_layer.T.dot(output_delta) * learning_rate
    W1 += X.T.dot(hidden_delta) * learning_rate

test_input = np.array([[0, 0, 1]])
test_output = sigmoid(np.dot(sigmoid(np.dot(test_input, W1)), W2))
print(test_output)

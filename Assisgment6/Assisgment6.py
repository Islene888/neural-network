import numpy as np


class ActivationFunctions:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


class NeuralNetwork:
    def __init__(self, layers, activation_functions):
        self.layers = layers
        self.activation_functions = activation_functions
        self.weights = [np.random.randn(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.biases = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]

    def forward_propagation(self, X):
        self.Z = []
        self.A = [X]
        for i in range(len(self.layers) - 1):
            self.Z.append(np.dot(self.A[-1], self.weights[i]) + self.biases[i])
            self.A.append(self.activation_functions[i](self.Z[-1]))
        return self.A[-1]

    def loss_function(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred))

    def backpropagation(self, X, y_true, learning_rate=0.01):
        m = X.shape[0]
        delta = (self.A[-1] - y_true) / m
        for i in range(len(self.layers) - 2, -1, -1):
            grad_weights = np.dot(self.A[i].T, delta)
            grad_biases = np.sum(delta, axis=0, keepdims=True)
            delta = np.dot(delta, self.weights[i].T) * self.activation_functions[i](self.Z[i], derivative=True)
            self.weights[i] -= learning_rate * grad_weights
            self.biases[i] -= learning_rate * grad_biases

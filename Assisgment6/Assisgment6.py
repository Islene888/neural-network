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


class Neuron:
    def __init__(self, activation_function):
        self.activation_function = activation_function

    def activate(self, x):
        return self.activation_function(x)

    def derivative(self, x):
        # Derivative of activation function
        if self.activation_function == ActivationFunctions.linear:
            return np.ones_like(x)
        elif self.activation_function == ActivationFunctions.relu:
            return np.where(x > 0, 1, 0)
        elif self.activation_function == ActivationFunctions.sigmoid:
            return self.activate(x) * (1 - self.activate(x))
        elif self.activation_function == ActivationFunctions.tanh:
            return 1 - np.square(self.activate(x))
        elif self.activation_function == ActivationFunctions.softmax:
            raise NotImplementedError("Derivative of softmax activation function is not implemented.")
        else:
            raise ValueError("Unknown activation function.")


class Parameter:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def update(self, delta_weights, delta_biases, learning_rate):
        self.weights -= learning_rate * delta_weights
        self.biases -= learning_rate * delta_biases


class ForwardPropagation:
    @staticmethod
    def propagate(X, weights, biases, activation_function):
        Z = np.dot(X, weights) + biases
        A = activation_function.activate(Z)
        return Z, A


class NeuralNetwork:
    def __init__(self, layers, activation_functions):
        self.layers = layers
        self.activation_functions = activation_functions
        self.neurons = [Neuron(activation) for activation in activation_functions]
        self.parameters = [Parameter(np.random.randn(layers[i], layers[i + 1]),
                                      np.zeros((1, layers[i + 1]))) for i in range(len(layers) - 1)]

    def forward_propagation(self, X):
        self.Z, self.A = [], [X]
        for i in range(len(self.layers) - 1):
            Z, A = ForwardPropagation.propagate(self.A[-1], self.parameters[i].weights,
                                                 self.parameters[i].biases, self.neurons[i])
            self.Z.append(Z)
            self.A.append(A)
        return self.A[-1]

    def loss_function(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred))

    def backpropagation(self, X, y_true, learning_rate=0.01):
        m = X.shape[0]
        delta = (self.A[-1] - y_true) / m
        for i in range(len(self.layers) - 2, -1, -1):
            grad_weights = np.dot(self.A[i].T, delta)
            grad_biases = np.sum(delta, axis=0, keepdims=True)
            delta = np.dot(delta, self.parameters[i].weights.T) * self.neurons[i].derivative(self.Z[i])
            self.parameters[i].update(grad_weights, grad_biases, learning_rate)

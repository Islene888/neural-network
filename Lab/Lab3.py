import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

class Layer:
    def __init__(self, size_input, size_output, activation_function='relu'):
        self.W = np.random.randn(size_input, size_output) * np.sqrt(2 / size_input)
        self.b = np.zeros((1, size_output))
        self.activation = activation_function

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W) + self.b
        if self.activation == 'relu':
            self.A = relu(self.Z)
        elif self.activation == 'sigmoid':
            self.A = sigmoid(self.Z)
        return self.A

    def backward(self, dA, lambda_reg=0.01):
        m = dA.shape[0]
        if self.activation == 'relu':
            dZ = dA * relu_derivative(self.Z)
        elif self.activation == 'sigmoid':
            dZ = dA * sigmoid_derivative(self.Z)
        self.dW = np.dot(self.A_prev.T, dZ) / m + (lambda_reg * self.W / m)
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = np.dot(dZ, self.W.T)
        return dA_prev

    def update_parameters(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

class NeuralNetwork:
    def __init__(self, layer_dims, activation_functions, lambda_reg=0.01, learning_rate=0.001):
        self.layers = [Layer(layer_dims[i], layer_dims[i + 1], activation_functions[i]) for i in range(len(layer_dims) - 1)]
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate

    def forward_propagation(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def compute_cost(self, AL, Y):
        m = Y.shape[0]
        AL_clipped = np.clip(AL, 1e-10, 1 - 1e-10)
        cross_entropy_cost = -np.sum(Y * np.log(AL_clipped) + (1 - Y) * np.log(1 - AL_clipped)) / m
        L2_cost = sum(np.sum(np.square(layer.W)) for layer in self.layers)
        L2_cost = (self.lambda_reg / (2 * m)) * L2_cost
        return np.squeeze(cross_entropy_cost + L2_cost)

    def backward_propagation(self, AL, Y):
        m = Y.shape[0]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA = dAL
        for layer in reversed(self.layers):
            dA = layer.backward(dA, self.lambda_reg)

    def train(self, X, Y, num_iterations=1000, print_cost=True):
        np.random.seed(1)  # for consistent results
        costs = []
        for i in range(num_iterations):
            AL = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            self.backward_propagation(AL, Y)
            for layer in self.layers:
                layer.update_parameters(self.learning_rate)
            if print_cost and i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost:.6f}")
        return costs

# Example setup and training call
(x_train, y_train), (x_test, y_test) = mnist.load_data()
filter_index = np.where((y_train == 0) | (y_train == 1))
x_train, y_train = x_train[filter_index], y_train[filter_index]
x_train = x_train.reshape(-1, 784) / 255.0
y_train = y_train.reshape(-1, 1)

nn = NeuralNetwork([784, 128, 64, 1], ['relu', 'relu', 'sigmoid'])
nn.train(x_train, y_train, num_iterations=2500, print_cost=True)



# output:
# C:\Users\40825\AppData\Local\Programs\Python\Python39\python.exe D:\Notets\python\Neurons\Lab\Lab3.py
# 2024-04-11 17:33:03.792806: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# WARNING:tensorflow:From C:\Users\40825\AppData\Roaming\Python\Python39\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
#
# Cost after iteration 0: 0.728075
# Cost after iteration 100: 0.507697
# Cost after iteration 200: 0.372771
# Cost after iteration 300: 0.279424
# Cost after iteration 400: 0.216941
# Cost after iteration 500: 0.173959
# Cost after iteration 600: 0.143313
# Cost after iteration 700: 0.120762
# Cost after iteration 800: 0.103721
# Cost after iteration 900: 0.090537
# Cost after iteration 1000: 0.080111
# Cost after iteration 1100: 0.071710
# Cost after iteration 1200: 0.064833
# Cost after iteration 1300: 0.059121
# Cost after iteration 1400: 0.054314
# Cost after iteration 1500: 0.050225
# Cost after iteration 1600: 0.046713
# Cost after iteration 1700: 0.043668
# Cost after iteration 1800: 0.041007
# Cost after iteration 1900: 0.038664
# Cost after iteration 2000: 0.036586
# Cost after iteration 2100: 0.034733
# Cost after iteration 2200: 0.033072
# Cost after iteration 2300: 0.031573
# Cost after iteration 2400: 0.030215
#
# 进程已结束，退出代码为 0

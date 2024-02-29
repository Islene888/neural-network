import numpy as np


class Parameters:
    def __init__(self):
        self.weights = None
        self.bias = None

    def set_bias(self, bias):
        self.bias = bias

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias


class Neuron:
    def __init__(self, input_size):
        self.weights = None
        self.bias = None
        self.input_size = input_size
        self.aggregate_signal = None
        self.activation = None
        self.output = None

    def neuron(self, inputs):
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be set before forward pass")

        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Inputs shape ({inputs.shape}) is incompatible with weights shape ({self.weights.shape})")

        self.aggregate_signal = np.sum(np.dot(inputs, self.weights.T) + self.bias)
        self.activation = self.layer.activation(self.aggregate_signal)
        self.output = self.activation


class Activation:
    def __init__(self, type):
        self.type = type

    def __call__(self, inputs):
        if self.type == "relu":
            return np.maximum(0, inputs)
        elif self.type == "sigmoid":
            return 1 / (1 + np.exp(-inputs))
        else:
            raise ValueError(f"Invalid activation function type: {self.type}")


class Layer:
    def __init__(self, neurons, parameters, activation_type):
        self.neurons = neurons
        self.parameters = parameters
        self.weights = self.parameters.get_weights()
        self.bias = self.parameters.get_bias()
        self.activation_type = activation_type
        self.activation = Activation(activation_type)
        self.neurons_layer = len(neurons)

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            neuron.weights = np.random.rand(self.neurons_layer)
            neuron.bias = self.bias
            neuron.layer = self
            neuron.neuron(inputs)
            outputs.append(neuron.output)
        return np.array(outputs)


input_size = 2
num_neurons = 3

inputs = np.random.randint(2, size=(input_size, num_neurons))

parameters = Parameters()
parameters.set_bias(np.random.rand(input_size))

layer = Layer([Neuron(input_size) for _ in range(num_neurons)], parameters, "sigmoid")
outputs = layer.forward(inputs)

print("Input:", inputs)
print("\nOutput:", outputs)
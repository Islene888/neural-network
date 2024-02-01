import numpy as np

# 定义激活函数类
class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

# 定义神经元类
class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def forward(self, inputs):
        self.inputs = inputs
        self.output = Activation.sigmoid(np.dot(self.weights, inputs) + self.bias)

    def backward(self, d_error):
        d_output = d_error * Activation.sigmoid_derivative(self.output)
        self.d_weights = np.outer(d_output, self.inputs)
        self.d_bias = d_output * 1
        self.d_inputs = np.dot(d_output, self.weights)

# 定义神经层类
class Layer:
    def __init__(self, input_size, num_neurons):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = [neuron.forward(inputs) for neuron in self.neurons]

    def backward(self, d_error):
        self.d_inputs = [neuron.backward(d_error) for neuron in self.neurons]

# 定义参数类
class Parameters:
    def __init__(self, model):
        self.model = model

    def get_all_weights(self):
        all_weights = []
        for layer in self.model.layers:
            for neuron in layer.neurons:
                all_weights.extend(neuron.weights)
                all_weights.append(neuron.bias)
        return all_weights

    def set_all_weights(self, weights):
        index = 0
        for layer in self.model.layers:
            for neuron in layer.neurons:
                neuron.weights = weights[index:index+len(neuron.weights)]
                index += len(neuron.weights)
                neuron.bias = weights[index]
                index += 1

# 定义模型类
class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = [Layer(input_size, hidden_size), Layer(hidden_size, output_size)]

    def forward(self, inputs):
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.outputs

    def backward(self, d_error):
        for layer in reversed(self.layers):
            layer.backward(d_error)
            d_error = layer.d_inputs

# 定义损失函数类
class LossFunction:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / len(y_true)

# 定义前向传播类
class ForwardProp:
    def __init__(self, model):
        self.model = model

    def forward(self, inputs):
        self.model.forward(inputs)
        return self.model.layers[-1].outputs

# 定义反向传播类
class BackProp:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def backward(self, inputs, targets):
        predictions = ForwardProp(self.model).forward(inputs)
        loss = self.loss_fn.mean_squared_error(targets, predictions)
        d_loss = self.loss_fn.mean_squared_error_derivative(targets, predictions)
        self.model.backward(d_loss)
        return loss

# 定义梯度下降类
class GradDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, params, d_params):
        for param, d_param in zip(params, d_params):
            param -= self.learning_rate * d_param

# 定义训练类
class Training:
    def train(self, model, X_train, y_train, epochs, loss_fn, optimizer):
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in zip(X_train, y_train):
                loss = BackProp(model, loss_fn).backward(inputs, targets)
                total_loss += loss
                weights = Parameters(model).get_all_weights()
                d_weights = [neuron.d_weights for layer in model.layers for neuron in layer.neurons]
                d_biases = [neuron.d_bias for layer in model.layers for neuron in layer.neurons]
                optimizer.update(weights, d_weights + d_biases)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X_train)}")

# 示例用法
input_size = 2
hidden_size = 4
output_size = 1
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])
model = Model(input_size, hidden_size, output_size)
optimizer = GradDescent(learning_rate=0.1)
trainer = Training()
trainer.train(model, X_train, y_train, epochs=1000, loss_fn=LossFunction, optimizer=optimizer)

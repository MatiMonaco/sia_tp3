import numpy as np
import pandas as pd

def sigmoid(x):
    return np.tanh(x)

class Neuron:

    def __init__(self, input_length):
        self.weights = np.random.rand(1, input_length+1)

    def excite(self, inputs):
        return np.dot(self.weights[0][1:], inputs)

    def activation(self, inputs):
        return np.tanh(self.excite(inputs))

    # def backpropagation(self, input, grad_output):


class Layer:

    def __init__(self, n_of_neurons, input_length, hidden=False):
        self.hidden = hidden
        self.n_of_neurons = n_of_neurons

        # Matriz de pesos correspondientes a cada neurona y array de biases
        self.weights = np.random.rand(n_of_neurons, input_length)
        self.biases = np.random.rand(n_of_neurons, 1)

    # Recibe UN input y lo procesa
    def propagate(self, input_):
        return sigmoid(np.dot(self.weights, input_) + self.biases)

    def test(self):
        print("hidden: {}; neurons: {}".format(self.hidden, self.n_of_neurons))


class Network:

    # 1 output layer y el resto hidden
    def __init__(self, n_of_layers, n_of_neurons, input_length, n_of_outputs):
        self.n_of_layers = n_of_layers
        self.n_of_outputs = n_of_outputs
        self.layers = np.empty(n_of_layers, dtype=object)
        for i in range(n_of_layers-1):
            self.layers[i] = Layer(n_of_neurons, input_length, True)
        self.layers[n_of_layers-1] = Layer(n_of_outputs, n_of_neurons)

    def train(self, inputs, outputs):
        return 1

    def predict(self, input_):
        outputs = self.layers[0].propagate(input_)
        return self.predict_rec(outputs, 1)

    def predict_rec(self, input_, layer):
        if layer == len(self.layers) - 1:
            return self.layers[layer].propagate(input_)

        outputs = self.layers[layer].propagate(input_)
        return self.predict_rec(outputs, layer+1)

    def test(self):
        print("layers: {}; outputs: {};".format(self.n_of_layers, self.n_of_outputs))
        for layer in self.layers:
            print(layer.test())


network = Network(2, 3, 5, 2)
inputsss = np.empty([5, 1])
print(network.predict(inputsss))

# a = np.random.rand(3, 5)
# inputsss = np.empty([5, 1])
# b = np.random.rand(3, 1)
# print(a)
# print(inputsss)
# print(b)
# output = sigmoid(np.dot(a, inputsss) + b)
# print(output)
#
# for i in range(3):
#     print("Neuron {} output: {}".format(i, output[i]))

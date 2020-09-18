import numpy as np
import pandas as pd


class Neuron:

    def __init__(self, input_length, hidden=False):
        self.hidden = hidden
        self.weights = np.random.rand(1, input_length+1)

    def excite(self, inputs):
        return np.dot(self.weights, inputs)

    def activation(self, inputs):
        return np.tanh(self.excite(inputs))

    # def backpropagation(self, input, grad_output):


class Layer:

    def __init__(self, n_of_neurons, input_length, hidden=False):
        self.n_of_neurons = n_of_neurons
        self.neurons = np.empty(n_of_neurons)
        for i in range(n_of_neurons):
            self.neurons[i] = Neuron(input_length, hidden)

    def propagate(self, weights):
        return np.dot(weights, np.array(self.neurons))


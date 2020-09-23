import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class Network:

    # -n_of_inputs: dimension de un input
    # -hidden_layers: array indicando cuantas neurona tiene cada layer
    #                 ej: [3, 2] son 2 capaz con 3 y 2 neuronas respectivamente
    # -n_of_outputs: neuronas de la ultima capa
    def __init__(self, n_of_inputs, hidden_layers, n_of_outputs):
        self.n_of_inputs = n_of_inputs
        self.hidden_layers = hidden_layers
        self.n_of_outputs = n_of_outputs

        # Array con la cantidad de neuronas de cada capa
        neurons_of_layer = [n_of_inputs] + hidden_layers + [n_of_outputs]

        weights = []
        biases = []
        derivatives = []
        deltas = []
        for i in range(len(neurons_of_layer) - 1):
            w = np.random.rand(neurons_of_layer[i], neurons_of_layer[i + 1])
            b = np.random.rand(neurons_of_layer[i+1], 1)
            d = np.zeros((neurons_of_layer[i], neurons_of_layer[i + 1]))
            deltas_i = np.zeros((neurons_of_layer[i+1], 1))
            deltas.append(deltas_i)
            derivatives.append(d)
            weights.append(w)
            biases.append(b)
        self.weights = weights
        self.biases = biases
        self.derivatives = derivatives
        self.deltas = deltas

        activations = []
        for i in range(len(neurons_of_layer)):
            a = np.zeros(neurons_of_layer[i])
            activations.append(a)
        self.activations = activations

        print("weights: {}".format(weights))
        print("biases: {}".format(biases))
        print("derivatives: {}".format(derivatives))
        print("deltas: {}".format(deltas))
        print("activations: {}".format(activations))

    def predict(self, input_):

        self.activations[0] = input_

        for i, w in enumerate(self.weights):
            x = np.dot(w.T, input_) + self.biases[i].T
            x = x.reshape(x.shape[1])
            input_ = sigmoid(x)
            self.activations[i+1] = input_

        return self.activations[-1]

    def train(self, inputs, outputs, epochs, eta):

        for i in range(epochs):
            for j, input_ in enumerate(inputs):
                predicted_output = self.predict(input_)

                error = outputs[j] - predicted_output

                self.back_propagate(error)

                self.update_weights(eta)
            print("Epoch: {}".format(i))

    def back_propagate(self, error):
        for i in reversed(range(len(self.derivatives))):
            output = self.activations[i+1]

            delta = sigmoid_derivative(output) * error
            delta = delta.reshape(delta.shape[0], -1).T

            inputs = self.activations[i]
            inputs = inputs.reshape(inputs.shape[0], -1)

            self.derivatives[i] = np.dot(inputs, delta)

            error = np.dot(delta, self.weights[i].T)
            error = error.reshape(error.shape[1])

    def update_weights(self, eta):

        for i in range(len(self.weights)):
            self.weights[i] += eta * self.derivatives[i]
            self.biases[i] += eta * self.deltas[i]


nn = Network(2, [3], 1)

input_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_test = np.array([0, 1, 1, 0])

nn.train(input_test, output_test, 10000, 0.5)

for inp in range(len(input_test)):
    output = nn.predict(input_test[inp])
    print("xor [{} , {}] = {}".format(input_test[inp][0], input_test[inp][1], output))
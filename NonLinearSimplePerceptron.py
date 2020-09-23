import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera


def g(h, b):
    return 1 / (1 + np.exp(-2 * b * h))


def g_d(h, b):
    aux_g = g(h, b)
    return 2 * b * aux_g * (1 - aux_g)


class NonLinearSimplePerceptron:
    def __init__(self):
        self.weights = None
        self.weights_history = None
        self.errors_history = np.array([])
        self.epochs = 0

    def plot(self):
        if not hasattr(self, 'weights'):
            print('The model is not trained yet!')
            return

        fig, axes = plt.subplots()
        x = []
        y = []
        print("Epochs:", self.epochs)
        plt.grid(True)
        length = len(self.errors_history)
        camera = Camera(fig)

        for i in range(length):
            x = np.append(x, [i + 1])
            y = np.append(y, [self.errors_history[i]])
            plt.plot(x, y, 'k')
            camera.snap()
        #     handles = [Line2D(range(1), range(1), marker='o', markerfacecolor="red", color='white', label='1'),
        #                Line2D([0], [0], marker='o', markerfacecolor="blue", color='white', label='-1')]
        # plt.legend(handles=handles, loc='lower right')
        animation = camera.animate(interval=length * 0.01, repeat=False)
        plt.show()

    def fit(self, initial_weights, learn_factor,beta, entries, expected_outputs, precision, limit):

        n_samples = entries.shape[0]

        self.epochs = 0
        # self.weights = np.random.random_sample(n_features + 1) * 2 - 1
        self.weights = initial_weights
        self.weights_history = [self.weights]
        # Add column of 1s
        x = np.concatenate([entries, np.ones((n_samples, 1))], axis=1)
        #count = 0
        for i in range(limit):
            # if count >= recalculation:
            #   self.weights = np.random.random_sample(n_features + 1) * 2 - 1
            #  count = 0
            error = 0
            for j in range(n_samples):
                h = np.dot(self.weights, x[j, :])
                theta = g(h, beta)
                delta_w = learn_factor * (expected_outputs[j] - theta) * g_d(h, beta) * x[j, :]
                self.weights += delta_w
                error += np.power(expected_outputs[j] - theta, 2) / 2

                # self.weights_history = np.concatenate((self.weights_history, [self.weights]))
            # self.errors_history = np.append(self.errors_history, [error])
            self.epochs += 1
            # count += 1
            if i == limit - 1:
                print("Epochs: ", self.epochs)
                print("Train error: ", error)
                return self.weights, error, self.epochs
                break

    def predict(self, entries, expected_outputs,beta):
        if not hasattr(self, 'weights'):
            print('The model is not trained yet!')
            return
        n_samples = entries.shape[0]
        # Add column of 1s
        x = np.concatenate([entries, np.ones((n_samples, 1))], axis=1)
        outputs = np.array([])
        error = 0
        for i in range(len(x)):
            h = np.dot(self.weights, x[i])
            theta = g(h, beta)
            error += np.power(expected_outputs[i] - theta, 2) / 2
            outputs = np.append(outputs, theta)

        return outputs, error

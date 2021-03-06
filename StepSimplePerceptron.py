import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from matplotlib.lines import Line2D


class StepSimplePerceptron:
    def __init__(self):

        self.weights = None
        self.weights_history = None
        self.epochs = 0
        self.limit = 0

    def fit(self, learn_factor, entries, expected_outputs, limit):
        self.limit = limit
        n_samples = entries.shape[0]
        n_features = entries.shape[1]

        self.weights = np.random.random_sample(n_features + 1) * 2 - 1
        self.weights_history = [self.weights]
        # Add column of 1s
        x = np.concatenate([entries, np.ones((n_samples, 1))], axis=1)

        for i in range(limit):
            error = 0
            for j in range(n_samples):

                if expected_outputs[j] * np.dot(self.weights, x[j, :]) <= 0:
                    self.weights += 2 * learn_factor * expected_outputs[j] * x[j, :]
                    error += 1
                self.weights_history = np.concatenate((self.weights_history, [self.weights]))

            self.epochs += 1
            if error == 0 or i == limit - 1:
                print("Epochs: ", i+1)
                print("Error: ", error)
                break

        self.plot(entries, expected_outputs)

    def predict(self, entries):
        if not hasattr(self, 'weights'):
            print('The model is not trained yet!')
            return

        n_samples = entries.shape[0]
        # Add column of 1s
        x = np.concatenate([entries, np.ones((n_samples, 1))], axis=1)
        y = np.matmul(x, self.weights)
        y = np.vectorize(lambda val: 1 if val > 0 else -1)(y)

        return y

    def plot(self, entries, expected_outputs):
        if not hasattr(self, 'weights'):
            print('The model is not trained yet!')
            return
        n_features = entries.shape[1]
        if n_features != 2:
            print('n_features must be 2')
            return

        fig, axes = plt.subplots()

        plt.grid(True)

        camera = Camera(fig)
        length = len(self.weights_history)
        entry_len = len(entries)
        for e in range(length):

            weights = self.weights_history[e]
            for entry, target in zip(entries, expected_outputs):
                plt.plot(entry[0], entry[1], 'ro' if (target == 1.0) else 'bo', label='%i' % target)

            y = np.array([])
            x = np.array([])
            slope = -(weights[2] / weights[1]) / (weights[2] / weights[0])
            intercept = -weights[2] / weights[1]
            k = 0

            delta = np.array([-2, 2])

            for xi in delta:
                y1 = ((slope * xi) + intercept)
                if delta[0] <= y1 <= delta[1]:

                    x = np.append(x, [xi])
                    y = np.append(y, [y1])
                    k += 1
                    if k == 2:
                        break
            if k < 2:
                for yi in delta:
                    x1 = (yi - intercept) / slope
                    if delta[0] <= x1 <= delta[1]:

                        x = np.append(x, [x1])
                        y = np.append(y, [yi])
                        k += 1
                        if k == 2:
                            break
            iteration = int(e / entry_len)
            plt.text(0.5, 1.01, "Epochs: %i" % iteration+"/%i"%self.limit, horizontalalignment='center',
                     verticalalignment='bottom',
                     transform=axes.transAxes)

            plt.plot(x, y, 'k')

            camera.snap()
            handles = [Line2D(range(1), range(1), marker='o', markerfacecolor="red", color='white', label='1'),
                       Line2D([0], [0], marker='o', markerfacecolor="blue", color='white', label='-1')]
        plt.legend(handles=handles, loc='lower right')
        animation = camera.animate(interval=1, repeat=False)
        plt.show()

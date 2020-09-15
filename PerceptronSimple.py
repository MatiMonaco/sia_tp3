import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from matplotlib.lines import Line2D


class Perceptron:
    def __init__(self, title):
        self.title = title
        self.weights = None
        self.weights_history = None
        self.epochs = 0

    def fit(self, learn_factor, entries, expected_outputs, n_iter=150):

        n_samples = entries.shape[0]
        n_features = entries.shape[1]

        self.weights = np.random.random_sample(n_features + 1) * 2 - 1
        self.weights_history = [self.weights]
        # Add column of 1s
        x = np.concatenate([entries, np.ones((n_samples, 1))], axis=1)

        for i in range(n_iter):
            error = 0
            for j in range(n_samples):

                if expected_outputs[j] * np.dot(self.weights, x[j, :]) <= 0:
                    self.weights += 2 * learn_factor * expected_outputs[j] * x[j, :]
                    error += 1
                self.weights_history = np.concatenate((self.weights_history, [self.weights]))

            self.epochs += 1
            if error == 0:
                print("error:",error)
                break

        print("Iterations:%i" % self.epochs)
        self.plot(entries)

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

    def plot(self, entries):
        if not hasattr(self, 'weights'):
            print('The model is not trained yet!')
            return
        n_features = entries.shape[1]
        if n_features != 2:
            print('n_features must be 2')
            return
        expected_outputs = self.predict(entries)

        fig = plt.figure()
        plt.title(self.title)

        plt.grid(True)

        camera = Camera(fig)

        for e in range(len(self.weights_history)):

            weights = self.weights_history[e]

            for entry, target in zip(entries, expected_outputs):
                plt.plot(entry[0], entry[1], 'ro' if (target == 1.0) else 'bo', label='%i' % target)

            y = np.array([])
            x = np.array([])
            slope = -(weights[0] / weights[2]) / (weights[0] / weights[1])
            intercept = -weights[0] / weights[2]
            k = 0

            delta = np.array([-1.5, 1.5])

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
            plt.plot(x, y, 'k')

            camera.snap()
            handles = [Line2D([0], [0], color='r', label='1'), Line2D([0], [0], color='b', label='-1')]
            plt.legend(handles=handles, loc='lower right')
        animation = camera.animate(interval=100, repeat=False)
        plt.show()
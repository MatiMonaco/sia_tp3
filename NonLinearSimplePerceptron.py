import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from matplotlib.lines import Line2D


def g(h, b):
    return np.tanh(b * h)


def g_d(h, b):
    return b * (1 - np.power(g(h, b), 2))


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

    def fit(self, learn_factor, beta, entries, expected_outputs,precision,limit, recalculation, max_error):

        print(limit)
        n_samples = entries.shape[0]
        n_features = entries.shape[1]

        self.weights = np.random.random_sample(n_features + 1) * 2 - 1
        self.weights_history = [self.weights]
        # Add column of 1s
        x = np.concatenate([entries, np.ones((n_samples, 1))], axis=1)
        count = 0
        for i in range(limit):
            print("i: ", i)
            if count >= recalculation:
                self.weights = np.random.random_sample(n_features + 1) * 2 - 1
                count = 0
            error = 0
            for j in range(n_samples):
                h = np.dot(self.weights, x[j, :])
                theta = g(h, beta)
                delta_w = learn_factor * (expected_outputs[j] - theta) * g_d(h, beta) * x[j, :]

                self.weights += delta_w
                # error += 1

                self.weights_history = np.concatenate((self.weights_history, [self.weights]))
                error += np.power(expected_outputs[j] - theta, 2) / 2
                count+=1
            self.errors_history = np.append(self.errors_history, [error])
            print("error:", error)
            self.epochs += 1

            if error <= precision:
                print("Epoch:",self.epochs)
                self.plot()
                return True
                break
            elif i == limit -1:
                self.plot()
                return False



            # fig, axes = plt.subplots();
            # axes.set_xlabel('Feature 1')
            # axes.set_ylabel('Feature 2')
            #
            # axes.scatter(entries[:,0], entries[:,1],'ro' if (expected_outputs == 1.0) else 'bo', label='%i' % target)
            # plotDecisionBoundary(self.weights, 2, axes)


# define function to map higher order polynomial features
def mapFeature(X1, X2, degree):
    res = np.ones(X1.shape[0])
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            res = np.column_stack((res, (X1 ** (i - j)) * (X2 ** j)))

    return res


# define a function to plot the decision boundary
def plotDecisionBoundary(theta, degree, axes):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    U, V = np.meshgrid(u, v)
    # convert U, V to vectors for calculating additional features
    # using vectorized implementation
    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(u) * len(v)))

    X_poly = mapFeature(U, V, degree)
    Z = X_poly.dot(theta)

    # reshape U, V, Z back to matrix
    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))

    cs = axes.contour(U, V, Z, levels=[0], cmap="Greys_r")
    axes.legend(labels=['good', 'faulty', 'Decision Boundary'])
    return cs



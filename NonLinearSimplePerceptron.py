import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from matplotlib.lines import Line2D


def g(h, b):
    return np.vectorize(lambda val: np.tanh(val))(h)


def g_d(h, b):
    return np.vectorize(lambda val: b * (1 - np.power(g(val), 2)))(h)


class NonLinearSimplePerceptron:
    def __init__(self, title):
        self.title = title
        self.weights = None
        self.weights_history = None
        self.epochs = 0

    def fit(self, learn_factor, beta, entries, expected_outputs):
        limit = int(1 / learn_factor)
        n_samples = entries.shape[0]
        n_features = entries.shape[1]

        self.weights = np.random.random_sample(n_features + 1) * 2 - 1
        self.weights_history = [self.weights]
        # Add column of 1s
        x = np.concatenate([entries, np.ones((n_samples, 1))], axis=1)

        for i in range(limit):
            error = 0
            for j in range(n_samples):
                h = np.dot(self.weights, x[j, :])
                self.weights += learn_factor * (expected_outputs[j] - g(h, beta)) * g_d(h, beta) * x[j, :]
                # error += 1
                self.weights_history = np.concatenate((self.weights_history, [self.weights]))

            self.epochs += 1
            if error == 0:
                print(error)
                break
        print("Iterations:%i" % self.epochs)
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

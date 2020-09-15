import numpy as np


class PerceptronSimple:
    def __init__(self, n):
        self.n = n
        self.weights = np.random.random_sample(n) * 2 - 1
        self.output = None
        self.entries = None
        self.threshold = None

    def propagation(self, entries, threshold):
        self.output = 1 if (self.weights.dot(entries) >= threshold) else -1
        self.entries = entries

    def update(self, alpha, expected_output):
        for i in range(0, self.n):
            self.weights[i] = self.weights[i] + alpha * (expected_output - self.output) * self.entries[i]

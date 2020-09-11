import numpy as np


class PerceptronSimple:
    def __init__(self, n):
        self.n = n

        self.weights = np.random.random_sample(2)*2-1

        self.output = None
        self.entries = None

    def propagation(self, entries, threshold):
        self.output = 1 if (self.weights.dot(entries) >= threshold) else -1
        self.entries = entries

    def update(self, alpha, expected_output):
        print("antes: ",self.weights)
        print("expected_output: ",expected_output)
        print("output: ",self.output)
        for i in range(0, self.n):
            print("entries[",i,"]: ",self.entries[i])
            self.weights[i] = self.weights[i] + alpha * (expected_output - self.output) * self.entries[i]

        print("updated: ", self.weights)

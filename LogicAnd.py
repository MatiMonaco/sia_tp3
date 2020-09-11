from PerceptronSimple import PerceptronSimple
import matplotlib.pyplot as plt
import numpy as np

entries = np.array(([-1, 1], [1, -1], [-1, -1], [1, 1]))
entries_len = len(entries)
expected_output = np.array(([-1, -1, -1, 1]))

and_perceptron = PerceptronSimple(2)

alpha = 0.5
threshold = 0
limit = 100

weights_history = [and_perceptron.weights]
print(and_perceptron.weights)
for i in range(0, limit):
    for j in range(0, entries_len):
        and_perceptron.propagation(entries[j, 0:2], threshold)
        and_perceptron.update(alpha, expected_output[j])
        weights_history = np.concatenate((weights_history, [and_perceptron.weights]))

plt.plot(weights_history[:, 0], 'k')
plt.plot(weights_history[:, 1], 'r')
plt.show()

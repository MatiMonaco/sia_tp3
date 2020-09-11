from PerceptronSimple import PerceptronSimple
import matplotlib.pyplot as plt
import numpy as np

entries = np.array(([-1, 1], [1, -1], [-1, -1], [1, 1]))
entries_len = len(entries)
expected_output = np.array(([1, 1, -1, -1]))

xor_perceptron = PerceptronSimple(2)

alpha = 0.005
threshold = 0
limit = int(1 / alpha)
error = 0
min_error = entries_len * 2
min_weight = np.zeros(entries_len)
weights_history = [xor_perceptron.weights]
print(xor_perceptron.weights)
i = 0
count = 0
while min_error > 0 and i < limit:
    if count > 100*entries_len:
        count = 0
        xor_perceptron.weights = 2*np.random.random_sample(2)-1
    error = 0
    for j in range(0, entries_len):
        xor_perceptron.propagation(entries[j, 0:2], threshold)
        xor_perceptron.update(alpha, expected_output[j])
        weights_history = np.concatenate((weights_history, [xor_perceptron.weights]))
        error += 1 if (xor_perceptron.output != expected_output[j]) else 0
    if error < min_error:
        min_error = error
        min_weight = xor_perceptron.weights.copy()
    elif error > min_error:
        xor_perceptron.weights = 2*np.random.random_sample(2)-1

    i += 1
    count += 1


xor_perceptron.weights = min_weight
for i in range(0, entries_len):
    xor_perceptron.propagation(entries[i, 0:2], threshold)
    print("Entry: ", entries[i, 0:2])
    print("Expected Output: ",expected_output[i])
    print("Output: ", xor_perceptron.output)

plt.plot(weights_history[:, 0], 'k')
plt.plot(weights_history[:, 1], 'r')
plt.show()
print(weights_history)

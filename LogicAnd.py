from PerceptronSimple import PerceptronSimple
import matplotlib.pyplot as plt
import numpy as np

entries = np.array(([-1, 1], [1, -1], [-1, -1], [1, 1]))
entries_len = len(entries)
expected_output = np.array(([-1, -1, -1, 1]))

and_perceptron = PerceptronSimple(2)

alpha = 0.02
threshold = 0.2
limit = int(1 / alpha)
error = 0
min_error = entries_len * 2
min_weight = np.zeros(entries_len)
weights_history = [and_perceptron.weights]
print(and_perceptron.weights)
i = 0
count = 0
while min_error > 0 and i < limit:
    if count > 100*entries_len:
        count = 0
        and_perceptron.weights = 2*np.random.random_sample(2)-1
    error = 0
    for j in range(0, entries_len):
        and_perceptron.propagation(entries[j, 0:2], threshold)
        and_perceptron.update(alpha, expected_output[j])
        weights_history = np.concatenate((weights_history, [and_perceptron.weights]))
        error += 1 if (and_perceptron.output != expected_output[j]) else 0
        print("error: ", error," output:",and_perceptron.output," exp_output:",expected_output[j])
    if error < min_error:
        min_error = error
        min_weight = and_perceptron.weights.copy()
    elif error > min_error:
        and_perceptron.weights = 2*np.random.random_sample(2)-1

    i += 1
    count += 1


and_perceptron.weights = min_weight
for i in range(0, entries_len):
    and_perceptron.propagation(entries[i, 0:2], threshold)
    print("Entry: ", entries[i, 0:2])
    print("Expected Output: ",expected_output[i])
    print("Output: ", and_perceptron.output)

plt.plot(weights_history[:, 0], 'k')
plt.plot(weights_history[:, 1], 'r')
plt.show()
print(weights_history)

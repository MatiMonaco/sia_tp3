from PerceptronSimple import Perceptron
import matplotlib.pyplot as plt
import numpy as np

entries = np.array(([-1, 1], [1, -1], [-1, -1], [1, 1]))
entries_len = len(entries)
expected_output = np.array(([1, 1, -1, 1]))

and_perceptron = Perceptron('Perceptrón Simple <<Y lógico>>')

and_perceptron.fit(0.01,entries,expected_output)
outputs = and_perceptron.predict(entries)
for i in range(0, entries_len):

    print("Entry: ", entries[i,0:2])
    print("Expected Output: ",expected_output[i])
    print("Output: ", outputs[i])
# alpha = 0.0001
# threshold = 0.2
# limit = int(1 / alpha)
# error = 0
# min_error = entries_len * 2
# min_weight = np.zeros(entries_len)
# weights_history = [and_perceptron.weights]
# print(and_perceptron.weights)
# i = 0
# count = 0
# err = 0
# while min_error > 0 and i < limit:
#     # if count > 100*entries_len:
#     #     count = 0
#     #     and_perceptron.weights = 2*np.random.random_sample(2)-1
#     error = 0
#     for j in range(0, entries_len):
#         and_perceptron.propagation(entries[j, 0:2], threshold)
#         and_perceptron.update(alpha, expected_output[j])
#         weights_history = np.concatenate((weights_history, [and_perceptron.weights]))
#         err = expected_output[j] - and_perceptron.output
#         error += 1 if (err != 0) else 0
#         #threshold = threshold - alpha*np.sign(err);
#
#     if error < min_error:
#         min_error = error
#         min_weight = and_perceptron.weights.copy()
#     i += 1
#     count += 1
#
#
#
#
#
# and_perceptron.weights = min_weight
# for i in range(0, entries_len):
#     and_perceptron.propagation(entries[i, 0:2], threshold)
#     print("Entry: ", entries[i, 0:2])
#     print("Expected Output: ",expected_output[i])
#     print("Output: ", and_perceptron.output)
#
# plt.plot(weights_history[:, 0], 'k')
# plt.plot(weights_history[:, 1], 'r')
# plt.ylabel('Weights')
# plt.xlabel('Iteration')
# plt.title('Logic AND')
# plt.show()
# print(weights_history)

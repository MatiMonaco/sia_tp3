from matplotlib.lines import Line2D

from SimpleLinearPerceptron import SimpleLinearPerceptron
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
import random


def createRandomSortedList(num, start=1, end=100):
    arr = []
    tmp = random.randint(start, end)

    for aux in range(num):

        while tmp in arr:
            tmp = random.randint(start, end)

        arr.append(tmp)

    arr.sort()

    return arr


entries_data = np.genfromtxt("./data/train_set.txt", delimiter=",", skip_header=1)
entries_norm = np.linalg.norm(entries_data)
entries_data = entries_data / entries_norm

z_data = np.genfromtxt("./data/expected_outputs.txt", delimiter=" ", skip_header=1)
z_norm = np.linalg.norm(z_data)
z_data = z_data / z_norm

entries_len = len(entries_data)
n_features = entries_data.shape[1]

slp = SimpleLinearPerceptron()

total_epochs = 250
epoch_step = 25
epochs_array = np.arange(epoch_step, total_epochs + epoch_step, epoch_step)
learn_factor = 0.02
precision = 0.01

test_size = 30
train_size = entries_len - test_size
test_indices = createRandomSortedList(test_size, 0, entries_len-1)
print("test_indices = ", test_indices, " len: ", len(test_indices))
test_set = np.take(entries_data, test_indices, axis=0)
train_set = np.delete(entries_data, test_indices, 0)
test_outputs = np.take(z_data, test_indices, axis=0)
train_outputs = np.delete(z_data, test_indices, 0)

weights = np.random.random_sample(n_features + 1)*2-1
train_set_error_history = np.array([])
test_set_error_history = np.array([])
epochs_history = np.array([])
test_output = []
for epoch in epochs_array:
    new_weights, train_error, epochs = slp.fit(weights, learn_factor, train_set, train_outputs, precision, epoch)
    train_set_error_history = np.append(train_set_error_history, train_error)
    epochs_history = np.append(epochs_history, epochs)
    weights = new_weights
    test_output, test_error = slp.predict(test_set, test_outputs)
    print("Test Error: ", test_error)
    test_set_error_history = np.append(test_set_error_history, test_error)


fig, axes = plt.subplots()
x = []
train_y = []
test_y = []
plt.grid(True)
length = len(epochs_history)
camera = Camera(fig)

for i in range(length):
    x = np.append(x, epochs_history[i])
    train_y = np.append(train_y, [train_set_error_history[i]])
    plt.plot(x, train_y, 'b')
    test_y = np.append(test_y, [test_set_error_history[i]])
    plt.plot(x, test_y, 'r')
    camera.snap()
handles = [Line2D([0], [0], color='blue', label='Train error'),
           Line2D(range(1), range(1), color='red', label='Test error')
           ]
plt.legend(handles=handles, loc='lower right')
plt.title("Test and Train errors")
plt.xlabel("Epochs")
plt.ylabel("Error")
animation = camera.animate(interval=length * 0.7, repeat=False)
plt.show()

for i in range(test_size):
    test_index = test_indices[i]
    t_o = test_output[i]
    e_o = z_data[test_index]
    print("----------------------------------------------------------------")
    print("Entry: ", entries_data[test_index]*entries_norm)
    print("Output: ", t_o*z_norm, " | Expected Output: ", e_o*z_norm)


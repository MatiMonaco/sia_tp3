from matplotlib.lines import Line2D

from SimpleLinearPerceptron import SimpleLinearPerceptron
import numpy as np
from celluloid import Camera
import matplotlib.pyplot as plt
import json

# Load data
entries_data = np.genfromtxt("./data/train_set.txt", delimiter=",", skip_header=1)
entries_norm = np.linalg.norm(entries_data)
entries_data = entries_data / entries_norm

z_data = np.genfromtxt("./data/expected_outputs.txt", delimiter=" ", skip_header=1)
z_norm = np.linalg.norm(z_data)
z_data = z_data / z_norm

entries_len = len(entries_data)
n_features = entries_data.shape[1]

with open('./data/config.json') as json_file:
    data = json.load(json_file)
    for p in data['ej2']:
        print('Total Epochs: ' + p['total_epochs'])
        print('Epoch Step: ' + p['epoch_step'])
        print('Learn Factor: ' + p['learn_factor'])
        print('Cross Validation: ' + p['cross_validation'])
        print('K: ' + p['k'])
        print('')

slp = SimpleLinearPerceptron()
k = int(p['k'])
total_epochs = int(p['total_epochs'])
epoch_step = int(p['epoch_step'])
epochs_array = np.arange(epoch_step, total_epochs + epoch_step, epoch_step)
learn_factor = float(p['learn_factor'])
cross_validation = p['cross_validation']
################################################

# Initialize train and test partitions
test_size = int(entries_len / k)
print(test_size)
train_size = entries_len - test_size
test_indexes = np.arange(0, entries_len)
np.random.shuffle(test_indexes)
test_sets_indexes = np.split(test_indexes, k)
test_sets = np.empty((0, 3), float)
train_sets = np.empty((0, 3), float)
test_outputs = np.array([])
train_outputs = np.array([])

for i in range(k):
    indexes = test_sets_indexes[i]
    test_sets = np.append(test_sets, np.take(entries_data, indexes, 0), axis=0)
    test_outputs = np.append(test_outputs, np.take(z_data, indexes, 0))
    train_sets = np.append(train_sets, np.delete(entries_data, indexes, 0), axis=0)

    train_outputs = np.append(train_outputs, np.delete(z_data, indexes, 0), axis=0)

test_sets = np.split(test_sets, k)

test_outputs = np.split(test_outputs, k)

train_sets = np.split(train_sets, k)

train_outputs = np.split(train_outputs, k)
################################################

# Run neural network
weights = np.random.random_sample(n_features + 1) * 2 - 1
train_set_error_history = np.array([])
test_set_error_history = np.array([])
epochs_history = np.array([])
test_output = []

min_weights = []
min_index = 0
min_test_error = 100000
min_train_error = 0
min_test_output = []
cross_validation_limit = k
if cross_validation == "false":
    cross_validation_limit = 1
for epoch in epochs_array:
    epochs_history = np.append(epochs_history, epoch)
    for i in range(cross_validation_limit):
        print("#######################################")
        print("K = ", i)
        new_weights, train_error, epochs = slp.fit(weights, learn_factor, train_sets[i], train_outputs[i],
                                                   epoch)
        test_output, test_error = slp.predict(test_sets[i], test_outputs[i])
        print("Test Error: ", test_error)
        if test_error < min_test_error:
            min_weights = new_weights
            min_train_error = train_error
            min_test_error = test_error
            min_test_outputs = test_output
            min_index = i

    weights = min_weights
    print("Minimim test error: ", min_test_error)
    train_set_error_history = np.append(train_set_error_history, min_train_error)
    test_set_error_history = np.append(test_set_error_history, min_test_error)

################################################

# Plot results
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
plt.legend(handles=handles, loc='upper right')
plt.title("Test and Train errors")
plt.xlabel("Epochs")
plt.ylabel("Error")
animation = camera.animate(interval=1, repeat=False)
plt.show()
################################################

# Print prediction

e_o = test_outputs[min_index]
train_s = train_sets[min_index]

for i in range(test_size):
    t_o = min_test_outputs[i]

    print("----------------------------------------------------------------")
    print("Entry: ", train_s[i] * entries_norm)
    print("Output: ", t_o * z_norm, " | Expected Output: ", e_o[i] * z_norm)

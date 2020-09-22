from SimpleLinearPerceptron import SimpleLinearPerceptron
import numpy as np

entries_data = np.genfromtxt("./data/train_set.txt", delimiter=",", skip_header=1)
entries_norm = np.linalg.norm(entries_data)
entries_data = entries_data / entries_norm

z_data = np.genfromtxt("./data/expected_outputs.txt", delimiter=" ", skip_header=1)
z_norm = np.linalg.norm(z_data)
z_data = z_data / z_norm

entries_len = len(entries_data)
n_features = entries_data.shape[1]

slp = SimpleLinearPerceptron()

total_epochs = 400
epoch_step = 50
epochs_array = np.arange(epoch_step, total_epochs + epoch_step, epoch_step)

learn_factor = 0.1
precision = 0.01

test_size = 50
train_size = entries_len - test_size
weights = np.random.random_sample(n_features + 1)
for epoch in epochs_array:
    new_weights, error, epochs = slp.fit(weights, learn_factor, entries_data, z_data, precision, epoch)

print(epochs_array)

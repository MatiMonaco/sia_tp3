from NonLinearSimplePerceptron import NonLinearSimplePerceptron
import numpy as np

entries_data = np.genfromtxt("./data/train_set.txt", delimiter=",", skip_header=1)
entries_norm = np.linalg.norm(entries_data)
entries_data = entries_data / entries_norm

z_data = np.genfromtxt("./data/expected_outputs.txt", delimiter=" ", skip_header=1)
z_norm = np.linalg.norm(z_data)
z_data = z_data / z_norm

entries_len = len(entries_data)

slp = NonLinearSimplePerceptron()

beta = 1
alpha = 0.1
precision = 0.1
limit = 400
recalculation_limit = 150
result = slp.fit(alpha, beta, entries_data, z_data,precision, limit, recalculation_limit)

outputs = slp.predict(entries_data,beta)
for i in range(0, entries_len):
    print("Entry: ", entries_data[i, 0:3])
    print("Expected Output: ", z_data[i])
    print("Output: ", outputs[i])
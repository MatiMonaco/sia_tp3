from SimpleLinearPerceptron import SimpleLinearPerceptron
import numpy as np
entries = np.genfromtxt("./data/train_set.txt", delimiter=",", skip_header=1)
z = np.genfromtxt("./data/expected_outputs.txt", delimiter=" ", skip_header=1)

entries2 = np.array(([-1, 1], [1, -1], [-1, -1], [1, 1]))
entries_len = len(entries2)
z2 = np.array(([-1, -1, -1, 1]))
slp = SimpleLinearPerceptron()
result = slp.fit(0.2,entries,z,0.1,100,50,1000)


if result:

    outputs = slp.predict(entries2)
    for i in range(0, entries_len):

        print("Entry: ", entries[i,0:2])
        print("Expected Output: ",z[i])
        print("Output: ", outputs[i])


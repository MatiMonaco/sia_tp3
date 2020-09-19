from NonLinearSimplePerceptron import NonLinearSimplePerceptron
import numpy as np
entries = np.genfromtxt("./data/train_set.txt", delimiter=",", skip_header=1)
z = np.genfromtxt("./data/expected_outputs.txt", delimiter=" ", skip_header=1)

entries2 = np.array(([-1, 1], [1, -1], [-1, -1], [1, 1]))
entries_len = len(entries2)
z2 = np.array(([-1, -1, -1, 1]))
slp = NonLinearSimplePerceptron()
result = slp.fit(0.2,1,entries,z,0.1,100,50,3000000)
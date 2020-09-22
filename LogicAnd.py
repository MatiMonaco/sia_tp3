from StepSimplePerceptron import StepSimplePerceptron
import numpy as np

entries = np.array(([-1, 1], [1, -1], [-1, -1], [1, 1]))
entries_len = len(entries)
expected_output = np.array(([-1, -1, -1, 1]))

and_perceptron = StepSimplePerceptron('Perceptrón Simple <<Y lógico>>')

and_perceptron.fit(0.01,entries,expected_output)
outputs = and_perceptron.predict(entries)
for i in range(0, entries_len):

    print("Entry: ", entries[i,0:2])
    print("Expected Output: ",expected_output[i])
    print("Output: ", outputs[i])

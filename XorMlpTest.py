# XOR TEST:
import numpy as np
from MultiLayerPerceptron import Network

input_xor = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
output_xor = np.array([1, 1, -1, -1])
nn = Network(2, [2], 1)
nn.train(input_xor, output_xor, 5000, 0.1)

for inp in range(len(input_xor)):
    output = nn.predict(input_xor[inp])
    print("xor [{} , {}] = {}".format(input_xor[inp][0], input_xor[inp][1], output))

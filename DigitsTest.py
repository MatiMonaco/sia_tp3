# DIGITS TEST:
import pandas as pd
import numpy as np
from MultiLayerPerceptron import Network

df = pd.read_csv('data/numerosenbits', delimiter="\n", header=None)

data_array = np.array(df)
data_array = data_array.reshape(10, 7)

remove_spaces = lambda x: x.replace(" ", "")

for i in range(len(data_array)):
    for j in range(len(data_array[i])):
        data_array[i][j] = int(remove_spaces(data_array[i][j]), 2)

input_numbers = data_array.astype(float) / 31
# output_numbers = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) / 9
output_numbers = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])


nn = Network(7, [5], 1)
nn.train(input_numbers, output_numbers, 1000, 0.5, True)

for inp in range(len(input_numbers)):
    output = nn.predict(input_numbers[inp])
    print("input: {}  = {}".format(inp, output))
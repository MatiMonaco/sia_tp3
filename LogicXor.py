import json

from StepSimplePerceptron import StepSimplePerceptron

import numpy as np

entries = np.array(([-1, 1], [1, -1], [-1, -1], [1, 1]))
entries_len = len(entries)
expected_output = np.array(([1, 1, -1, -1]))

and_perceptron = StepSimplePerceptron('Perceptrón Simple <<XOR lógico>>')
with open('./data/config.json') as json_file:
    data = json.load(json_file)
    for p in data['ej1']:
        print('Learn Factor: ' + p['learn_factor'])
        print('Limit: ' + p['limit'])
        print('')

limit = int(p['limit'])
learn_factor = float(p['learn_factor'])
and_perceptron.fit(learn_factor,entries,expected_output,limit)
outputs = and_perceptron.predict(entries)
for i in range(0, entries_len):

    print("Entry: ", entries[i,0:2])
    print("Expected Output: ",expected_output[i])
    print("Output: ", outputs[i])
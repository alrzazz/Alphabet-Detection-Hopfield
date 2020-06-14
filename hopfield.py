import numpy as np

def relu(x):
    return 1 if x >= 0 else -1

class hofield():
    def __init__(self, func=relu, neurons=1000):
        self.func = func
        self.neurons_count = neurons
        self.matrix = np.zeros(shape=(neurons, neurons))

    def train(self, data):
        for i in range(data.shape[0]):
            for j in range(i):
                self.matrix[i][j] += (data[i] * data[j])
                self.matrix[j][i] += (data[i] * data[j])

    def predict(self, data):
        changed = True
        res = data
        while changed:
            changed = False
            for row in range(len(res)):
                s = 0
                for i in range(len(res)):
                    s += self.matrix[row][i] * res[i]
                n = self.func(s)
                if n != res[row]:
                    changed = True
                    res[row] = n
        return res
                

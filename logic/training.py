import numpy as np
from logic.graphics import plotErrorEvolution, plotWeightsEvolution, resume

class Perceptron:
    def __init__(self,eta, weight, X, Y, tolerancy):
        self.eta = eta
        self.weight = weight
        self.X = X
        self.Y = Y
        self.tolerancy = tolerancy    
    def functionStep(x):
        return 1 if x >= 0 else 0

weightList = []
errorList=[]

def initialization(data):
    errorList.clear()
    weightList.clear()
    num_columns = data.csv_read.shape[1]
    Y = data.csv_read.iloc[:, -1].to_numpy().reshape(-1, 1)
    X = data.csv_read.iloc[:,0: num_columns-1]
    X.insert(0, None, 1)
    X = X.to_numpy()
    weights = np.array( [round(w, 2) for w in np.random.uniform(-1, 1, num_columns)]).reshape(1, -1)
    #weightList.append( weights.flatten())
    neuron = Perceptron(data.eta,weights, X, Y, data.tolerancy)
    cycle(neuron, data.epoch)

def cycle(neuron, epoch):
    for i in range(epoch):
        training(neuron)
    plotWeightsEvolution(weightList)
    plotErrorEvolution(errorList)
    resume(weightList, epoch, neuron.eta, neuron.tolerancy)

def training(neuron):
    weightList.append(neuron.weight.flatten().tolist())
    U = np.dot(neuron.X, neuron.weight.T)
    yCalculate = np.vectorize(Perceptron.functionStep)(U)
    error = neuron.Y - yCalculate
    normaError = np.linalg.norm(error)
    errorList.append(normaError)
    if neuron.tolerancy >= normaError >= 0 or normaError == 0:
        neuron.weight = neuron.weight
    else: 
        newWeights = neuron.eta * np.dot(error.T, neuron.X)
        neuron.weight = neuron.weight + newWeights  
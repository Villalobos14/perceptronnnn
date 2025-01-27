import numpy as np
from logic.graphics import plotErrorEvolution, plotWeightsEvolution, resume

class Perceptron:
    def __init__(self, eta, weight, X, Y, tolerancy):
        self.eta = eta
        self.weight = weight
        self.X = X
        self.Y = Y
        self.tolerancy = tolerancy    

    def functionStep(x):
        return 1 if x >= 0 else 0

weightList = []
errorList = []

def initialization(data):
    errorList.clear()
    weightList.clear()
    num_columns = data.csv_read.shape[1]

    # Separar características (X) y etiquetas (Y), ignorando la primera columna (índice)
    Y = data.csv_read.iloc[:, -1].to_numpy().reshape(-1, 1)  # Última columna como etiquetas
    X = data.csv_read.iloc[:, 1: num_columns - 1]  # Ignorar la primera columna (índice)

    # Verificar si las etiquetas no son binarias y convertirlas
    unique_labels = np.unique(Y)
    if len(unique_labels) > 2 or not np.all(np.isin(Y, [0, 1])):
        print("Convirtiendo etiquetas a binarias (0 y 1).")
        min_label, max_label = np.min(Y), np.max(Y)
        Y = np.where(Y == min_label, 0, 1)

    # Agregar la columna de sesgo (unos) al inicio de las características
    X = np.hstack((np.ones((X.shape[0], 1)), X.to_numpy()))
    
    # Normalizar las características excepto la columna de sesgo
    X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)

    # Inicializar pesos aleatorios entre -1 y 1
    weights = np.array([round(w, 2) for w in np.random.uniform(-1, 1, X.shape[1])]).reshape(1, -1)
    
    # Crear instancia del Perceptron
    neuron = Perceptron(data.eta, weights, X, Y, data.tolerancy)
    
    # Ejecutar el ciclo de entrenamiento
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

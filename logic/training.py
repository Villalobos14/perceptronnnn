import numpy as np
from logic.graphics import (
    plotErrorEvolution,
    plotWeightsEvolution,
    plotDesiredVsCalculated,
    plotAbsoluteError,
    resume
)

class Perceptron:
    def __init__(self, eta, weight, X, Y, tolerancy):
        self.eta = eta
        self.weight = weight
        self.X = X
        self.Y = Y
        self.tolerancy = tolerancy    

    def functionStep(x):
        return 1 if x >= 0 else 0

# Variables globales para almacenar la evolución
weightList = []
errorList = []

def initialization(data):
    # Limpiamos las listas globales
    errorList.clear()
    weightList.clear()

    num_columns = data.csv_read.shape[1]

    # Toma la última columna como Y
    Y = data.csv_read.iloc[:, -1].to_numpy().reshape(-1, 1)

    # Toma las columnas intermedias (ignorando la primera)
    X = data.csv_read.iloc[:, 1 : num_columns - 1]

    # Si las etiquetas no son {0,1}, se convierten a binario
    unique_labels = np.unique(Y)
    if len(unique_labels) > 2 or not np.all(np.isin(Y, [0, 1])):
        print("Convirtiendo etiquetas a binarias (0 y 1).")
        min_label, max_label = np.min(Y), np.max(Y)
        Y = np.where(Y == min_label, 0, 1)

    # Agregamos columna de 1's para el sesgo
    X = np.hstack((np.ones((X.shape[0], 1)), X.to_numpy()))

    # Normalización (excepto la columna de sesgo)
    X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)

    # Inicializar pesos aleatorios
    weights = np.array([
        round(w, 2) for w in np.random.uniform(-1, 1, X.shape[1])
    ]).reshape(1, -1)

    # Crear instancia del perceptron
    neuron = Perceptron(data.eta, weights, X, Y, data.tolerancy)

    # Ejecutar entrenamiento
    cycle(neuron, data.epoch)

def cycle(neuron, epoch):
    for i in range(epoch):
        training(neuron)

    # Graficar evolución de pesos
    plotWeightsEvolution(weightList)
    # Graficar evolución del error (norma)
    plotErrorEvolution(errorList)

    # Calcular salida final
    U_final = np.dot(neuron.X, neuron.weight.T)
    y_final = np.vectorize(Perceptron.functionStep)(U_final)

    # Graficar "Y deseada vs Y calculada"
    plotDesiredVsCalculated(neuron.Y, y_final)

    # Graficar el error absoluto |Y_d - Y_c|
    plotAbsoluteError(neuron.Y, y_final)

    # Mostrar ventana emergente con resumen
    resume(weightList, epoch, neuron.eta, neuron.tolerancy)

def training(neuron):
    # Guardamos pesos actuales
    weightList.append(neuron.weight.flatten().tolist())

    # Calculamos la salida
    U = np.dot(neuron.X, neuron.weight.T)
    yCalculate = np.vectorize(Perceptron.functionStep)(U)

    # Calculamos el error y su norma
    error = neuron.Y - yCalculate
    normaError = np.linalg.norm(error)
    errorList.append(normaError)

    # Verificar tolerancia
    if neuron.tolerancy >= normaError >= 0 or normaError == 0:
        pass
    else:
        newWeights = neuron.eta * np.dot(error.T, neuron.X)
        neuron.weight = neuron.weight + newWeights












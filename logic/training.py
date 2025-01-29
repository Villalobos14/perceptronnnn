import numpy as np
from logic.graphics import (
    plotErrorEvolution,
    plotWeightsEvolution,
    plotDesiredVsCalculated,
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

# Variables globales para almacenar evolución de pesos y de error
weightList = []
errorList = []

def initialization(data):
    # Limpiar las listas cada vez que se inicializa
    errorList.clear()
    weightList.clear()

    num_columns = data.csv_read.shape[1]

    # Separa la última columna como Y
    Y = data.csv_read.iloc[:, -1].to_numpy().reshape(-1, 1)
    # Toma las columnas intermedias (ignorando la primera, si es índice)
    # Ajusta a tus necesidades
    X = data.csv_read.iloc[:, 1 : num_columns - 1]

    # Verificar si Y está en {0,1}; si no, convertir a binario
    unique_labels = np.unique(Y)
    if len(unique_labels) > 2 or not np.all(np.isin(Y, [0, 1])):
        print("Convirtiendo etiquetas a binarias (0 y 1).")
        min_label, max_label = np.min(Y), np.max(Y)
        Y = np.where(Y == min_label, 0, 1)

    # Agregar la columna de sesgo
    X = np.hstack((np.ones((X.shape[0], 1)), X.to_numpy()))

    # Normalizar características (excepto la columna de sesgo)
    X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)

    # Inicializar pesos aleatorios
    weights = np.array([
        round(w, 2) for w in np.random.uniform(-1, 1, X.shape[1])
    ]).reshape(1, -1)

    # Crear el Perceptron
    neuron = Perceptron(
        eta=data.eta,
        weight=weights,
        X=X,
        Y=Y,
        tolerancy=data.tolerancy
    )

    # Iniciar ciclo de entrenamiento
    cycle(neuron, data.epoch)

def cycle(neuron, epoch):
    for i in range(epoch):
        training(neuron)

    # Al terminar las épocas, graficar evolución de pesos y de error
    plotWeightsEvolution(weightList)
    plotErrorEvolution(errorList)

    # == NUEVO: Cálculo final para la gráfica "Deseada vs Calculada" ==
    U_final = np.dot(neuron.X, neuron.weight.T)
    y_final = np.vectorize(Perceptron.functionStep)(U_final)
    plotDesiredVsCalculated(neuron.Y, y_final)

    # Ventana emergente con resumen
    resume(weightList, epoch, neuron.eta, neuron.tolerancy)

def training(neuron):
    # Guardamos pesos actuales
    weightList.append(neuron.weight.flatten().tolist())

    # Calcular la salida actual
    U = np.dot(neuron.X, neuron.weight.T)
    yCalculate = np.vectorize(Perceptron.functionStep)(U)

    # Calcular error y su norma
    error = neuron.Y - yCalculate
    normaError = np.linalg.norm(error)
    errorList.append(normaError)

    # Si el error <= tolerancia, no se ajustan pesos
    if neuron.tolerancy >= normaError >= 0 or normaError == 0:
        # No hacemos nada
        pass
    else:
        # Ajuste de pesos = Perceptron rule
        newWeights = neuron.eta * np.dot(error.T, neuron.X)
        neuron.weight = neuron.weight + newWeights

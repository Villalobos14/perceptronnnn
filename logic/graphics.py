import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

def plotWeightsEvolution(weights, save_filename="WeightsEvolution.jpg"):
    fig, axes = plt.subplots(figsize=(8, 6))
    iterations = list(range(0, len(weights)))
    weights_array = np.array(weights)
    for i in range(weights_array.shape[1]):
        axes.plot(iterations, weights_array[:, i], label=f'w{i}', linestyle="-")
    axes.set_title('Evolución del valor de los pesos')
    axes.set_xlabel('Iteración')
    axes.set_ylabel('Valor del Peso')
    axes.legend()
    plt.savefig(save_filename)
    plt.show()
    plt.close(fig)

def plotErrorEvolution(errors, save_filename="ErrorEvolution.jpg"):
    fig, axes = plt.subplots(figsize=(8, 6))
    iterations = list(range(0, len(errors)))
    axes.plot(iterations, errors, linestyle="-" ,c="red")
    axes.set_title('Evolución de la norma del error')
    axes.set_xlabel('Iteración')
    axes.set_ylabel('Error')
    plt.savefig(save_filename)
    plt.show()
    plt.close(fig)


def resume(weightList, epoch, eta, tolerancy):
    ventana_emergente = tk.Tk()
    ventana_emergente.title("Resultado")
    msg = (f'Pesos iniciales: {weightList[0]} \n Pesos finales: {weightList[-1]} \n Iteraciones: {epoch} \n Tasa de aprendizaje: {eta} \n Tolerancia {tolerancy}')
    etiqueta_mensaje = tk.Label(ventana_emergente, text=msg)
    etiqueta_mensaje.pack(padx=20, pady=20)
    ventana_emergente.mainloop()
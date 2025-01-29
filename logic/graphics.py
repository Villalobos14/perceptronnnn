import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

def plotWeightsEvolution(weights, save_filename="WeightsEvolution.jpg"):
    fig, axes = plt.subplots(figsize=(8, 6))
    iterations = list(range(len(weights)))
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
    iterations = list(range(len(errors)))
    axes.plot(iterations, errors, linestyle="-" ,c="red")
    axes.set_title('Evolución de la norma del error')
    axes.set_xlabel('Iteración')
    axes.set_ylabel('Error')
    plt.savefig(save_filename)
    plt.show()
    plt.close(fig)

def plotDesiredVsCalculated(Y, Y_calc, save_filename="DesiredVsCalculated.jpg"):
    fig, ax = plt.subplots(figsize=(8, 6))
    indices = np.arange(len(Y))
    ax.plot(indices, Y, 'ro', label='Y deseada')
    ax.plot(indices, Y_calc, 'bx', label='Y calculada')
    ax.set_title('Salida Deseada vs Salida Calculada (Final)')
    ax.set_xlabel('Índice de la Muestra')
    ax.set_ylabel('Valor de Salida')
    ax.legend()
    plt.savefig(save_filename)
    plt.show()
    plt.close(fig)

def plotAbsoluteError(Y, Y_calc, save_filename="AbsoluteError.jpg"):
    error_abs = np.abs(Y - Y_calc)
    plt.figure(figsize=(7, 5))
    plt.plot(error_abs, "b-", label="|Y_d - Y_c|", linewidth=1)
    plt.xlabel("Índice de la muestra")
    plt.ylabel("Diferencia Absoluta")
    plt.title("Error Absoluto entre Y_d y Y_c")
    plt.legend()
    plt.grid()
    plt.savefig(save_filename)
    plt.show()
    plt.close()

def resume(weightList, epoch, eta, tolerancy):
    ventana_emergente = tk.Tk()
    ventana_emergente.title("Resultado")
    msg = (
        f'Pesos iniciales: {weightList[0]}\n'
        f'Pesos finales:   {weightList[-1]}\n'
        f'Iteraciones: {epoch}\n'
        f'Tasa de aprendizaje: {eta}\n'
        f'Tolerancia: {tolerancy}'
    )
    etiqueta_mensaje = tk.Label(ventana_emergente, text=msg)
    etiqueta_mensaje.pack(padx=20, pady=20)
    ventana_emergente.mainloop()

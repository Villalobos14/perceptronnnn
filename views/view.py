from tkinter import *
from tkinter import ttk, filedialog
from logic.training import initialization
import pandas as pd
import numpy as np

class DataObject:
    def __init__(self, eta, tolerancy, epoch, csv_read):
        self.eta = eta
        self.tolerancy = tolerancy
        self.epoch = epoch
        self.csv_read = csv_read

    def __str__(self):
        return (
            f"Tasa de aprendizaje: {self.eta},\n"
            f"Tolerancia: {self.tolerancy},\n"
            f"Epoch: {self.epoch},\n"
            f"Data:\n{self.csv_read}"
        )

def read_csv(filename, delimiter):
    try:
        data = pd.read_csv(filename, sep=delimiter, header=None)
        if data.shape[1] > 1:
            return data
    except pd.errors.ParserError:
        pass
    return None

def upload_csv():
    global csv_read
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filename:
        semicolon_data = read_csv(filename, ';')
        comma_data = read_csv(filename, ',')

        if semicolon_data is not None and not semicolon_data.empty:
            csv_read = semicolon_data
        elif comma_data is not None and not comma_data.empty:
            csv_read = comma_data
        else:
            print("Error en el formato")

def save_data():
    eta_value = float(eta.get())
    tolerancy_value = float(tolerancy.get())
    epoch_value = int(epoch.get())

    csv_read_value = csv_read
    data = DataObject(
        eta=eta_value,
        tolerancy=tolerancy_value,
        epoch=epoch_value,
        csv_read=csv_read_value
    )
    initialization(data)        

root = Tk()
root.title("How to train your perceptron")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
for i in range(4):
    root.rowconfigure(i, weight=1)
    for j in range(3):
        mainframe.columnconfigure(j, weight=1)

eta = StringVar()
ttk.Label(mainframe, text="Tasa de aprendizaje:").grid(column=1, row=1, sticky=W)
eta.set("0.01")
ttk.Entry(mainframe, textvariable=eta).grid(column=2, row=1, sticky=W)

tolerancy = StringVar()
ttk.Label(mainframe, text="Tolerancia:").grid(column=1, row=2, sticky=W)
tolerancy.set("0.1")
ttk.Entry(mainframe, textvariable=tolerancy).grid(column=2, row=2, sticky=W)

epoch = StringVar()
ttk.Label(mainframe, text="Número de iteraciones:").grid(column=1, row=3, sticky=W)
epoch.set("1000")
ttk.Entry(mainframe, textvariable=epoch).grid(column=2, row=3, sticky=W)

ttk.Button(mainframe, text="Entrenar", command=save_data).grid(column=3, row=4, sticky=W)
ttk.Button(mainframe, text="Abrir CSV", command=upload_csv).grid(column=1, row=4, sticky=W)

for child in mainframe.winfo_children():
    child.grid_configure(padx=15, pady=5)

root.update()

window_width = root.winfo_reqwidth()
window_height = root.winfo_reqheight()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = int((screen_width - window_width) / 2)
y_coordinate = int((screen_height - window_height) / 2)
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

root.mainloop()

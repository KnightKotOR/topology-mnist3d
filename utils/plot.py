import gudhi as gd
import numpy as np
from matplotlib import pyplot as plt

def plot_persistence(diagrams, labels, digit):
    """
    Выводит пример диаграммы персистентности для цифры digit.
    """
    idx = np.where(labels == digit)[0][0]
    gd.plot_persistence_diagram(diagrams[idx], legend = True)
    plt.show()

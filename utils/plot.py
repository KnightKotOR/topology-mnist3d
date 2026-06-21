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


def plot_all_digits(diagrams, labels):
    _, axes = plt.subplots(2, 5, figsize=(15, 6))

    for digit, ax in enumerate(axes.flat):
        idx = np.where(labels == digit)[0][0]

        gd.plot_persistence_diagram(
            diagrams[idx],
            axes=ax,
            legend=False
        )

        ax.set_title(f"Digit {digit}")

    plt.tight_layout()
    plt.show()

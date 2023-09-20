import random
import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import var
import config
from modules.self_organizing_map import SelfOrganizingMap


def load_iris_data():
    classes_mapping = {'Iris-setosa': 0,
                       'Iris-versicolor': 0.5, 'Iris-virginica': 1}

    df = pd.read_csv(config.Paths.IRIS_DATA)
    df = df.replace(classes_mapping)
    data_list = df.values.tolist()

    params = np.array([x[0:4] for x in data_list])
    classes = np.array([x[4] for x in data_list])

    return (params, classes_mapping)


def main():

    data, classes_mapping = load_iris_data()

    som = SelfOrganizingMap(4, var.N, var.SIGMA0,
                            var.T_SIGMA, var.ETA0, var.T_ETA)
    som.init_weights(data)
    som.train(data, var.N_ITER)

    views = list(itertools.combinations([0, 1, 2, 3], 2))

    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(14)
    fig.set_figheight(7)
    fig.set_dpi(100)

    axes = axes.flatten()
    for v, ax in zip(views, axes):
        ax.set_xlim((np.min(data[:, v[0]]), np.max(data[:, v[0]])))
        ax.set_ylim((np.min(data[:, v[1]]), np.max(data[:, v[1]])))

    for v, ax in zip(views, axes.flatten()):
        ax.set_title(f"attr. {v[0]} vs attr. {v[1]}")
        ax.scatter(data[:, v[0]], data[:, v[1]],
                   c='dodgerblue', marker='.')
        ax.scatter(som.weights[:, v[0]], som.weights[:, v[1]],
                   c='orangered', s=50, marker='x')

    plt.tight_layout(pad=2)
    plt.show()


if __name__ == '__main__':
    main()

import random

import numpy as np
from matplotlib import pyplot as plt

import vars
import config
from modules.swarm_path_search import SwarmPathSearch
from modules.graph import Graph


def main():

    points = np.array([np.array([random.uniform(0, config.WIDTH),
                      random.uniform(0, config.HEIGHT)]) for i in range(vars.N)])
    distances = np.array([[np.linalg.norm(a - b)
                         for b in points] for a in points])

    g = Graph()
    g.set_from_list(distances)

    rsa = SwarmPathSearch(
        vars.ANT_N, vars.Vf, vars.Df, vars.Wf, vars.dw, vars.wr, vars.Q)
    rsa.set_graph(g)
    rsa.set_vertex_priority(np.ones(vars.N))

    n_iter = 30
    for i in range(n_iter):
        print(f"iterations: {i}/{n_iter}")
        rsa.iterate()

    fig = plt.figure(2, dpi=100, figsize=(12, 7))
    ax: plt.Axes = fig.add_subplot(111)
    ax.set_title('pheromone distribution matrix')

    ax.imshow(rsa.weights)

    plt.show()


if __name__ == "__main__":
    main()

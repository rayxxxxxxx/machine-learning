import sys

from dataclasses import dataclass, field
from typing import List

import random

import numpy as np


sys.path.append('./')
sys.path.append('../')
try:
    from Graph_package.Graph2D import Graph2D
    from Graph_package.MatrixGraph import MatrixGraph
except Exception as e:
    assert (e)


@dataclass
class RSA:
    __Q: float = field(default=1, init=True)
    __alpha: float = field(default=1, init=True)
    __beta: float = field(default=1, init=True)
    __gamma: float = field(default=1, init=True)
    __ph_r: float = field(default=0.5, init=True)
    __pr_c: float = field(default=1, init=True)

    __vertex_priority: np.ndarray = field(default_factory=list, init=False)
    __ph_distr: np.ndarray = field(default_factory=list, init=False)
    __G: Graph2D = field(default=None, init=False)

    @property
    def Q(self):
        return self.__Q

    @property
    def alpha(self):
        return self.__alpha

    @property
    def beta(self):
        return self.__beta

    @property
    def gamma(self):
        return self.__gamma

    @property
    def vertex_priority(self):
        return self.__vertex_priority

    @property
    def ph_r(self):
        return self.__ph_r

    @property
    def pr_c(self):
        return self.__pr_c

    @property
    def G(self):
        return self.__G

    def run(self, iters_num: int = 100):
        for i in range(iters_num):
            self.iterate()

    def iterate(self):
        ph_addition: List = []
        for n in range(self.__G.N):
            w = np.array([self.__ph_distr[n][j]**self.alpha *
                          (0.0 if j == n else (self.Q / self.__G.get(n, j))**self.beta) *
                          self.__vertex_priority[j]**self.gamma for
                          j in range(self.__G.N)])

            ph_addition.append(self.__pr_c *
                               np.array(self.vertex_priority) *
                               (w/w.max()))

        self.__ph_distr = self.__ph_distr*self.ph_r+np.array(ph_addition)
        self.__ph_distr = np.apply_along_axis(np.vectorize(
            lambda x: 0.0 if x < 1.0e-3 else x), 0, self.__ph_distr)

    def reset_pheromone(self):
        self.__ph_distr = np.ones(
            (self.__G.N, self.__G.N))

    def set_graph(self, graph: MatrixGraph):
        self.__G = graph

        self.__ph_distr = np.ones(
            (self.__G.N, self.__G.N))

    def set_vertex_priority(self, priority: np.ndarray):
        self.__vertex_priority = priority

    def get_result(self):
        return self.__ph_distr


def main():
    n = 4

    g = Graph2D()
    g.set_from_list(
        [[random.uniform(0, 1000), random.uniform(0, 1000)] for i in range(n)])

    rsa = RSA(1, 0.1, 2, 1, 0.9, 1)
    rsa.set_graph(g)
    rsa.set_vertex_priority(np.random.randint(1, 10, n))

    print(rsa.get_result())
    rsa.run(100)
    print(rsa.get_result())


if __name__ == '__main__':
    main()

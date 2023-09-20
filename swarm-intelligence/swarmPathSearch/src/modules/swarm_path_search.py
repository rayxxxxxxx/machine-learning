import math
import random

import numpy as np

from modules.graph import Graph


class SwarmPathSearch:
    def __init__(self, ants_num=1, Pf=1, Df=1, Wf=1, dw=1, wr=1, Q=1) -> None:
        self.graph: Graph = Graph()
        self.weights: np.ndarray = None
        self.vertex_priority: np.ndarray = None

        self.n_ants: int = ants_num
        self.Vf: float = Pf
        self.Df: float = Df
        self.Wf: float = Wf
        self.dw: float = dw
        self.wr: float = wr
        self.Q: float = Q

    def set_vertex_priority(self, priorities: np.ndarray) -> None:
        self.vertex_priority = priorities

    def set_graph(self, graph: Graph):
        self.graph = graph
        self.weights = np.ones(graph.edges_copy.shape)

    def run(self, iters_num: int):
        for i in range(iters_num):
            self.iterate()

    def iterate(self):
        delta_weights = np.zeros(self.weights.shape)
        for i in range(self.n_ants):
            path = [random.randint(0, self.graph.N - 1)]
            for j in range(self.graph.N-1):
                available_vertices = [x for x in range(
                    self.graph.N) if x not in path]

                priorities = self._calculate_priorities(
                    path, available_vertices)
                probabilities = priorities / np.sum(priorities)

                self._update_delta_weights(
                    path, available_vertices, probabilities, delta_weights)

                winner_id = self._choose_winner(
                    available_vertices, probabilities)
                path.append(winner_id)

        self.weights = self.wr * self.weights + delta_weights

    def _calculate_priorities(self, path: list, ids: list):
        v = np.array([self.vertex_priority[x] for x in ids])
        d = np.array([self.graph.get_edge(path[-1], x) for x in ids])
        w = np.array([self.weights[path[-1]][x] for x in ids])

        V = np.power(v, self.Vf)
        # V = np.exp(self.Vf*v)
        D = np.power(self.Q/d, self.Df)
        # D = np.exp(-self.Df*d/self.Q)
        W = np.power(w, self.Wf)
        # W = np.exp(self.Wf*w)

        return V * D * W

    def _update_delta_weights(self, path: list, ids: list, probs: np.ndarray, delta_weights: np.ndarray):
        for i, j in enumerate(ids):
            delta_weights[path[-1]][j] += self.dw*probs[i]

    def _choose_winner(self, ids: list, probs: np.ndarray):
        return np.random.choice(a=ids, size=1, replace=False, p=probs)[0]

    def reset_weights(self):
        self.weights = np.ones(self.graph.edges_copy.shape)

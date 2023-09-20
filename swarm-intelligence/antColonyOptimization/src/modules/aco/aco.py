from dataclasses import dataclass
from typing import List

import random

import numpy as np

try:
    from modules.aco.aaco import AACO
    from modules.graph.matrix_graph import MatrixGraph
except Exception as e:
    assert (e)


@dataclass(init=True)
class ACO(AACO):
    def set_graph(self, graph: MatrixGraph) -> None:
        self._G = graph

        self._ph_distr = np.ones((self._G.N, self._G.N))

    def run(self, iters_num: int = 100) -> None:
        for i in range(iters_num):
            self.iterate()

    def iterate(self) -> None:
        # array of list of vertex indices (paths)
        solutions = []
        for a in range(self.ants_number):
            # get next solution and append it to array
            solutions.append(self._generate_solution())
        # add pheromone after solutions generated
        self._update_pheromone(solutions)

    def _generate_solution(self) -> List[int]:
        # push first vertex in path
        solution = [random.randint(0, self._G.N-1)]
        # moving through graph and adding vertices to solution path
        for i in range(self._G.N-1):
            # get probabilities sum from current vertex to all possible, which have not visited yet
            probs_sum = np.sum([self._get_vertex_prob(solution[-1], k)
                               for k in range(self._G.N) if k not in solution])
            # calculate probabilities of of available vertices (which have not visited yet)
            probs = [(k, self._get_vertex_prob(solution[-1], k)/probs_sum)
                     for k in range(self._G.N) if k not in solution]
            # randomly select vertex to go next
            solution.append(self._choose_vertex(probs))
        # close path with first vertex
        solution.append(solution[0])
        return solution

    def _get_vertex_prob(self, i: int, j: int) -> float:
        # get probability depends on pheromone and weight value of edge
        return self.ph_distr[i][j]**self.alpha*(1.0/self._G.get(i, j))**self.beta

    def _choose_vertex(self, prob_list: List[List]) -> int:
        '''
        get cumsum to choose next vertex
        e.g. probs = [(0,0.1), (1,0.5), (2,0.4)], rand = 0.4 ---> 1
                         0.4      
        0   0.1           |          0.6               1
        |----|------------|-----------|----------------|
        '''
        rnd = random.uniform(0, 1)
        cumsum = np.cumsum([d[1] for d in prob_list])
        for i in range(len(cumsum)):
            if rnd <= cumsum[i]:
                return prob_list[i][0]
        return prob_list[-1][0]

    def _update_pheromone(self, solutions: List[List]) -> None:
        l = len(solutions[0])-1
        # matrix of for pheromone to add
        new_ph = np.zeros((self._G.N, self._G.N))
        for s in solutions:
            # length of path of solution
            s_len = np.sum([self._G.get(s[k], s[k+1]) for k in range(l)])
            for k in range(l):
                # add pheromone to corresponding edge
                new_ph[s[k]][s[k+1]] += self._Q/s_len

        self._ph_distr = self._ph_distr * self._ph_r + new_ph

    def rest_pheromone(self) -> None:
        self._ph_distr = np.ones(
            (self._G.N, self._G.N))

    def get_pheromone_distribution(self) -> np.ndarray:
        return self.ph_distr.copy()

    def get_result(self) -> dict:
        result_path = [0]
        result_value = 0.0

        n = len(self.ph_distr)
        for i in range(n-1):
            # get possible ways to go, except visited ones
            ways = [(i, v) for (i, v) in enumerate(
                self._ph_distr[result_path[-1]]) if i not in result_path]
            # append vertex with max pheromone
            result_path.append(max(ways, key=lambda x: x[1])[0])
            result_value += self._G.get(result_path[-2], result_path[-1])

        result_path.append(result_path[0])
        result_value += self._G.get(result_path[-2], result_path[-1])

        return {'path': result_path, 'value': result_value}


def main():
    n = 10
    g = MatrixGraph()
    g.set_from_list(
        [np.random.randint(1, 100, n).tolist() for i in range(n)])

    aco = ACO(n, 1, 5, 3, 0.2)
    aco.set_graph(g)

    aco.run(100)
    print(aco.get_result())

    print('='*100)

    aco.rest_pheromone()
    for i in range(10):
        aco.iterate()
        print(aco.get_result())


if __name__ == '__main__':
    main()

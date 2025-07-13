from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List

import numpy as np

try:
    from modules.graph.matrix_graph import MatrixGraph
except Exception as e:
    assert e


@dataclass()
class AACO(ABC):
    _ants_number: int = field(default=20, init=True)

    _Q: float = field(default=1, init=True)
    _alpha: float = field(default=1, init=True)
    _beta: float = field(default=1, init=True)
    _ph_r: float = field(default=0.65)

    _ph_distr: np.ndarray = field(default=None)
    _G: MatrixGraph = field(default=None)

    @property
    def ants_number(self):
        return self._ants_number

    @property
    def Q(self):
        return self._Q

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def ph_r(self):
        return self._ph_r

    @property
    def ph_distr(self):
        return self._ph_distr

    @property
    def G(self):
        return self._G

    @abstractmethod
    def set_graph(self, graph: MatrixGraph):
        pass

    @abstractmethod
    def _generate_solution(self):
        pass

    @abstractmethod
    def _get_vertex_prob(self, i: int, j: int):
        pass

    @abstractmethod
    def _choose_vertex(self, prob_list: List[List]) -> int:
        pass

    @abstractmethod
    def _update_pheromone(self, solutions: List[List]):
        pass

    @abstractmethod
    def get_result(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

try:
    from .abstract_graph import AGraph
    from .vertex import Vertex2D, distance2D
except Exception as e:
    assert (e)


def get_distances_matrix(verticies: List[Vertex2D]) -> np.ndarray:
    return np.array([[distance2D(a, b) for a in verticies] for b in verticies])


@dataclass(init=True)
class Graph2D(AGraph):
    _V: List[Vertex2D] = field(default_factory=list, init=False)

    @property
    def V(self):
        return self._V

    def set_from_list(self, cooedinates: List | np.ndarray) -> None:
        self._N = len(cooedinates)

        for (i, c) in enumerate(cooedinates):
            self._V.append(Vertex2D(i, c[0], c[1]))

        self._edges = get_distances_matrix(self.V)

    def set_from_excel(self, file: str, sheet_name: str | int = 0, header: int = None, index_col: int = None) -> None:
        data = pd.read_excel(file, sheet_name=sheet_name,
                             header=header, index_col=index_col, engine='openpyxl')
        self._N = len(data.index)

        for (i, c) in enumerate(data.values.tolist()):
            self._V.append(Vertex2D(i, c[0], c[1]))

        self._edges = get_distances_matrix(self.V)

    @property
    def vertex_as_list(self) -> List:
        return [v.as_list for v in self._V]

    @property
    def vertex_as_matrix(self) -> np.matrix:
        return np.matrix(data=[v.as_list for v in self._V])


def main():
    import sys
    for p in sys.path:
        print(p)

    g = Graph2D()
    g.set_from_excel('./test_data_2.xlsx', header=0)
    print(g.V)
    print(g.vertex_as_matrix)
    print(g.vertex_as_list)

    print(get_distances_matrix(g.V))


if __name__ == '__main__':
    main()

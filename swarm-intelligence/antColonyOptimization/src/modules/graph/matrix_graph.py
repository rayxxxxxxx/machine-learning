import sys

from dataclasses import dataclass, field
from typing import List

import numpy as np
from numpy.typing import NDArray
import pandas as pd

sys.path.append('./')
sys.path.append('../')

try:
    from .abstract_graph import AGraph
except Exception as e:
    assert (e)


@dataclass(init=True)
class MatrixGraph(AGraph):

    def set_from_list(self, edges: List | np.ndarray) -> None:
        self._N = len(edges)
        self._edges = edges

    def set_from_excel(self, file: str, sheet_name: str | int = 0, header: int = None, index_col: int = None) -> None:
        data = pd.read_excel(file, sheet_name=sheet_name,
                             header=header, index_col=index_col, engine='openpyxl')
        self._N = len(data.index)
        self._edges = data.values.tolist()


def main():
    g = MatrixGraph()

    g.set_from_list(
        [list(np.random.randint(0, 10, 5)) for i in range(5)])
    print(g.N)
    print(g.edges_copy)

    g.set_from_excel(
        file='test_data.xlsx', header=0, index_col=0)

    print(g.N)
    print(g.edges_copy)

    print(g.get(0, 0))
    g.set(0, 0, 1000000)
    print(g.get(0, 0))

    print(g.get_row(0))
    print(g.get_col(0))

    print(g.as_matrix)
    print(g.as_dataframe)


if __name__ == '__main__':
    main()

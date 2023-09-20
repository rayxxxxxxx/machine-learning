from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

try:
    from .movable_vertex import MovableVertex2D
    from .graph_2d import Graph2D, get_distances_matrix
except Exception as e:
    assert (e)


@dataclass(init=True)
class InteractiveGraph2D(Graph2D):
    _V: List[MovableVertex2D] = field(default_factory=list, init=False)

    def set_from_list(self, cooedinates: List | np.ndarray) -> None:
        self._N = len(cooedinates)

        for (i, c) in enumerate(cooedinates):
            self._V.append(MovableVertex2D(i, c[0], c[1]))

        self._edges = get_distances_matrix(self.V)

    def set_from_excel(self, file: str, sheet_name: str | int = 0, header: int = None, index_col: int = None) -> None:
        data = pd.read_excel(file, sheet_name=sheet_name,
                             header=header, index_col=index_col, engine='openpyxl')
        self._N = len(data.index)

        for (i, c) in enumerate(data.values.tolist()):
            self._V.append(MovableVertex2D(i, c[0], c[1]))

        self._edges = get_distances_matrix(self.V)

    def update_edges(self) -> None:
        self._edges = get_distances_matrix(self.V)


def main():
    ig = InteractiveGraph2D()


if __name__ == '__main__':
    main()

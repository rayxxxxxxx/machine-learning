from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


@dataclass
class AGraph(ABC):
    _N: int = field(default=0, init=False)
    _edges: np.ndarray = field(
        default=None, init=False)

    @property
    def N(self) -> int:
        return self._N

    @property
    def edges_copy(self) -> np.ndarray:
        return self._edges.copy()

    def get(self, i: int, j: int) -> int | float:
        return self._edges[i][j]

    def set(self, i: int, j: int, value: int | float) -> None:
        self._edges[i][j] = value

    def get_row(self, index: int) -> List:
        return self._edges[index]

    def get_col(self, index: int) -> List:
        return [row[index] for row in self._edges]

    @abstractmethod
    def set_from_list(self, edges: List | np.ndarray) -> None:
        pass

    @abstractmethod
    def set_from_excel(self, file: str, sheet_name: str | int = 0, header: int = None, index_col: int = None) -> None:
        pass

    @property
    def as_matrix(self) -> np.matrix:
        return np.matrix(self.edges_copy)

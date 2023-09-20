from typing import List, Tuple

import numpy as np
import pandas as pd


class Graph:
    __N: int = None
    __edges: np.ndarray = None

    @property
    def N(self) -> int:
        return self.__N

    @property
    def edges_copy(self) -> np.ndarray:
        return self.__edges

    def set_edge(self, row, col, val) -> None:
        self.__edges[row][col] = val

    def get_edge(self, row, col) -> float:
        return self.__edges[row][col]

    def get_row(self, index) -> np.ndarray:
        return self.__edges[index]

    def get_col(self, index) -> np.ndarray:
        return self.__edges[:, index]

    def set_from_list(self, edges: List | np.ndarray) -> None:
        self.__N = len(edges)
        self.__edges = np.array(edges)

    def set_from_npy_file(self, file: str, pickle_allow: bool = True) -> None:
        self.__edges = np.load(file, allow_pickle=pickle_allow)
        self.__N = self.__edges.shape[0]

    def set_from_excel(self, file: str, sheet_name: str | int = 0, header: int = None, index_col: int = None) -> None:
        data = pd.read_excel(file, sheet_name=sheet_name,
                             header=header, index_col=index_col, engine='openpyxl')
        self.__N = len(data.index)
        self.__edges = np.array(data)

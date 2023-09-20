from dataclasses import dataclass, field
from abc import ABC
from typing import List, Tuple

import numpy as np


@dataclass()
class AVertex(ABC):
    _ID: int = field(default=0)

    @property
    def ID(self):
        return self._ID


@dataclass(init=True)
class Vertex2D(AVertex):
    __X: int | float = field(default=0, init=True)
    __Y: int | float = field(default=0, init=True)

    @property
    def X(self):
        return self.__X

    @property
    def Y(self):
        return self.__Y

    @property
    def as_list(self):
        return [self.__X, self.__Y]

    @property
    def as_np_array(self):
        return np.array([self.__X, self.__Y])

    def set(self, xy: Tuple):
        self.__X = xy[0]
        self.__Y = xy[1]


def distance2D(v1: Vertex2D, v2: Vertex2D) -> int | float:
    return np.fabs(np.linalg.norm(v1.as_np_array-v2.as_np_array))


def main():
    v1 = Vertex2D(0, 0)
    v2 = Vertex2D(100, 100)

    print(v1.ID)
    print(v1.X)
    print(v1.Y)
    print(v1.as_list)
    print(v1.as_np_array)

    print(distance2D(v1, v2))


if __name__ == '__main__':
    main()

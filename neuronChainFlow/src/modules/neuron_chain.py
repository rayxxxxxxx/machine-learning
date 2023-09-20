import math

import numpy as np


class NeuronChain:
    def __init__(self, n: int, p: float, a: float) -> None:
        self.n = n
        self.p = p
        self.a = a
        self.t = np.zeros(n)
        self.s = np.zeros(n)

    def update(self, x: float | int = 0):
        self.s[0] = self.p*self.s[0]+x

        self.s[1:] = self.p*self.s[1:] + \
            self.s[:self.n-1]*np.exp(-self.t[:self.n-1])

        self.t = self.a*self.t+self.s

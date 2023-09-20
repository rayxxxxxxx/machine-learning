import math
import numpy as np
from numba import njit

import vars


@njit(fastmath=True)
def convolve(X: np.ndarray, W: np.ndarray):
    A = np.zeros((X.shape[0]-1, X.shape[1]-1))
    for i in range(X.shape[0]-1):
        for j in range(X.shape[1]-1):
            A[i][j] = np.sum(X[i:i+2, j:j+2]*W[i][j])
    return A


@njit(fastmath=True)
def update(S: np.ndarray, W: np.ndarray, X: np.ndarray, p: float, a: float, b: float):
    A = np.zeros(S.shape)
    for i in range(S.shape[0]-1):
        for j in range(S.shape[1]-1):
            A[i:i+2, j:j+2] += X[i:i+2, j:j+2]*W[i][j]

    newW = np.zeros(W.shape)
    for i in range(S.shape[0]-1):
        for j in range(S.shape[1]-1):
            newW[i][j] = a*W[i][j]+b*(X[i:i+2, j:j+2]-S[i:i+2, j:j+2])

    return p*S+A, newW


class ConvolutionFadeMemoryNN:
    def __init__(self, mem_shape: tuple, p: float, a: float, b: float) -> None:
        self.__S: np.ndarray = np.zeros(mem_shape)
        self.__W: np.ndarray = np.array([[np.full((2, 2), 1e-6) for i in range((mem_shape[0]-1))]
                                        for j in range(mem_shape[1]-1)])
        # self.__W: np.ndarray = np.array([[np.zeros((2, 2)) for i in range((mem_shape[0]-1))]
        #                                 for j in range(mem_shape[1]-1)])

        self.__p = p
        self.__a = a
        self.__b = b

    @property
    def S(self):
        return self.__S.copy()

    @property
    def W(self):
        return self.__W.copy()

    def update(self, X: np.ndarray):
        self.__S, self.__W = update(
            self.__S, self.__W, X, self.__p, self.__a, self.__b)
        np.clip(self.__S, vars.MIN_S, vars.MAX_S)

    def predict(self, X: np.ndarray):
        return convolve(X, self.__W)

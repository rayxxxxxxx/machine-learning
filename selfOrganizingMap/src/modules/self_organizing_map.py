import math
import random

import numpy as np
from numba import njit


@njit
def find_winner(x: np.ndarray, weights: np.ndarray) -> int:
    return np.argmin(np.array([np.linalg.norm(x-w) for w in weights]))


@njit
def update_weight(x, weights, bmu_id, n_neurons, sigma0, t_sigma, eta0, t_eta, t) -> np.ndarray:
    sigma = sigma0*math.exp(-t*t_sigma)
    eta = eta0*math.exp(-t*t_eta)

    bmu_w = weights[bmu_id]
    d = np.array([np.linalg.norm(w-bmu_w) for w in weights])
    s = np.exp(-np.square(d)/(2*sigma**2))

    return eta * s.reshape((n_neurons, 1))*(x-weights)


class SelfOrganizingMap:
    def __init__(self, n_dim, n_neurons, sigma0, t_sigma, eta0, t_eta) -> None:
        self.n_dim: int = n_dim
        self.n_neurons: int = n_neurons
        self.weights: np.ndarray = None
        self.sigma0: float = sigma0
        self.t_sigma: float = t_sigma
        self.eta0: float = eta0
        self.t_eta: float = t_eta

    def init_weights(self, train_data: np.ndarray):
        min_vals = np.min(train_data, axis=0)
        max_vals = np.max(train_data, axis=0)
        p = []
        for i in range(self.n_dim):
            v = np.random.normal(
                loc=(max_vals[i]+min_vals[i])/2, scale=0.5, size=self.n_neurons)
            p.append(np.clip(v, min_vals[i], max_vals[i]))
        self.weights = np.array(p).transpose()

    def predict(self, x: np.ndarray) -> float:
        return np.array([math.sqrt(np.sum(np.square(x-w))) for w in self.weights])

    def train(self, train_data: np.ndarray, n_epochs: int) -> None:
        for t in range(n_epochs):
            sample = random.choice(train_data)
            bmu_id = self._find_winner(sample)
            self._update_weights(sample, bmu_id, t+1)

    def _find_winner(self, sample: np.ndarray) -> int:
        return find_winner(sample, self.weights)

    def _update_weights(self, sample: np.ndarray, bmu_id: int, t: int) -> None:
        self.weights += update_weight(
            sample,
            self.weights,
            bmu_id,
            self.n_neurons,
            self.sigma0,
            self.t_sigma,
            self.eta0,
            self.t_eta,
            t
        )

from typing import List
import random

import numpy as np


class DNN:
    def __init__(self, n_neurons: int, F: object, dF: object) -> None:
        self.N = n_neurons
        self.w: np.ndarray = np.ones((n_neurons, n_neurons))-np.eye(n_neurons)
        self.a: np.ndarray = np.ones(n_neurons)
        self.k: np.ndarray = np.ones(n_neurons)
        self.b: np.ndarray = np.zeros(n_neurons)
        self.t: np.ndarray = np.zeros(n_neurons)

        self.F: object = np.vectorize(F)
        self.dF: object = np.vectorize(dF)

    def predict(self, x: np.ndarray):
        return np.dot(self.F(x, self.a, self.k, self.b), self.w)

    def train(self, x_train: List[np.ndarray], y_train: List[np.ndarray], x_test: List[np.ndarray], y_test: List[np.ndarray], batches=2, learning_rate=0.01, desired_error=0.001):
        Err = self.get_error(x_test, y_test)
        batch_size = len(x_train)//batches
        while Err > desired_error:
            for i in range(batches):
                batch_start = i*batch_size
                batch_end = (i+1)*batch_size

                a_grads = np.zeros(self.N)
                k_grads = np.zeros(self.N)
                b_grads = np.zeros(self.N)
                t_grads = np.zeros(self.N)

                for x, y_hat in zip(x_train[batch_start:batch_end], y_train[batch_start:batch_end]):
                    self.update(x, y_hat, a_grads, k_grads, b_grads, t_grads)

                self.a -= learning_rate * a_grads
                self.k -= learning_rate * k_grads
                self.b -= learning_rate * b_grads
                self.t -= learning_rate * t_grads

            Err = self.get_error(x_test, y_test)
            x_train, y_train = shuffle_data(x_train, y_train)

            print("Error value: {0:.4E}".format(Err))

    def update(self, x: np.ndarray, y_hat: np.ndarray, a_grads: np.ndarray, k_grads: np.ndarray, b_grads: np.ndarray, t_grads: np.ndarray):
        u = np.dot(self.F(x, np.ones(self.N), self.k, self.b), self.w)
        d = x-self.b

        dE_dy = -(y_hat-u)
        q = np.dot(dE_dy, self.w)

        a_grads += u * q
        k_grads += -self.a * np.square(d) * u * q
        b_grads += self.a * self.k * d * u * q
        t_grads += q

    def get_error(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        return np.mean(np.array([1/self.N*np.sum(np.square(yt-self.predict(xt))) for (xt, yt) in zip(x_test, y_test)]))


def shuffle_data(data: List[np.ndarray], answers: List[np.ndarray]):
    ziped = [(x, y) for x, y in zip(data, answers)]
    random.shuffle(ziped)
    return [x[0] for x in ziped], [x[1] for x in ziped]

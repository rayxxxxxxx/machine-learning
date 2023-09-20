from typing import List
import random

import numpy as np


class DNN:
    def __init__(self, w_shape: tuple, F: object, dF: object) -> None:
        self.w_shape: tuple = w_shape
        self.w: List[np.ndarray] = [np.random.uniform(-1, 1, (w_shape[i], w_shape[i+1]))
                                    for i in range(len(w_shape)-1)]
        self.b: List[np.ndarray] = [np.zeros(n) for n in w_shape[1:]]
        self.F: object = np.vectorize(F)
        self.dF: object = np.vectorize(dF)

    def predict(self, x: np.ndarray):
        y = x
        for i, w in enumerate(self.w):
            y = np.dot(y, w) + self.b[i]
            y = self.F(y)
        return y

    def train(self, x_train: List[np.ndarray], y_train: List[np.ndarray], x_test: List[np.ndarray], y_test: List[np.ndarray], batches=2, learning_rate=0.01, desired_error=0.001):
        Err = self.get_error(x_test, y_test)
        batch_size = len(x_train)//batches
        while Err > desired_error:
            for i in range(batches):
                batch_start = i*batch_size
                batch_end = (i+1)*batch_size

                w_grads = [np.zeros((self.w_shape[i], self.w_shape[i+1]))
                           for i in range(len(self.w_shape)-1)]
                b_grads = [np.zeros(n) for n in self.w_shape[1:]]

                for x, y_hat in zip(x_train[batch_start:batch_end], y_train[batch_start:batch_end]):
                    self.back_propagation(x, y_hat, w_grads, b_grads)

                for i, pair in enumerate(zip(w_grads, b_grads)):
                    self.w[i] -= learning_rate * pair[0]
                    self.b[i] -= learning_rate * pair[1]

            Err = self.get_error(x_test, y_test)
            x_train, y_train = shuffle_data(x_train, y_train)

            print("Error: {0:.4E}".format(Err))

    def back_propagation(self, x: np.ndarray, y_hat: np.ndarray, w_grads: List[np.ndarray], b_grads: List[np.ndarray]):
        Y = [x]
        A = [x]

        y = x
        for i, w in enumerate(self.w):
            y = np.dot(y, w) + self.b[i]
            Y.append(y)
            y = self.F(y)
            A.append(y)

        dE_da = -(y_hat-A[-1])
        da_dy = self.dF(Y[-1])

        w_grads[-1] += (dE_da * da_dy * A[-2]).reshape(self.w[-1].shape)
        b_grads[-1] += dE_da * da_dy

        for L in range(len(self.w)-2, 0, -1):
            dE_da = np.dot(self.w[L+1], dE_da)
            da_dy = self.dF(Y[L])

            w_grads[L] += np.array([d * da_dy * A[L-1] for d in dE_da]
                                   ).reshape(self.w[L-1].shape)
            b_grads[L-1] += dE_da * da_dy

    def get_error(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        return np.mean(np.array([1.0/self.w_shape[-1]*np.sum(np.square(yt-self.predict(xt))) for (xt, yt) in zip(x_test, y_test)]))


def shuffle_data(data: List[np.ndarray], answers: List[np.ndarray]):
    ziped = [(x, y) for x, y in zip(data, answers)]
    random.shuffle(ziped)
    return [x[0] for x in ziped], [x[1] for x in ziped]

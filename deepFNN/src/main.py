import math

import numpy as np
import pandas as pd

import config
from modules.deep_neural_network import DNN
import utils.neural_network_utils as neural_network_utils


def F(x: float):
    return 1/(1+math.exp(-x))


def dF(x: float):
    return 1/(1+math.exp(-x))*(1-1/(1+math.exp(-x)))


def main():

    df: pd.DataFrame = pd.read_csv(
        config.Paths.IRIS_DATA.as_posix(), delimiter=",")

    class_values = {"Iris-setosa": 0.0,
                    "Iris-versicolor": 0.5, "Iris-virginica": 1.0}

    df["class-value"] = pd.Series([class_values[c] for c in df["class"]])
    df = df.sample(frac=1)

    train_part = 0.8
    dlen = df.shape[0]
    slice_index = int(train_part * dlen)

    x_train = np.array(df.iloc[0:slice_index, 0:4].values.tolist())
    y_train = np.array(df.iloc[0:slice_index, -1].values.tolist())
    x_test = np.array(df.iloc[slice_index:, 0:4].values.tolist())
    y_test = np.array(df.iloc[slice_index:, -1].values.tolist())

    layers_shape = (4, 2, 1)
    model = DNN(layers_shape, F, dF)

    model.train(x_train, y_train, x_test, y_test, 5, 0.005, 0.025)
    # neural_network_utils.save_weights(model, 'weights.npy')

    # neural_network_utils.load_weights(model, 'weights.npy')
    print('Iris-setosa: ', model.predict(np.array([5.1, 3.5, 1.4, 0.2])))
    print('Iris-versicolor: ', model.predict(np.array([7.0, 3.2, 4.7, 1.4])))
    print('Iris-virginica: ', model.predict(np.array([6.3, 3.3, 6.0, 2.5])))
    print("Total error: ", model.get_error(x_test, y_test))


if __name__ == '__main__':
    main()

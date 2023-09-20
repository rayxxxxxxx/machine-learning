from pathlib import Path

import numpy as np

import config
from modules.single_layer_neural_network import SLNN


def save_weights(model: SLNN, file_name: str = 'weights.npy'):
    weights_path = Path(config.Paths.WEIGHTS, file_name)
    bias_path = Path(config.Paths.WEIGHTS, f'bias_{file_name}')
    np.save(weights_path, model.w, True)
    np.save(bias_path, model.b, True)


def load_weights(model: SLNN, file_name: str = 'weights.npy'):
    weights_path = Path(config.Paths.WEIGHTS, file_name)
    bias_path = Path(config.Paths.WEIGHTS, f'bias_{file_name}')
    model.w = np.load(weights_path, allow_pickle=True)
    model.b = np.load(bias_path, allow_pickle=True)

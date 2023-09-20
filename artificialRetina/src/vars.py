import math
import numpy as np

N = 48
MEMSHAPE = (N, N)

MIN_S = -1e6
MAX_S = 1e6
P = 0.5
ALPHA = 0.9
BETA = 1e-2

RND_THRESHOLD = 1
MIN_W = -1
MAX_W = 1

NOISE = 0

CELL_SIZE = 15


class colors:
    WHITE = np.array([255, 255, 255])
    BLACK = np.array([0, 0, 0])
    GREY = np.array([128, 128, 128])

    RED = np.array([255, 0, 0])
    GREEN = np.array([0, 255, 0])
    BLUE = np.array([0, 0, 255])

    LIGHTRED = np.array([255, 64, 64])
    LIGHTGREEN = np.array([0, 255, 128])
    LIGHTBLUE = np.array([0, 128, 255])

    DARKRED = np.array([128, 0, 0])
    DARKGREEN = np.array([0, 128, 0])
    DARKBLUE = np.array([0, 0, 128])

    MAGENTA = np.array([255, 0, 255])
    YELLOW = np.array([255, 255, 0])
    CAYN = np.array([0, 255, 255])

    ORANGE = np.array([255, 128, 0])
    ORANGERED = np.array([255, 64, 0])

import numpy as np

# =======================
# pygame
# =======================

FPS = 24

WIDTH = 1240
HEIGHT = 720

VERTEX_R = 20

# =======================
# Algorithm parameters
# =======================

N = 10
A_N = N//2
Q = 1
ALPHA = 8
BETA = 12
PH_R = 0.25

# =======================
# Colors
# =======================


class colors:
    BLACK = np.array([0, 0, 0])
    WHITE = np.array([255, 255, 255])
    GRAY = np.array([128, 128, 128])

    LIGHT_RED = np.array([255, 128, 128])
    RED = np.array([255, 0, 0])
    DARK_RED = np.array([128, 0, 0])

    LIGHT_GREEN = np.array([141, 255, 141])
    GREEN = np.array([0, 200, 0])
    DARK_GREEN = np.array([0, 128, 0])

    LIGHT_ORANGE = np.array([255, 172, 88])
    ORANGE = np.array([255, 128, 0])
    DARK_ORANGE = np.array([230, 116, 0])

    ORANGERED = np.array([227, 66, 52])

    SKY_BLUE = np.array([0, 182, 255])
    BLUE = np.array([0, 0, 255])
    DARK_BLUE = np.array([0, 0, 180])

    LIGHT_MAGENTA = np.array([255, 128, 255])
    MAGENTA = np.array([255, 0, 255])
    DARK_MAGENTA = np.array([170, 0, 170])

    LIGHT_YELLOW = np.array([255, 255, 128])
    YELLOW = np.array([255, 255, 0])
    DARK_YELLOW = np.array([170, 170, 0])

    LIGHT_CYAN = np.array([128, 255, 255])
    CYAN = np.array([0, 255, 255])
    DARK_CYAN = np.array([0, 170, 170])

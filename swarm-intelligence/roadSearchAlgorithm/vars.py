from typing import List, Tuple
import numpy as np

SOME_VAR = 0

# =======================
# pygame
# =======================

FPS = 24

WIDTH = 1240
HEIGHT = 720

VERTEX_R = 20

# =======================
# tkinter
# =======================

TK_WIDTH = 600
TK_HEIGHT = 300

# =======================
# Algorithm parameters
# =======================

N = 30
Q = 1
ALPHA = 0.25
BETA = 1.5
GAMMA = 1
PH_R = 0.9
PR_C = 0.1

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

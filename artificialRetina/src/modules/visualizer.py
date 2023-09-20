import math
import numpy as np

import pygame as pg

from modules.convolution_fade_mem_nn import ConvolutionFadeMemoryNN
from utils import math_utils

import vars


def render_retina(srf: pg.Surface, cfmnn: ConvolutionFadeMemoryNN, dest: tuple = (0, 0)):
    maxs = np.max(np.absolute(cfmnn.S))
    cell_size = vars.CELL_SIZE
    for i in range(vars.N):
        for j in range(vars.N):
            t = math.fabs(cfmnn.S[i][j]/maxs)

            clr = math_utils.clampclr(math_utils.lerp(
                vars.colors.BLACK, vars.colors.WHITE, t))

            x = i*cell_size+dest[0]
            y = j*cell_size+dest[1]

            pg.draw.rect(srf, clr, (x, y, cell_size, cell_size))

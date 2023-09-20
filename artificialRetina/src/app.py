import sys
import glob
from pathlib import Path

import numpy as np
from numba import njit

from PIL import Image
from matplotlib import pyplot as plt

import pygame as pg

from modules.convolution_fade_mem_nn import ConvolutionFadeMemoryNN
from modules.visualizer import render_retina

import config
import vars


def load_images(path: Path) -> list[np.ndarray]:
    files = glob.glob(path.as_posix())
    images = []
    for f in files:
        img = Image.open(f).convert('L')
        img.thumbnail(vars.MEMSHAPE,
                      Image.Resampling.LANCZOS)
        img = np.array(img).transpose().astype('float64')
        images.append(img/np.max(img))
    return images


def recunstruct_mem(cfmnn: ConvolutionFadeMemoryNN):
    r = np.zeros(cfmnn.S.shape)
    for i in range(cfmnn.S.shape[0]-1):
        for j in range(cfmnn.S.shape[1]-1):
            r[i:i+2, j:j+2] += cfmnn.W[i][j]
    return r.transpose()


def app():
    clock = pg.time.Clock()
    screen = pg.display.set_mode((config.WIDTH, config.HEIGHT))

    cfmnn = ConvolutionFadeMemoryNN(
        vars.MEMSHAPE, vars.P, vars.ALPHA, vars.BETA)

    images = load_images(Path('src', 'images', 'pictures', '*'))
    # images = load_images(Path('src', 'images', 'numbers', '*'))

    start_index = 0
    curr_img = start_index

    i = curr_img
    t = 0
    is_paused = False
    is_run = True
    while is_run:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                is_run = False
                continue
            if e.type == pg.KEYDOWN and pg.key.get_pressed()[pg.K_p]:
                is_paused = not is_paused

            if e.type == pg.MOUSEBUTTONDOWN and pg.mouse.get_pressed()[0]:
                i += 1
                curr_img = i % len(images)
            elif e.type == pg.MOUSEBUTTONDOWN and pg.mouse.get_pressed()[2]:
                i -= 1
                curr_img = i % len(images)

        if is_paused:
            continue

        screen.fill(vars.colors.BLACK)

        # pg.draw.circle(screen, vars.colors.WHITE, pg.mouse.get_pos(), 50)

        # img = Image.fromarray((pg.surfarray.array3d(screen))).convert('L')
        # img.thumbnail(vars.MEMSHAPE,
        #               Image.Resampling.LANCZOS)
        # img = np.array(img).astype('float64')

        # cfmnn.update(img)

        cfmnn.update(
            images[curr_img]+np.random.uniform(-vars.NOISE, vars.NOISE, vars.MEMSHAPE))

        render_retina(screen, cfmnn, (config.WIDTH//2 -
                      vars.MEMSHAPE[0]*vars.CELL_SIZE//2, config.HEIGHT//2-vars.MEMSHAPE[1]*vars.CELL_SIZE//2))

        clock.tick(config.FPS)
        pg.display.update()

    pg.quit()


if __name__ == '__main__':
    app()

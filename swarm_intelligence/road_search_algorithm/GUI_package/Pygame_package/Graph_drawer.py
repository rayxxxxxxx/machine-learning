import sys
from typing import List

import math
import numpy as np
import pygame as pg

sys.path.append('./')
try:
    import vars
except Exception as e:
    assert (e)

pg.font.init()


def clamp(x, a, b):
    return a if x < a else (b if x > b else x)


def lerp(a, b, t):
    return (1-t)*a+t*b


def interpolate(t, a, b):
    return int(a+t*(b-a))


def angle(A, B, aspectRatio):
    x = B[0] - A[0]
    y = B[1] - A[1]
    angle = math.atan2(-y, x / aspectRatio)
    return angle


def draw_graph_simpe(srf: pg.surface.Surface, n: int, verticies: List[List[int | float]], edges: List[List[int | float]]):
    ifnt = pg.font.SysFont('arial', 14)
    wfnt = pg.font.SysFont('arial', 12, True)

    for i in range(n):
        for j in range(n):
            if i != j and edges[i][j]:
                pg.draw.line(srf, vars.colors.WHITE,
                             verticies[i], verticies[j], 2)

    for i in range(n):
        pg.draw.circle(srf, vars.colors.GRAY, verticies[i], vars.VERTEX_R)
        pg.draw.circle(srf, vars.colors.BLACK, verticies[i], vars.VERTEX_R-2)

        id_text = ifnt.render(
            f'{i}', True, vars.colors.WHITE)

        srf.blit(id_text, np.array(
            verticies[i])-np.array(id_text.get_rect().size)/2)


def draw_graph(srf: pg.surface.Surface, n: int, verticies: List[List[int | float]], edges: List[List[int | float]], priority=None) -> None:

    min_width = 1.0
    max_width = 10

    clr1 = vars.colors.BLACK
    clr2 = vars.colors.LIGHT_GREEN

    ifnt = pg.font.SysFont('arial', 14)
    wfnt = pg.font.SysFont('arial', 12, True)
    pfnt = pg.font.SysFont('arial', 16, True)

    vradius = vars.VERTEX_R

    max_ph = max([max(x) for x in edges])

    for i in range(n):
        for j in range(n):
            if i != j:
                e = (i, j) if edges[i][j] > edges[j][i] else (j, i)

                t = max(edges[i][j], edges[j][i])/max_ph
                if t < 0.1:
                    continue

                w = interpolate(t, min_width, max_width)
                clr = lerp(clr1, clr2, t)
                clr = lerp(vars.colors.BLACK, clr, t)

                pg.draw.line(srf, clr, verticies[i], verticies[j], w)

    for i in range(n):
        for j in range(i+1, n):
            e = (i, j) if edges[i][j] > edges[j][i] else (j, i)

            t = max(edges[i][j], edges[j][i])/max_ph
            if t < 0.5:
                continue

            clr = lerp(clr1, clr2, t)
            clr = lerp(vars.colors.BLACK, clr, t)

            center = (np.array(verticies[i])+np.array(verticies[j]))/2
            text_srf = wfnt.render(f'{round(edges[e[0]][e[1]],2)}', True,
                                   clr, vars.colors.BLACK)
            text_size = np.array(text_srf.get_rect().size)

            srf.blit(text_srf, center-text_size/2)

    for i in range(n):
        pg.draw.circle(srf, vars.colors.GRAY, verticies[i], vradius)
        pg.draw.circle(srf, vars.colors.BLACK, verticies[i], vradius-2)

        id_text = ifnt.render(f'{i}', True,
                              vars.colors.LIGHT_RED, vars.colors.BLACK)

        srf.blit(id_text, np.array(
            verticies[i])+np.array([vradius/2, vradius/2]))

        prior_text = pfnt.render(
            f'{priority[i]}', True, vars.colors.WHITE)

        srf.blit(prior_text, np.array(
            verticies[i])-np.array(prior_text.get_rect().size)/2)

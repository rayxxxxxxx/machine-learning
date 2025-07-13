import numpy as np
import pygame as pg

from modules.swarm_path_search import SwarmPathSearch

pg.font.init()


def render(srf: pg.Surface, points: np.ndarray, rsa: SwarmPathSearch):
    pid_font = pg.font.SysFont("Ubuntu", 12, True)

    maxw = np.max(rsa.weights)

    for i in range(points.shape[0]):
        for j in range(i):
            t = rsa.weights[i][j]/maxw
            if t > 0.1:
                pg.draw.line(srf, 'white', points[i], points[j], 2)

    for i, p in enumerate(points):
        pg.draw.circle(srf, 'red', p, 12)
        pg.draw.circle(srf, 'black', p, 10)

        id_text = pid_font.render(
            f"{rsa.vertex_priority[i]:.0f}", True, 'yellow', 'black'
        )

        srf.blit(id_text, p - np.array(id_text.get_rect().size) / 2)

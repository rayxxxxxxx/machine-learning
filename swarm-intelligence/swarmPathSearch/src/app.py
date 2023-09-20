import sys
import random

import numpy as np
import pygame as pg

import vars
import config
from modules.graph import Graph
from modules.swarm_path_search import SwarmPathSearch
import modules.swarm_path_search_visualizer as visualizer


def app():
    clock = pg.time.Clock()
    screen = pg.display.set_mode((config.WIDTH, config.HEIGHT))

    points = np.array([np.array([random.randint(0, config.WIDTH),
                      random.randint(0, config.HEIGHT)]) for i in range(vars.N)])
    distances = np.array([[np.linalg.norm(a - b)
                         for b in points] for a in points])

    g = Graph()
    g.set_from_list(distances)

    model = SwarmPathSearch(
        vars.ANT_N, vars.Vf, vars.Df, vars.Wf, vars.dw, vars.wr, vars.Q)
    model.set_graph(g)
    model.set_vertex_priority(np.ones(vars.N))

    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if e.type == pg.KEYDOWN and pg.key.get_pressed()[pg.K_r]:
                model.reset_weights()
            if e.type == pg.KEYDOWN and pg.key.get_pressed()[pg.K_g]:
                points = np.array([np.array([random.uniform(0, config.WIDTH),
                                            random.uniform(0, config.HEIGHT)]) for i in range(vars.N)])
                distances = np.array([[np.linalg.norm(a - b)
                                       for b in points] for a in points])
                g.set_from_list(distances)
                model.set_graph(g)

        screen.fill('black')

        visualizer.render(screen, points, model)
        model.iterate()

        clock.tick(config.FPS)
        pg.display.update()


if __name__ == "__main__":
    app()

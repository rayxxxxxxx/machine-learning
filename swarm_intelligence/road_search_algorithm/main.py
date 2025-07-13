
import sys
import random

import numpy as np
import pygame as pg

import vars

sys.path.append('./')
try:
    from Graph_package.MovableVertex import MovableVertex2D, interact_manager
    from Graph_package.Graph2D import Graph2D, InteractiveGraph2D
    from GUI_package.Pygame_package import Graph_drawer
    from RSA.RoadSearchAlgorithm import RSA
except Exception as e:
    assert (e)


def main():

    MovableVertex2D.R = vars.VERTEX_R

    clock = pg.time.Clock()
    screen: pg.Surface = pg.display.set_mode((vars.WIDTH, vars.HEIGHT))

    N = vars.N
    g = InteractiveGraph2D()
    g.set_from_list(
        [[random.uniform(30, vars.WIDTH-30), random.uniform(30, vars.HEIGHT-30)] for i in range(N)])

    rsa = RSA(vars.Q, vars.ALPHA, vars.BETA,
              vars.GAMMA, vars.PH_R, vars.PR_C)
    rsa.set_graph(g)
    rsa.set_vertex_priority(np.random.randint(1, 100, N))

    interact_manager.curr_v = None
    interact_manager.pressed = False

    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()
                sys.exit()

            dxy = pg.mouse.get_rel()
            interact_manager(e, rsa.G.V, dxy)

            if e.type == pg.KEYDOWN and pg.key.get_pressed()[pg.K_UP]:
                rsa.iterate()

            if e.type == pg.KEYDOWN and pg.key.get_pressed()[pg.K_r]:
                rsa.reset_pheromone()

            if e.type == pg.KEYDOWN and pg.key.get_pressed()[pg.K_p]:
                rsa.set_vertex_priority(np.random.randint(1, 100, N))
                rsa.reset_pheromone()

        screen.fill(vars.colors.BLACK)

        rsa.G.update_edges()

        Graph_drawer.draw_graph(
            screen, N, rsa.G.vertex_as_list, rsa.get_result(), rsa.vertex_priority)

        rsa.iterate()

        clock.tick(vars.FPS)
        pg.display.update()


if __name__ == '__main__':
    main()

try:
    import vars
    from modules import ACO, InteractiveGraph2D, MovableVertex2D, interact_manager, draw_graph

    import pygame as pg

    pg.font.init()
except Exception as e:
    assert (e)


def pygame_app():
    import sys
    import random
    import numpy as np

    MovableVertex2D.R = vars.VERTEX_R

    clock = pg.time.Clock()

    screen: pg.Surface = pg.display.set_mode((vars.WIDTH, vars.HEIGHT))

    fnt = pg.font.Font(None, 24)

    interact_manager.curr_v = None
    interact_manager.pressed = False

    n = vars.N
    g = InteractiveGraph2D()
    g.set_from_list(
        [[random.uniform(50, vars.WIDTH-50), random.uniform(50, vars.HEIGHT-50)] for i in range(n)])

    aco = ACO(vars.A_N, vars.Q, vars.ALPHA, vars.BETA, vars.PH_R)
    aco.set_graph(g)

    i = 0
    is_run = True
    while is_run:

        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()
                sys.exit()

            if e.type == pg.KEYDOWN and pg.key.get_pressed()[pg.K_e]:
                is_run = False

            dxy = pg.mouse.get_rel()
            interact_manager(e, aco.G.V, dxy)

            if e.type == pg.KEYDOWN and pg.key.get_pressed()[pg.K_r]:
                aco._ph_distr = np.ones(
                    (g.N, g.N))
                i = 0

        screen.fill('black')

        aco.G.update_edges()

        draw_graph(
            screen, g.N, g.vertex_as_list, aco.get_pheromone_distribution())
        screen.blit(fnt.render(
            f'Iteration: {i}', True, 'white', 'black'), (10, 10))
        screen.blit(fnt.render(
            f'Result: {round(aco.get_result()["value"])}', True, 'white', 'black'), (10, 40))
        screen.blit(fnt.render(
            f'Path: {" _ ".join([str(x) for x in aco.get_result()["path"]])}', True, 'white', 'black'), (10, 70))

        aco.iterate()
        i += 1

        clock.tick(vars.FPS)
        pg.display.update()

    pass


def main():
    pygame_app()


if __name__ == '__main__':
    main()

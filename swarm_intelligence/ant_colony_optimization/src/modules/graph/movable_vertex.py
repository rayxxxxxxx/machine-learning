from dataclasses import dataclass, field
from typing import List

import pygame as pg


try:
    from .vertex import Vertex2D
except Exception as e:
    print(e)


@dataclass(init=True)
class MovableVertex2D(Vertex2D):
    R: int = field(default=15, init=False)
    __rect: pg.rect.Rect = field(default=None, init=False)

    def __post_init__(self):
        self.__rect = pg.Rect(self.X-self.R, self.Y -
                              self.R, self.R*2, self.R*2)

    @property
    def rect(self):
        return self.__rect.copy()

    def move(self, dxy):
        self.__rect.move_ip(dxy)


def interact_manager(e: pg.event.Event, verticies: List[MovableVertex2D], dxy):
    interact_manager.curr_v: MovableVertex2D

    if e.type == pg.MOUSEBUTTONDOWN and pg.mouse.get_pressed()[0]:
        interact_manager.pressed = True

    if e.type == pg.MOUSEBUTTONUP and not pg.mouse.get_pressed()[0]:
        interact_manager.pressed = False
        interact_manager.curr_v = None

    if interact_manager.curr_v and interact_manager.pressed:
        interact_manager.curr_v.move(dxy)
        interact_manager.curr_v.set((interact_manager.curr_v.X +
                                     dxy[0], interact_manager.curr_v.Y+dxy[1]))
        return

    if interact_manager.pressed:
        vs = [v for v in verticies if v.rect.collidepoint(pg.mouse.get_pos())]
        if vs:
            interact_manager.curr_v = vs[0]
            interact_manager.curr_v.move(dxy)


def main():
    import sys

    MovableVertex2D.R = 50

    mv = MovableVertex2D(0, 100, 100)

    clock = pg.time.Clock()
    screen: pg.Surface = pg.display.set_mode((800, 600))

    vlist = [mv]

    interact_manager.curr_v = None
    interact_manager.pressed = False
    while True:

        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()
                sys.exit()

            dxy = pg.mouse.get_rel()
            interact_manager(e, vlist, dxy)

        screen.fill('black')
        pg.draw.circle(screen, 'green', mv.rect.center, mv.rect.size[0]//2)
        pg.draw.rect(screen, 'red', mv.rect, 2)

        clock.tick(30)
        pg.display.update()


if __name__ == '__main__':
    main()

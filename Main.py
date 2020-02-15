import os

import pygame

from Ball import Ball
from Fluid import Fluid
import numpy as np
from Shape import Shape
from Rect import Rect


def main():

    HEIGTH = Shape.HEIGTH
    WIDTH = Shape.WIDTH
    FORCE_VALUE = 5000
    DELAY_MILISECONDS = 17

    enviroment = Fluid(1.225, np.array([0, 0]))
    ball = Ball(10, (250, 0, 0), 15, enviroment)
    #ball = Rect(10, (250, 0, 0), 200, 10, enviroment)
    ball.position[0] = 200
    #ball.angle = 1
    east_wall = Rect(100, (0, 250, 0), 50, HEIGTH, enviroment, True)
    east_wall.position[0] = WIDTH - 5
    east_wall.position[1] = HEIGTH/2

    west_wall = Rect(100, (0, 250, 0), 50, HEIGTH, enviroment, True)
    west_wall.position[0] = 0
    west_wall.position[1] = HEIGTH/2

    south_wall = Rect(100, (0, 250, 0), WIDTH, 50, enviroment, True)
    south_wall.position[0] = WIDTH/2
    south_wall.position[1] = HEIGTH - 5

    north_wall = Rect(100, (0, 250, 0), WIDTH, 50, enviroment, True)
    north_wall.position[0] = WIDTH/2
    north_wall.position[1] = 0


    square = Rect(1, (0, 250, 0), 15, 15, enviroment, False)
    square.position[0] = 250
    square.position[1] = 250

    sprite_ball = Ball(100, (0, 250, 0), 20, enviroment, True)
    sprite_ball.position[0] = 250
    sprite_ball.position[1] = 400

    enviroment.add(ball)
    enviroment.add(east_wall)
    enviroment.add(west_wall)
    enviroment.add(south_wall)
    enviroment.add(north_wall)
    enviroment.add(square)
    enviroment.add(sprite_ball)

    pygame.init()
    window = pygame.display.set_mode( (WIDTH, HEIGTH))
    pygame.display.set_caption("Koralovo")

    while True:
        pygame.time.delay(DELAY_MILISECONDS)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit(0)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            ball.left(FORCE_VALUE)
        if keys[pygame.K_RIGHT]:
            ball.right(FORCE_VALUE)
        if keys[pygame.K_UP]:
            ball.up(FORCE_VALUE)
        if keys[pygame.K_DOWN]:
            ball.down(FORCE_VALUE)
        for shape in enviroment.shapes:
            if not shape.sprite:
                shape.move()
        enviroment.collide()
        window.fill((0, 0, 250))
        #pygame.draw.circle(window, ball.color, ball.position, ball.radius, 1)
        for i in range(0, len(enviroment.shapes)):
            r = enviroment.shapes[i]
            r.resetForce()
            if isinstance(r, Rect):
                rect, surface = r.draw_pos()
                window.blit(surface, rect)
            elif isinstance(r, Ball):
                color, position, radius = r.draw_pos()
                pygame.draw.circle(window, color, position, radius, 0)
        pygame.display.flip()





if __name__ == '__main__':
    main()



import pygame

from Ball import Ball
from Fluid import Fluid
import numpy as np
from Shape import Shape
from Rect import Rect
from Debug import Debug

def loop(obj, force, env, window):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        obj.left(force)
    if keys[pygame.K_RIGHT]:
        obj.right(force)
    if keys[pygame.K_UP]:
        obj.up(force)
    if keys[pygame.K_DOWN]:
        obj.down(force)
    for shape in env.shapes:
        if not shape.sprite:
            shape.move()
    env.collide()
    window.fill((0, 0, 250))
    # pygame.draw.circle(window, obj.color, obj.position, obj.radius, 1)
    for i in range(0, len(env.shapes)):
        r = env.shapes[i]
        r.resetForce()
        if isinstance(r, Rect):
            rect, surface = r.draw_pos()
            window.blit(surface, rect)
        elif isinstance(r, Ball):
            color, position, radius = r.draw_pos()
            pygame.draw.circle(window, color, position, radius, 0)
    Debug.refresh(window)
    pygame.display.flip()

def main():

    HEIGTH = Shape.HEIGTH
    WIDTH = Shape.WIDTH
    FORCE_VALUE = 500
    DELAY_MILISECONDS = 17
    FRICTION = 0.5

    Shape.SCALING_FACTOR = 10 #TODO CHANGE THIS TO CHANGE SCALING
    #Note - acts weird if scaling is >= 14

    enviroment = Fluid(1.225, np.array([0, 0]))



    #TODO COMMENT AND UNCOMMENT TO SWITCH FROM RECTANGLE TO BALL AND VICE VERSA

    ball = Ball(10, (250, 0, 0), 15, enviroment, FRICTION)
    #ball = Rect(10, (250, 0, 0), 200, 10, enviroment, FRICTION)

    #TODO COMMENT AND UNCOMMENT TO SWITCH FROM RECTANGLE TO BALL AND VICE VERSA


    east_wall = Rect(100, (0, 250, 0), 50, HEIGTH, enviroment, FRICTION, True)
    east_wall.position[0] = WIDTH - 5
    east_wall.position[1] = HEIGTH/2

    west_wall = Rect(100, (0, 250, 0), 50, HEIGTH, enviroment, FRICTION, True)
    west_wall.position[0] = 50
    west_wall.position[1] = HEIGTH/2

    south_wall = Rect(100, (0, 250, 0), WIDTH, 50, enviroment, FRICTION, True)
    south_wall.position[0] = WIDTH/2
    south_wall.position[1] = HEIGTH - 5

    north_wall = Rect(100, (0, 250, 0), WIDTH, 50, enviroment, FRICTION, True)
    north_wall.position[0] = WIDTH/2
    north_wall.position[1] = 50


    square = Rect(1, (0, 250, 0), 100, 20, enviroment, FRICTION, False)
    square.position[0] = 250
    square.position[1] = 250

    sprite_ball = Ball(100, (0, 250, 0), 20, enviroment, FRICTION, True)
    sprite_ball.position[0] = 250
    sprite_ball.position[1] = 400

    enviroment.add(ball)
    enviroment.add(east_wall)
    enviroment.add(west_wall)
    enviroment.add(south_wall)
    enviroment.add(north_wall)
    enviroment.add(square)
    enviroment.add(sprite_ball)

    enviroment.recalibrate_shapes_positions()

    pygame.init()
    window = pygame.display.set_mode( (WIDTH, HEIGTH))
    pygame.display.set_caption("Koralovo")

    run = True

    while True:
        pygame.time.delay(DELAY_MILISECONDS)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    run = not run
                elif event.key == pygame.K_SPACE and not run:
                    loop(ball, FORCE_VALUE, enviroment, window)
        if run:
            loop(ball, FORCE_VALUE, enviroment, window)


if __name__ == '__main__':
    main()


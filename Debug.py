import numpy as np
import pygame

from Shape import Shape


class Debug:
    queue = []

    @staticmethod
    def draw_point(point, color=(250, 250, 250)):
        Debug.queue.append((color, point.astype(int)))

    @staticmethod
    def refresh(window):
        X = np.array([
            [Shape.SCALING_FACTOR, 0, 0],
            [0, Shape.SCALING_FACTOR, 0],
            [0, 0, 1]
        ])
        if window is not None:
            for point in Debug.queue:
                pg = np.array((point[1][0], point[1][1], 1))
                ps = X.dot(pg)
                ps = np.array((ps[0], ps[1]))
                pygame.draw.circle(window, point[0], ps, 10)

        Debug.queue.clear()

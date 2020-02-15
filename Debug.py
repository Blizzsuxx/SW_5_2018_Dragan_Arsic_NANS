import pygame

class Debug:

    window = None

    queue = []

    @staticmethod
    def draw_point(point, color=(250, 250, 250)):
        Debug.queue.append([color, point.astype(int)])

    @staticmethod
    def refresh():
        if Debug.window is not None:
            for point in Debug.queue:
                pygame.draw.circle(Debug.window, point[0], point[1], 10)

        Debug.queue = []
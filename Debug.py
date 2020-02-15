import pygame

class Debug:
    queue = []

    @staticmethod
    def draw_point(point, color=(250, 250, 250)):
        Debug.queue.append([color, point.astype(int)])

    @staticmethod
    def refresh(window):
        if window is not None:
            for point in Debug.queue:
                pygame.draw.circle(window, point[0], point[1], 10)

        Debug.queue = []

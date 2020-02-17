from Shape import Shape
import numpy as np
import pygame

class Rect(Shape):

    def __init__(self, mass, color, a, b, enviroment, friction, sprite=False):
        super().__init__(mass, color, enviroment, friction, sprite)
        self.a = a/Shape.SCALING_FACTOR
        self.b = b/Shape.SCALING_FACTOR
        self.density = mass/(a*b)
        self.drag_constant = 1 / 2 * enviroment.density * Shape.C_CUBE * b**2
        self.inertia = self.mass * (self.a * self.a + self.b * self.b) / 12
        self.pygame_surface = pygame.Surface((a, b))
        self.pygame_surface.set_colorkey((0, 0, 0))
        self.pygame_surface.fill(color)
        if not sprite:
            self.inverse_inertia = 1/self.inertia
        else:
            self.inertia = float("inf")

    def getExtremities(self):
        extremities = []
        extremities.append(self.position + np.array([-self.a / 2, -self.b / 2]))
        extremities.append(self.position + np.array([self.a / 2, -self.b / 2]))
        extremities.append(self.position + np.array([self.a/2, self.b/2]))
        extremities.append(self.position + np.array([-self.a/2, self.b/2]))

        for i in range(len(extremities)):
            x = extremities[i][0]
            y = extremities[i][1]

            x_origin = self.position[0]
            y_origin = self.position[1]
            cs = np.cos(self.angle)
            sn = np.sin(self.angle)
            x_rotated = ((x - x_origin) * cs) - ((-y_origin + y) * sn) + x_origin
            y_rotated = ((-y_origin + y) * cs) + ((x - x_origin) * sn) + y_origin
            extremities[i][0] = x_rotated
            extremities[i][1] = y_rotated


        return np.array(extremities)

    def getNormals(self):
        normals = []
        extremities = self.getExtremities()
        v1 = extremities[0]
        v2 = extremities[1]
        v1 = v1 - v2
        v1 = v1 / np.sqrt(v1.dot(v1))
        normals.append(v1)
        v1 = extremities[1]
        v2 = extremities[2]
        v1 = v1 - v2
        v1 = v1 / np.sqrt(v1.dot(v1))
        normals.append(v1)
        return np.array(normals)


    def getCoallisionEdge(self, normal):

        extremities1 = self.getExtremities()
        index = 0
        max_projection = -float("inf")
        for i in range(len(extremities1)):
            projection = normal.dot(extremities1[i])
            if projection > max_projection:
                max_projection = projection
                index = i
        v = extremities1[index]
        v0 = extremities1[index-1]
        v1 = extremities1[ (index+1) % len(extremities1)]
        l = v - v1
        r = v - v0
        l = l/np.linalg.norm(l)
        r = r/np.linalg.norm(r)

        if r.dot(normal) <= l.dot(normal):
            return v, v0, v
        else:
            return v, v, v1


    def getCenter(self):
        return self.position

    """
    def calculate_force(self, i):
        if i == 0:
            return ((self.force[i] - 1 / 2 * self.sign(i) * self.drag_constant * (
                    self.vector[i] + self.enviroment.velocity[i]) ** 2) / self.mass)
        else:
            return ((self.force[i] + self.mass * 9.81 - 1 / 2 * self.sign(i) * self.drag_constant * (
                    self.vector[i] + self.enviroment.velocity[i]) ** 2) / self.mass)
    """
    
    def draw_pos(self):

        X = np.array([
            [Shape.SCALING_FACTOR, 0, 0],
            [0, Shape.SCALING_FACTOR, 0],
            [0, 0, 1]
        ])
        pg = np.array([self.position[0], self.position[1], 1])
        ps = X.dot(pg)
        ps = np.array((ps[0], ps[1]))

        new_surface = pygame.transform.rotate(self.pygame_surface, -self.angle*180/np.pi)
        #new_surface = self.pygame_surface.copy()

        rect = new_surface.get_rect()
        if np.any(self.position > 2147483647) or np.any(self.position < -2147483647):
            self.position[0] = 100
            self.position[1] = 100
            self.vector[0] = 10
            self.vector[1] = 10
        rect.center = ps

        return rect, new_surface


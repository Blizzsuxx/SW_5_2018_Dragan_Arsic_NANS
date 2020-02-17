from Shape import Shape
from math import pi
import numpy as np

def skalar(n, d):
    s = 0
    for i in range(0,len(n)):
        s += n[i] * d[i]
    return s


class Ball(Shape):

    def __init__(self, mass, color, radius, enviroment, friction, sprite=False):
        super().__init__(mass, color, enviroment, friction, sprite)
        self.radius = radius/Shape.SCALING_FACTOR
        self.density = mass / (radius**2 * np.pi)
        self.drag_constant = 1/2 * enviroment.density * Shape.C_SPHERE * radius**2 * pi
        self.inertia = 1/4 * self.mass * self.radius * self.radius
        if not sprite:
            self.inverse_inertia = 1/self.inertia
        else:
            self.inertia = float("inf")

    def getExtremities(self, normal):
        return [self.position + normal*self.radius, self.position - normal*self.radius]

    def getNormals(self):

        return np.array([[1.0, 0.0]])

    def getBorder(self, other_object):

        center = other_object.getCenter()
        vector = self.getCenter() - center
        vector = vector / np.linalg.norm(vector)

        return vector * self.radius



    def getCoallisionEdge(self, normal):

        v = self.getCenter() + normal*self.radius

        return v, v, self.getCenter()


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

        if np.any(self.position > 2147483647) or np.any(self.position < -2147483647):
            self.position[0] = 100
            self.position[1] = 100
            self.vector[0] = 10
            self.vector[1] = 10

        return self.color, ps.astype(int), int(self.radius*Shape.SCALING_FACTOR)

from Shape import Shape
from math import pi
import numpy as np
import pygame

def skalar(n, d):
    s = 0
    for i in range(0,len(n)):
        s += n[i] * d[i]
    return s


class Ball(Shape):

    def __init__(self, mass, color, radius, enviroment, sprite=False):
        super().__init__(mass, color, enviroment, sprite)
        self.radius = radius
        self.density = mass / (radius**2 * np.pi)
        self.drag_constant = 1/2 * enviroment.density * Shape.C_SPHERE * radius**2 * pi
        self.inertia = 1/4 * self.mass * self.radius * self.radius
        #self.pygame_surface = pygame.Surface((self.radius, self.radius), pygame.SRCALPHA)

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

        v = self.getCenter() - normal*self.radius

        return v, v, self.getCenter()


    def move(self):
        #print(self.position, self.vector, self.force, self.mass)
        pos = self.position - self.radius
        #self.torque = pos[0] * self.force[1] - pos[1] * self.force[0]
        #print("ANGULAR SPEED: ", self.angularspeed)
        self.angularspeed = self.angularspeed + Shape.DELTA_TIME * self.torque / self.inertia
        self.angle = self.angle + Shape.DELTA_TIME * self.angularspeed
        self.angularspeed = 0

        for i in range(len(self.vector)):

            self.vector[i] = self.vector[i] + Shape.DELTA_TIME * self.calculate_force(i)
            temp = Shape.DELTA_TIME * self.vector[i]
            self.position[i] = self.position[i] + temp

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
        if np.any(self.position > 2147483647) or np.any(self.position < -2147483647):
            self.position[0] = 100
            self.position[1] = 100
            self.vector[0] = 10
            self.vector[1] = 10

        return self.color, self.position.astype(int), self.radius

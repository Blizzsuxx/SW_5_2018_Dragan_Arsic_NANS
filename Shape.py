import numpy as np
import pygame



class Shape:
    """
    CONSTANTS
    """
    WIDTH = 500
    HEIGTH = 500



    DELTA_TIME = 0.017

    C_SPHERE = 0.47
    C_HALF_SPHERE = 0.42
    C_CONE = 0.50
    C_CUBE = 1.05
    C_ANGLED_CUBE = 0.80
    C_LONG_CYLINDER = 0.82
    C_SHORT_CYLINDER = 1.15
    C_STREAMLINED_BODY = 0.04
    C_STREAMLINED_HALF_BODY = 0.09

    def __init__(self, mass, color, enviroment, sprite=False):
        self.position = np.array([100.0, 100.0])
        self.force = np.array([0.0, 0.0])
        self.vector = np.array([0.0, 0.0])
        self.mass = mass
        self.color = color
        self.enviroment = enviroment
        self.sprite = sprite
        self.inertia = 0
        self.torque = 0
        self.angle = 0
        self.angularspeed = 0
        self.drag_constant = 0


    def move(self):
        pass

    def getBorder(self, direction):
        pass

    def getCenter(self):
        pass

    def left(self, force):
        self.force[0] -= force

    def right(self, force):
        self.force[0] += force

    def up(self, force):
        self.force[1] -= force

    def down(self, force):
        self.force[1] += force

    def draw_pos(self):
        pass


    def setPosition(self, pos):
        X = np.array([
            [2, 0, self.position[0]],
            [0, 2, self.position[1]],
            [0, 0, 1]
        ])
        invX = np.linalg.inv(X)
        pg = np.array([pos[0], pos[1], 1])
        ps = invX.dot(pg)
        print(self.position, ps)
        self.position[0], self.position[1] = ps[0], ps[1]

    def updateSpeeds(self):
        self.angularspeed = self.angularspeed + Shape.DELTA_TIME * self.torque / self.inertia
        for i in range(len(self.vector)):
            self.vector[i] = self.vector[i] + Shape.DELTA_TIME * self.calculate_force(i)

    def resetForce(self):
        self.torque = 0
        for i in range(len(self.force)):
            self.force[i] = 0


    def sign(self, i):
        vec = self.vector[i]
        if vec == 0:
            force = self.force[i]
            if force == 0:
                return 1
            else:
                return force / abs(force)
        return vec / abs(vec)


    def calculate_force(self, i):
        if i == 0:
            return self.force[i]/self.mass
        else:
            return (self.force[i] + 9.81*self.mass)/self.mass

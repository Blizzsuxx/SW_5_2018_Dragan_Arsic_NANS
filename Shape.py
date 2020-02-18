import numpy as np



class Shape:
    """
    CONSTANTS
    """
    WIDTH = 500
    HEIGTH = 500


    SCALING_FACTOR = 1



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

    def __init__(self, mass, color, enviroment, friction, sprite=False):
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
        self.friction = friction
        self.inverse_inertia = 0

    def move(self):
        self.angularspeed = self.angularspeed + Shape.DELTA_TIME * self.torque / self.inertia
        self.angle = self.angle + Shape.DELTA_TIME * self.angularspeed
        self.angularspeed *= 0.9
        for i in range(len(self.vector)):
            self.vector[i] = self.vector[i] + Shape.DELTA_TIME * self.calculate_force(i)
            temp = Shape.DELTA_TIME * self.vector[i]
            self.position[i] = self.position[i] + temp

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

    def recalibrate_position(self):
        self.setPosition(self.position)


    def add_impulse(self, impulse, normal):
        if self.sprite:
            return
        self.vector += (1 / self.mass) * impulse
        self.angularspeed += self.inverse_inertia * np.cross(normal, impulse)



    def setPosition(self, pos):
        X = np.array([
            [Shape.SCALING_FACTOR, 0, 0],
            [0, Shape.SCALING_FACTOR, 0],
            [0, 0, 1]
        ])
        invX = np.linalg.inv(X)
        pg = np.array([pos[0], pos[1], 1])
        ps = invX.dot(pg)
        self.position[0], self.position[1] = ps[0], ps[1]

    def updateSpeeds(self):
        if self.sprite:
            return

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

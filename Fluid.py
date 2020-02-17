
from Ball import Ball
from node import Node
from Shape import Shape
from Debug import Debug
import numpy as np





def cross(scalar, vector):
    if isinstance(vector, np.ndarray):
        new_x = -scalar*vector[1]
        new_y = scalar*vector[0]
        new_vector = np.array((new_x, new_y))
        return new_vector
    elif isinstance(scalar, np.ndarray): #scalar je vektor zapravo
        scalar, vector = vector, scalar

        new_x = scalar*vector[1]
        new_y = -scalar*vector[0]
        new_vector = np.array((new_x, new_y))
        return new_vector
    return None


def clamp(a, low, high):
    return max(low, min(a, high))




def projektuj(shape, n):
    if not isinstance(shape, Ball):
        extremi = shape.getExtremities()
    else:
        extremi = shape.getExtremities(n)
    levo = n.dot(extremi[0])
    desno = levo
    for i in range(1, len(extremi)):
        value = n.dot(extremi[i])
        if value < levo:
            levo = value
        if value > desno:
            desno = value
    return Node(levo, desno)


class Fluid:

    BETA = 0.2
    SLOP = 0.03



    def __init__(self, density, velocity, shapes=[]):
        self.density = density
        self.velocity = velocity
        self.shapes = shapes
        self.cashed_data_old = {}
        self.cashed_data_new = {}

    def add(self, other):
        self.shapes.append(other)

    def recalibrate_shapes_positions(self):
        for shape in self.shapes:
            shape.recalibrate_position()

    def check_coallision(self):

        the_list = []
        temp = None
        for i in range(0, len(self.shapes)):
            shape = self.shapes[i]
            for j in range(i+1, len(self.shapes)):
                other = self.shapes[j]
                if isinstance(shape, Ball):
                    if isinstance(other, Ball):
                        temp = self.check_balls(shape, other)
                    else:
                        temp = self.checkObjectss(other, shape)
                        if temp is None:
                            temp = self.check_object_ball(shape, other)
                elif isinstance(other, Ball):
                    temp = self.checkObjectss(shape, other)
                    if temp is None:
                        temp = self.check_object_ball(other, shape)
                else:
                    temp = self.checkObjectss(shape, other)
                if temp:
                    the_list.append(temp)
        return the_list

    def check_balls(self, shape, other):

        v = shape.getCenter() - other.getCenter()
        d = np.sqrt(v.dot(v))
        if (d - shape.radius - other.radius) < 0:
            v = v / np.sqrt(v.dot(v))
            return Node(v, shape.radius - other.radius, [shape, other])

        return None


    def checkObjectss(self, shape, other):
        if shape.sprite & other.sprite:
            return
        normale = shape.getNormals()
        length = len(normale)
        if not isinstance(other, Ball):
            normale2 = (other.getNormals())
            normale = np.concatenate((normale, normale2))
        biggest_d = -float("inf")
        inner_most_extreme = None
        object_penetrating = None
        object_being_penetrated = None
        for i in range(len(normale)):
            n = normale[i]


            p1 = projektuj(shape, n)
            p2 = projektuj(other, n)
            d = p1.inRange(p2)
            if d is None:
                return
            if d > biggest_d:
                biggest_d = d
                inner_most_extreme = n
                if i < length:
                    object_being_penetrated = shape
                    object_penetrating = other
                else:
                    object_penetrating = shape
                    object_being_penetrated = other

        return Node(inner_most_extreme, biggest_d, [object_being_penetrated, object_penetrating])

    def check_object_ball(self, ball, other):
        if ball.sprite & other.sprite:
            return
        e1 = ball.getCenter()
        min_d = float("inf")
        min_edge = None
        for e2 in other.getExtremities():
            d = e1 - e2
            d = np.linalg.norm(d)
            if min_d < d:
                min_d = d
                min_edge = e2
        if min_d < ball.radius:
            normal = min_edge / np.linalg.norm(min_edge)
            return Node(normal, min_d, (ball, other))
        else:
            return None



    def find_point(self, node):
        # Kod pisan po uzoru na http://www.dyn4j.org/2011/11/contact-points-using-clipping/

        object1 = node.items[1] # incident object
        object2 = node.items[0] # reference object
        normal = node.value1

        max_extreme_e1, point1_e1, point2_e1 = object1.getCoallisionEdge(-normal) # incident edge
        max_extreme_e2, point1_e2, point2_e2 = object2.getCoallisionEdge(normal) # reference edge
        #print(max_extreme_e1, point1_e1, point2_e1, object2.angle, normal)
        if isinstance(object1, Ball):
            return max_extreme_e1
        if isinstance(object2, Ball):
            d1 = object2.getCenter() - point1_e1
            d2 = object2.getCenter() - point2_e1
            d1 = np.linalg.norm(d1)
            d2 = np.linalg.norm(d2)
            if d1 < d2:
                return point1_e1
            return point2_e1

        e2 = point2_e2 - point1_e2

        e2 = e2 / np.linalg.norm(e2)

        o1 = e2.dot(point1_e2)

        clips = self.clip(point1_e1, point2_e1, e2, o1)
        if len(clips) < 2:
            return

        o2 = e2.dot(point2_e2)

        clips = self.clip(clips[0], clips[1], -e2, -o2)
        if len(clips) < 2:
            return

        max_value = normal.dot(max_extreme_e2)

        if normal.dot(clips[0]) - max_value > 0:
            clips.pop(0)
            if normal.dot(clips[0]) - max_value > 0:
                clips.pop(0)
        elif normal.dot(clips[1]) - max_value > 0:
            clips.pop(1)
        for c in clips:
            Debug.draw_point(c, (50, 50, 50))
        dot = None
        if len(clips) == 2:
            clips[0] = (clips[0] + clips[1])/2
            clips.pop()
        if len(clips) == 1:
            dot = clips[0]
        return dot

    def clip(self, point1_e1, point2_e1, e2, o1):
        # Kod pisan po uzoru na http://www.dyn4j.org/2011/11/contact-points-using-clipping/

        clipped = []
        d1 = e2.dot(point1_e1) - o1
        d2 = e2.dot(point2_e1) - o1

        if d1 >= 0:
            clipped.append(point1_e1)
        if d2 >= 0:
            clipped.append(point2_e1)

        if d1 * d2 < 0:
            u = d1 / (d1 - d2)
            e1 = point2_e1 - point1_e1
            e1 *= u
            e1 += point1_e1
            clipped.append(e1)
        return clipped



    def collide(self):
        # Kod pisan po uzoru na https://github.com/erincatto/box2d-lite/

        coallisions = self.check_coallision()
        if not coallisions:
            return

        self.cashed_data_new = {}
        for i in range(len(coallisions)):
            node = coallisions[i]

            referenced_object = node.items[0]
            incident_object = node.items[1]

            referenced_object.updateSpeeds()
            incident_object.updateSpeeds()

            n = incident_object.getCenter() - referenced_object.getCenter()
            if n.dot(node.value1) < 0:
                node.value1 *= -1
            normal = node.value1
            point_of_coallision = self.find_point(node)
            if point_of_coallision is None:
                coallisions[i] = None
                continue
            # DRAW COLLISION POINTS
            else:
                Debug.draw_point(point_of_coallision)
            r1 = point_of_coallision - referenced_object.getCenter()
            r2 = point_of_coallision - incident_object.getCenter()


            rn1 = r1.dot(normal)
            rn2 = r2.dot(normal)
            k_normal = 1/referenced_object.mass + 1/incident_object.mass

            k_normal += referenced_object.inverse_inertia * (r1.dot(r1) - rn1*rn1) + incident_object.inverse_inertia * (r2.dot(r2) - rn2*rn2)

            mass_normal = 1/k_normal

            bias = -Fluid.BETA * (1/Shape.DELTA_TIME)*min(0, node.value2 + Fluid.SLOP)


            tangent = cross(normal, 1)
            rt1 = r1.dot(tangent)
            rt2 = r2.dot(tangent)
            k_tangent = 1/referenced_object.mass + 1/incident_object.mass
            k_tangent += referenced_object.inverse_inertia * (r1.dot(r1) - rt1*rt1) + incident_object.inverse_inertia * (r2.dot(r2) - rt2*rt2)
            mass_tangent = 1/k_tangent
            node.mass_tangent = mass_tangent

            node.r1 = r1
            node.r2 = r2
            node.mass_normal = mass_normal
            node.bias = bias
            friction = np.sqrt(referenced_object.friction * incident_object.friction)
            node.friction = friction

            key = (str(id(referenced_object)) + str(id(incident_object)))
            if key in self.cashed_data_old:
                node.normal_impulse = self.cashed_data_old[key].normal_impulse
                node.tangent_impulse = self.cashed_data_old[key].tangent_impulse
                self.cashed_data_new[key] = node

                old_impuse = node.normal_impulse * normal + node.tangent_impulse*tangent
                referenced_object.add_impulse(-old_impuse, node.r1)
                incident_object.add_impulse(old_impuse, node.r2)
                node.key = key
            else:
                node.normal_impulse = 0
                node.tangent_impulse = 0
                node.key = key
                self.cashed_data_new[key] = node

        for i in range(10):
            for node in coallisions:
                if node is None:
                    continue
                referenced_object = node.items[0]
                incident_object = node.items[1]
                normal = node.value1
                velocity = incident_object.vector + cross(incident_object.angularspeed, node.r2) - referenced_object.vector - cross(referenced_object.angularspeed, node.r1)

                velocity_value = velocity.dot(normal)

                normal_impulse = node.mass_normal*(-velocity_value + node.bias)

                if node.key in self.cashed_data_new:
                    temp = node.normal_impulse
                    node.normal_impulse = max(temp + normal_impulse, 0)
                    normal_impulse = node.normal_impulse - temp
                else:
                    normal_impulse = max(normal_impulse, 0)


                impulse = normal_impulse*normal
                referenced_object.add_impulse(-impulse, node.r1)
                incident_object.add_impulse(impulse, node.r2)

                velocity = incident_object.vector + cross(incident_object.angularspeed, node.r2) - cross(
                    referenced_object.angularspeed, node.r1) - referenced_object.vector
                tangent = cross(normal, 1)
                tangent_velocity_value = velocity.dot(tangent)
                tangent_impulse_value = node.mass_tangent*(-tangent_velocity_value)

                friction = node.friction

                if node.key in self.cashed_data_new:
                    max_tangent_value = friction * node.normal_impulse
                    old_tangent_impulse = node.tangent_impulse
                    node.tangent_impulse = clamp(old_tangent_impulse + tangent_impulse_value, -max_tangent_value, max_tangent_value)
                    tangent_impulse_value = node.tangent_impulse-old_tangent_impulse

                else:
                    max_value = friction * normal_impulse
                    tangent_impulse_value = clamp(tangent_impulse_value, -max_value, max_value)

                end_tangent_impulse = tangent_impulse_value * tangent


                referenced_object.add_impulse(-end_tangent_impulse, node.r1)
                incident_object.add_impulse(end_tangent_impulse, node.r2)

        self.cashed_data_old = self.cashed_data_new
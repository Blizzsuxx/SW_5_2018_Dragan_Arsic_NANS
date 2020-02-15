
from Ball import Ball
from node import Node
from Shape import Shape
from Debug import Debug
import numpy as np










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
                        temp = self.check_object_ball(shape, other)
                elif isinstance(other, Ball):
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
        normale = other.getNormals()
        lista = []
        the_list = []
        e1 = ball.getCenter()
        for e2 in other.getExtremities():
            broken = False
            for n in normale:
                d = e2 - e1
                p = abs(n.dot(d))
                s = p - ball.radius
                if s >= 0:
                    broken = True
                    break
                else:
                    lista.append(Node(n, s, [ball, other]))
                    #print("NORMALA: ", n)
            if not broken:
                lista.sort()
                the_list.append(lista[0])
            else:
                lista.clear()

        if the_list:
            the_list.sort()
            print("LIST WORKED")
            return the_list[0]
        else:
            return self.checkObjectss(other, ball)



    def find_point(self, node):

        object1 = node.items[1] # incident object
        object2 = node.items[0] # reference object
        normal = node.value1

        max_extreme_e1, point1_e1, point2_e1 = object1.getCoallisionEdge(-normal) # incident edge
        max_extreme_e2, point1_e2, point2_e2 = object2.getCoallisionEdge(normal) # reference edge
        #print(max_extreme_e1, point1_e1, point2_e1, object2.angle, normal)
        if isinstance(object1, Ball):
            return max_extreme_e1
        if isinstance(object2, Ball):
            return max_extreme_e2

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
        coallisions = self.check_coallision()
        if not coallisions:
            return

        #print(coallisions[0].value1, coallisions[0].value2, "AA")
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

            k_normal += (1/referenced_object.inertia) * (r1.dot(r1) - rn1*rn1) + (1/incident_object.inertia) * (r2.dot(r2) - rn2*rn2)

            mass_normal = 1/k_normal

            bias = -Fluid.BETA * (1/Shape.DELTA_TIME)*min(0, node.value2 + Fluid.SLOP)

            node.r1 = r1
            node.r2 = r2
            node.mass_normal = mass_normal
            node.bias = bias

        #self.cashed_data_new = {}
        for node in coallisions:
            if node is None:
                continue
            referenced_object = node.items[0]
            incident_object = node.items[1]
            normal = node.value1
            velocity = incident_object.vector + incident_object.angularspeed * node.r2 - referenced_object.angularspeed * node.r1 - referenced_object.vector

            velocity_value = velocity.dot(normal)
            print(velocity_value)

            #print(velocity_value, normal, incident_object.angularspeed, type(incident_object))
            normal_impulse = node.mass_normal*(-velocity_value + node.bias)

            key = (str(id(referenced_object)) + str(id(incident_object)))
            if key in self.cashed_data_new:
                p = self.cashed_data_new[key]
                self.cashed_data_new[key] = max(p + normal_impulse, 0)
                normal_impulse = self.cashed_data_new[key] - p
                self.cashed_data_new.pop(key)
            else:
                normal_impulse = max(normal_impulse, node.bias * node.mass_normal)
                self.cashed_data_new[key] = normal_impulse





            """
            key = (str(id(referenced_object)) + str(id(incident_object)))
            if key in self.cashed_data_old:
                p = self.cashed_data_old[key]
                self.cashed_data_new[key] = max(p + normal_impulse, 0)
                normal_impulse = self.cashed_data_new[key] - p
            else:
                normal_impulse = max(normal_impulse, node.bias*node.mass_normal)
                self.cashed_data_new[key] = normal_impulse
            """

            #print(normal_impulse, node.bias)

            impulse = normal_impulse*normal
            if not referenced_object.sprite:
                referenced_object.vector -= 1/referenced_object.mass * impulse
                referenced_object.angularspeed -= 1/referenced_object.inertia * np.cross(node.r1, impulse)

            if not incident_object.sprite:
                incident_object.vector += 1 / incident_object.mass * impulse
                incident_object.angularspeed += 1 / incident_object.inertia * np.cross(node.r2, impulse)
        #self.cashed_data_old = self.cashed_data_new



        """
            object1 = node.items[0]
            object2 = node.items[1]
            normal = node.value1

            n = object2.getCenter() - object1.getCenter()
            #print(normal, "aaaaaaaaaaaaaaaaaaa")
            if(n.dot(normal) < 0):
                normal *= -1
            #print(n/np.linalg.norm(n))

            point_of_coallision = self.find_point(node)
            if point_of_coallision is None:
                continue
            #print("SPEEDS BEFORE: ", object1.angularspeed, object1.vector)
            if not object1.sprite:
                object1.updateSpeeds()
            if not object2.sprite:
                object2.updateSpeeds()
            #print("SPEEDS AFTER: ", object1.angularspeed, object1.vector)

            r1 = object1.getCenter() - point_of_coallision # vector of distance between object1 and coallision point
            r2 = object2.getCenter() - point_of_coallision

            effective_mass = 1/object1.mass + 1/object2.mass

            temp = (1 / object1.inertia * (np.cross(r1, normal) * r1))
            temp += (1 / object2.inertia * (np.cross(r2, normal) * r2))
            temp = temp.dot(normal)

            effective_mass += temp
            effective_mass = 1 / effective_mass

            delta_velocity = object2.vector + (object2.angularspeed * r2) - (object1.vector + (object1.angularspeed * r1))
            effective_velocity = delta_velocity.dot(normal)
            #effective_velocity = max(effective_velocity, 0)

            bias = (Fluid.BETA/Shape.DELTA_TIME)*(node.value2-Fluid.SLOP)
            #bias = max(bias, 0)
            print(bias, effective_velocity)
            effective_impulse = -effective_mass*(effective_velocity + bias)

            effective_impulse = max(effective_impulse, 0)

            impulse = effective_impulse * normal
            print(impulse, ":IMPULS ---- NORMAL: ", normal)
            #print(object1.vector, impulse)
            #print("EFFECTIVE IMPUSLE: ", effective_impulse, normal)
            #print(node.value1, node.value2, object2.vector)
            #print("OBJECT PENETRATING: ", object2.position, object2.angle, object2.getExtremities())
            #print("OBJECT BEING PENETRATED: ", object1.position, object1.angle)
            if not object1.sprite:
                object1.vector -= impulse / object1.mass
                object1.angularspeed -= 1 / object1.inertia * (np.cross(r1, impulse))
                object1.angle = object1.angle + Shape.DELTA_TIME * object1.angularspeed
                object1.angle %= 360
                for i in range(len(object1.vector)):
                    temp = Shape.DELTA_TIME * object1.vector[i]
                    object1.position[i] = object1.position[i] + temp

            #print(object1.vector, impulse)
            if not object2.sprite:
                object2.vector += impulse / object2.mass
                object2.angularspeed += 1 / object2.inertia * (np.cross(r2, impulse))
                object2.angle = object2.angle + Shape.DELTA_TIME * object2.angularspeed
                object2.angle %= 360
                for i in range(len(object2.vector)):
                    temp = Shape.DELTA_TIME * object2.vector[i]
                    object2.position[i] = object2.position[i] + temp
            #print("OBJECT PENETRATING: ", object2.position)
            #print("OBJECT BEING PENETRATED: ", object1.position)
            #print("IMPULSE: ", impulse)
            """

        """
        n_new = np.zeros((len(coallisions), len(coallisions)*2 + 2))
        v_new = np.zeros(len(coallisions)*2 + 2)
        s_new = np.zeros(len(coallisions))
        m_new = np.zeros((len(coallisions)*2 + 2, len(coallisions)*2 + 2))
        f_new = np.zeros(len(coallisions)*2 + 2)
        for i in range(len(coallisions)):
            j = i*2
            c = coallisions[i]
            n_new[i][j] = -c.value2[0]
            n_new[i][j+1] = -c.value2[1]
            n_new[i][j + 2] = c.value2[0]
            n_new[i][j + 3] = c.value2[1]
            s_new[i] = c.value1
            o1 = c.items[0]
            o2 = c.items[1]
            v_new[j] = o1.vector[0]
            v_new[j+1] = o1.vector[1]
            v_new[j+2] = o2.vector[0]
            v_new[j+3] = o2.vector[1]
            m_new[j][j] = o1.mass
            m_new[j+1][j + 1] = o1.mass
            m_new[j+2][j + 2] = o2.mass
            m_new[j+3][j + 3] = o2.mass
            f_new[j] = o1.force[0]
            f_new[j + 1] = o1.force[1]
            f_new[j + 2] = o2.force[0]
            f_new[j + 3] = o2.force[1]
        n_dot_m = n_new.dot(1/m_new)
        A = n_dot_m.dot(n_new.T)
        print(n_dot_m, "aaaaaaaaa")
        print(n_new, "bbbbbbbbbbb")
        p1 = -Fluid.BETA * (1 / Shape.DELTA_TIME**2)*s_new
        p2 = 1/Shape.DELTA_TIME * n_new.dot(v_new)
        try:
            p3 = n_dot_m.dot(f_new)
        except:
            p3 = n_dot_m * f_new
        b = p1 - p2 - p3
        print(A)
        print(b)
        x = np.linalg.solve(A, b)
    """

    """ def check_object_ball(self, ball, other):
            edges = other.getExtremities()
            min_d = float('inf')
            min_e = None
            for e in edges:
                a = ball.position - e
                d = np.sqrt(a.dot(a))

                if d > ball.radius:
                    continue

                if d < min_d:
                    min_d = d
                    min_e = e
            if min_e is None:
                return
            return Node(min_d, min_e, [ball, other])"""
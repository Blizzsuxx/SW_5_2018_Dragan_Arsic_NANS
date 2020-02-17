class Node:

    def __init__(self, value1, value2, items=None):
        self.value1 = value1
        self.value2 = value2
        self.items = items

    def inRange(self, node2):
        if (self.value2 >= node2.value1) and (self.value1 <= node2.value1):
            return -self.value2 + node2.value1
        elif self.value1 <= node2.value2 and self.value2 >= node2.value1:
            return -node2.value2 + self.value1
        return None

    def __lt__(self, other):
        return self.value2 > other.value2


    def __gt__(self, other):
        return self.value2 < other.value2

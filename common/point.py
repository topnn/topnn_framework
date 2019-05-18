from collections import namedtuple


class Point2D(namedtuple('Point2D', ('x', 'y'))):
    def __add__(self, other):
        return Point2D(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other):
        return Point2D(x=self.x - other.x, y=self.y - other.y)

    def __pow__(self, power, modulo=None):
        return Point2D(x=self.x ** power, y=self.y ** power)


Point3D = namedtuple('Point3D', ('x', 'y', 'z'))

import math as mt
import numpy as np


class Calculations:
    def __init__(self):
        pass

    def lineGradient(self, line):
        ## first coordinate point
        x1, y1 = line[0]

        # second co-ordinate point
        x2, y2 = line[1]
        try:
            gradient = (y1 - y2) / (x1 - x2)
        except ArithmeticException:
             print("angle of rotation is 90 degrees")
             gradient = 10000

        return gradient

    def center_point(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def rotationAngle(self, line_1, line_2):
        m1 = self.lineGradient(line_1)
        m2 = self.lineGradient(line_2)
        tan = (m1 - m2) / (1 + m1 * m2)
        angle = mt.atan(tan)

        return angle * -1

    def rotation(self, c, point, theta):
        """
        Rotates a point about a point c
        c - rotation point
        """

        T = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        cx, cy = c
        x, y = point

        c = np.array([cx, cy])
        point = np.array([x, y])

        return T.dot(point - c) + c


class Transformation:
    """
    Does mathematical transformation calculations

    """
    def __init__(self):
        pass

    def enlargment(self, p1, scale=1):
        """
        computes enlargment by scaling the point given

        Args:
            p1(tuple): point on x, y, z plane
            scale(int): scaling factor to be applied to point p1
        """
        return p1 * scale

    def rotation(self, p1, theta):

        """
        rotates point given about origin

        Args:
            p1(numpy array): point on x, y, z plane
            theta(float): angle in radians
        Returns the rotated co-ordinate points
        """
        R = np.array([[np.cos(theta), np.sin(theta) * -1],
                      [np.sin(theta), np.cos(theta)]])
        res = R.dot(p1)
        return res

    def translation(self, p1, t_vector):
        """
        Translates a point using translation vector.

        Args:
            p1(tuple): point to be translated
            t_vector(tuple): the translation vector

        Returns the translated version of p1
        """
        return p1 + t_vector

def main():
    pass
if __name__=="__main__":
    main()

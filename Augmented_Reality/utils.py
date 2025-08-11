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

def main():
    x1, y1 = 10, 0
    x2, y2 = 16, 0

    x3, y3 = 5, 7
    x4, y4 = -19, 700
    line_1 = [(x1, y1), (x2, y2)]
    line_2 = [(x3, y3), (x4, y4)]

    calc = Calculations()
    angle = calc.rotationAngle(line_1, line_2)
    print(angle)

if __name__=="__main__":
    main()

from math import sqrt
import cv2
import numpy as np


def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['im'], (x, y), 3, (0, 255, 255), 5, 16)
        cv2.imshow("Image", data['im'])
        if len(data['points']) < 4:
            data['points'].append([x, y])


def get_four_points(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)

    return points


def remove_duplicates(lst):
    return [t for t in (set(tuple(i) for i in lst))]


def create_line(x1, y1, x2, y2):
    m = float(y2 - y1)/(x2-x1)
    q = y1 - (m*x1)
    return m, q


def dy(distance, m):
    return m * dx(distance, m)


def dx(distance, m):
    return sqrt(distance / (m ** 2 + 1))


def line_intersection(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)

    uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d

    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
    return x, y


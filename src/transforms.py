import math
import numpy as np

def translate(x, y, z, tx=0, ty=0, tz=0):
    return x + tx, y + ty, z + tz

def scale_point(x, y, z, sx=1, sy=1, sz=1):
    return sx * x, sy * y, sz * (z - 5) + 5

def rotate_point(x, y, z, angle_x=0, angle_y=0, angle_z=0,
                 center=(0, 0, 5)):
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    cy, sy = math.cos(angle_y), math.sin(angle_y)
    cz, sz = math.cos(angle_z), math.sin(angle_z)

    Rx = np.array([
        [1,  0,   0],
        [0, cx, -sx],
        [0, sx,  cx],
    ])

    Ry = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy],
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0, 1],
    ])

    rotation = Rz @ Ry @ Rx
    center = np.array(center)
    point = np.array([x, y, z])

    return tuple(rotation @ (point - center) + center)
import math
import numpy as np
import matplotlib.pyplot as plt


def projectPoint(x, y, z, fx=150, fy=150, cx=128, cy=128):
    if z <= 0:
        return None

    u = fx * x / z + cx
    v = cy - fy * y / z
    return u, v

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


points = [
    [-1, -1, 4],
    [-1, -1, 6],
    [-1,  1, 4],
    [-1,  1, 6],
    [ 1, -1, 4],
    [ 1, -1, 6],
    [ 1,  1, 4],
    [ 1,  1, 6],
]


edges = []

for i in range(len(points)):
    for j in range(i + 1, len(points)):
        differences = sum(
            points[i][axis] != points[j][axis]
            for axis in range(3)
        )

        if differences == 1:
            edges.append((i, j))

theta = math.radians(30)

# Preserve point ordering so edge indices remain valid.
projected_points = []

for x, y, z in points:
    x, y, z = scale_point(x, y, z, sx=1.5, sy=1, sz=1)
    x, y, z = rotate_point(x, y, z, angle_x=theta, angle_y=theta+30)
    projected_points.append(projectPoint(x, y, z))


for i, j in edges:
    start = projected_points[i]
    end = projected_points[j]

    if start is not None and end is not None:
        plt.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="black"
        )


for projected in projected_points:
    if projected is not None:
        u, v = projected
        plt.scatter(u, v, color="red", zorder=3)

plt.xlim(0, 256)
plt.ylim(256, 0)
plt.gca().set_aspect("equal")
plt.xlabel("u (pixels)")
plt.ylabel("v (pixels)")
plt.title("Projected cube")
plt.grid()
plt.show()
import math
import numpy as np
import matplotlib.pyplot as plt
from transforms import translatePoint, scalePoint, rotatePoint
from camera import projectPoint

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
        
        differences = sum(points[i][axis] != points[j][axis] for axis in range(3))

        if differences == 1:
            edges.append((i, j))

theta = math.radians(30)

# Preserve point ordering so edge indices remain valid.
projected_points = []

for x, y, z in points:
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
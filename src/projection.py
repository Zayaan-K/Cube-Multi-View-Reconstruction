import math
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


def rotate_y(x, y, z, theta):
    local_z = z - 5

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    rotated_x = cos_theta * x + sin_theta * local_z
    rotated_z = -sin_theta * x + cos_theta * local_z

    return rotated_x, y, rotated_z + 5

def rotate_x(x, y, z, theta):
    local_z = z - 5

    c = math.cos(theta)
    s = math.sin(theta)

    rotated_y = c * y - s * local_z
    rotated_z = s * y + c * local_z

    return x, rotated_y, rotated_z + 5


def rotate_z(x, y, z, theta):
    c = math.cos(theta)
    s = math.sin(theta)

    rotated_x = c * x - s * y
    rotated_y = s * x + c * y

    return rotated_x, rotated_y, z


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

    x, y, z = rotate_x(x, y, z, theta)


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
plt.title("Projected wireframe cube")
plt.grid()
plt.show()
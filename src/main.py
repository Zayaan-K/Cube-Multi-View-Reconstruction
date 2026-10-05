import math
import matplotlib.pyplot as plt

from geometry import createCube
from transforms import translatePoint, scalePoint, rotatePoint
from camera import Camera
from render import drawWireframe


def main():
    points, edges = createCube()

    scale = (1, 1, 1)
    angles = (0, math.radians(0), 0)
    translation = (0, 0, 0)

    camera = Camera(position=(0, 0, 0), angles=(0, 0, 0))

    transformed_points = []
    for point in points:
        point = scalePoint(*point, *scale)
        point = rotatePoint(*point, *angles)
        point = translatePoint(*point, *translation)
        transformed_points.append(point)

    projected_points = [camera.project(point) for point in transformed_points]
    drawWireframe(projected_points, edges)
    plt.show()


if __name__ == "__main__":
    main()

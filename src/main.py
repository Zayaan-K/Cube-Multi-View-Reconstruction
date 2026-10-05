import math
import matplotlib.pyplot as plt

from geometry import createCube
from transforms import translatePoint, scalePoint, rotatePoint
from camera import Camera
from render import drawWireframe
from triangulation import triangulatePoint


def main():
    points, edges = createCube()

    scale = (1, 1, 1)
    angles = (0, math.radians(0), 0)
    translation = (0, 0, 0)

    cameraA = Camera(position=(-1, 0, 0))
    cameraB = Camera(position=(1, 0, 0))

    transformed_points = []
    for point in points:
        point = scalePoint(*point, *scale)
        point = rotatePoint(*point, *angles)
        point = translatePoint(*point, *translation)
        transformed_points.append(point)

    projectedA = [cameraA.project(point) for point in transformed_points]
    projectedB = [cameraB.project(point) for point in transformed_points]

    for original, pointA, pointB in zip(
        transformed_points, projectedA, projectedB
    ):
        reconstructed = triangulatePoint(
            pointA, pointB, cameraA, cameraB
        )

        print("Original:", original)
        print("Reconstructed:", reconstructed)
        print()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    drawWireframe(projectedA, edges, ax=axes[0], title="Camera A")
    drawWireframe(projectedB, edges, ax=axes[1], title="Camera B")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
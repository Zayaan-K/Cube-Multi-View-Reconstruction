from dataclasses import dataclass
import numpy as np

from .transforms import rotatePoint


def projectPoint(x, y, z, fx=150, fy=150, cx=128, cy=128):
    #positive Z is forward, positive Y is up
    if z <= 0:
        return None
    return fx * x / z + cx, cy - fy * y / z


@dataclass
class Camera:
    position: tuple = (0, 0, 0)

    angles: tuple = (0, 0, 0)
    fx: float = 150
    fy: float = 150
    cx: float = 128
    cy: float = 128

    def worldToCamera(self, point):

        rotation = np.column_stack([
            rotatePoint(*axis, *self.angles, center=(0, 0, 0))
            for axis in np.eye(3)])
        relative = np.asarray(point, dtype=float) - np.asarray(self.position)
        return rotation.T @ relative

    def project(self, point):
        return projectPoint(*self.worldToCamera(point), fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)

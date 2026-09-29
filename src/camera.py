def projectPoint(x, y, z, fx=150, fy=150, cx=128, cy=128):
    if z <= 0:
        return None

    u = fx * x / z + cx
    v = cy - fy * y / z
    return u, v
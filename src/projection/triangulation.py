def triangulatePoint(pointA, pointB, cameraA, cameraB):
    uA, vA = pointA
    uB, vB = pointB

    baseline = cameraB.position[0] - cameraA.position[0]
    disparity = uA - uB

    if abs(disparity) < 1e-9:
        return None

    depth = cameraA.fx * baseline / disparity

    if depth <= 0:
        return None

    x = cameraA.position[0] + (uA - cameraA.cx) * depth / cameraA.fx
    y = cameraA.position[1] + (cameraA.cy - vA) * depth / cameraA.fy
    z = cameraA.position[2] + depth

    return x, y, z
def createCube():

    points = [
        [-1, -1, 4], [-1, -1, 6],
        [-1, 1, 4], [-1, 1, 6],
        [1, -1, 4], [1, -1, 6],
        [1, 1, 4], [1, 1, 6],
    ]
    edges = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            differences = sum(points[i][axis] != points[j][axis]
                              for axis in range(3))
            if differences == 1:
                edges.append((i, j))
    return points, edges

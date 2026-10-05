
import matplotlib.pyplot as plt


def drawWireframe(projected_points, edges, ax=None, width=256, height=256, title="Projected cube"):

    if ax is None:
        _, ax = plt.subplots()
    for i, j in edges:
        start, end = projected_points[i], projected_points[j]
        if start is not None and end is not None:
            ax.plot([start[0], end[0]], [start[1], end[1]], color="black")
    visible = [point for point in projected_points if point is not None]
    if visible:
        u, v = zip(*visible)
        ax.scatter(u, v, color="red", zorder=3)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")
    ax.set_title(title)
    ax.grid()
    return ax

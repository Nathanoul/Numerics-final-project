import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def plot_steady_state(points, edges, solution, cmap="inferno", show_mesh=False, triag=None):
    """
    Plots heat equation solution on an unstructured mesh.

    parameters
    ----------
    points : dict
        points[id] = {"x": x, "y": y}
    solution : np.ndarray
        temperature value at each point (indexed by point id)
    edges : dict | None
        optional, for drawing mesh edges
    cmap : str
        matplotlib colormap
    """

    # sort points by id to match solution ordering
    x = np.array(list(points["x"].values()))
    y = np.array(list(points["y"].values()))
    T = solution

    # triangulate automatically
    triang = mtri.Triangulation(x, y)

    plt.figure(figsize=(7, 6))
    contour = plt.tricontourf(triang, T, levels=50, cmap=cmap)
    plt.colorbar(contour, label="Temperature")

    # optionally draw mesh edges
    if show_mesh:
        for edge_id in edges["edgeID"]:
            n1 = edges["n1"][edge_id]
            n2 = edges["n2"][edge_id]
            plt.plot(
                [x[n1], x[n2]],
                [y[n1], y[n2]],
                color="blue",
                linewidth=0.5,
                alpha=0.6,
                origin='lower'
            )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Heat Equation Solution")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
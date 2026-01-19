import matplotlib.pyplot as plt


def plot_mesh(points_list, edges_list, show_ids=False):
    """
    Plot 2D mesh using points and edges key lists.

    Parameters
    ----------
    points_list : list of dict
        Keys: pointID, x, y
    edges_list : list of dict
        Keys: edgeID, n1, n2
    show_ids : bool
        If True, plot point and edge IDs
    """

    fig, ax = plt.subplots()

    # Convert points to dict for fast access
    point_map = {p["pointID"]: (p["x"], p["y"]) for p in points_list}

    # Plot edges
    for edge in edges_list:
        n1 = int(edge["n1"])
        n2 = int(edge["n2"])

        x1, y1 = point_map[n1]
        x2, y2 = point_map[n2]

        ax.plot([x1, x2], [y1, y2], "k-", linewidth=1)

        if show_ids:
            xm = 0.5 * (x1 + x2)
            ym = 0.5 * (y1 + y2)
            ax.text(xm, ym, int(edge["edgeID"]), color="blue", fontsize=8)

    # Plot points
    xs = [p["x"] for p in points_list]
    ys = [p["y"] for p in points_list]
    ax.scatter(xs, ys, c="red", s=15)

    if show_ids:
        for p in points_list:
            ax.text(p["x"], p["y"], int(p["pointID"]),
                    color="red", fontsize=8, verticalalignment="bottom")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Mesh")
    ax.grid(True)

    plt.show()

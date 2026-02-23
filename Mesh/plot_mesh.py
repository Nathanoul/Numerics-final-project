import matplotlib.pyplot as plt


def plot_mesh(points, edges, show_ids=False):
    """
    Plot mesh from dictionary structure created by df.to_dict('dict')

    Parameters
    ----------
    points : dict
        Output of points_df.to_dict('dict')
    edges : dict
        Output of edges_df.to_dict('dict')
    show_ids : bool
        If True, show point and edge IDs
    """

    fig, ax = plt.subplots()

    # Number of edges
    num_edges = len(edges["edgeID"])

    # Plot edges
    for i in range(num_edges):
        n1 = int(edges["n1"][i])
        n2 = int(edges["n2"][i])

        x1 = points["x"][n1]
        y1 = points["y"][n1]
        x2 = points["x"][n2]
        y2 = points["y"][n2]

        ax.plot([x1, x2], [y1, y2], "k-", linewidth=1)

        if show_ids:
            xm = 0.5 * (x1 + x2)
            ym = 0.5 * (y1 + y2)
            ax.text(xm, ym, str(edges["edgeID"][i]),
                    color="blue", fontsize=8)

    # Plot points
    num_points = len(points["pointsID"])
    xs = [points["x"][i] for i in range(num_points)]
    ys = [points["y"][i] for i in range(num_points)]

    ax.scatter(xs, ys, s=15)

    if show_ids:
        for i in range(num_points):
            ax.text(points["x"][i], points["y"][i],
                    str(points["pointsID"][i]),
                    fontsize=8, verticalalignment="bottom")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Mesh")
    ax.grid(True)

    plt.show()
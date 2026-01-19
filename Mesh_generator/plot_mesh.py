import pandas as pd
import matplotlib.pyplot as plt


def plot_mesh(points_df, edges_df, show_ids=False):
    """
    Plot 2D mesh using points and edges tables.

    Parameters
    ----------
    points_df : DataFrame
        Columns: pointID, x, y
    edges_df : DataFrame
        Columns: edgeID, n1, n2, ...
    show_ids : bool
        If True, plot point and edge IDs
    """

    fig, ax = plt.subplots()

    # Plot edges
    for _, edge in edges_df.iterrows():
        n1 = int(edge["n1"])
        n2 = int(edge["n2"])

        id, x1, y1 = points_df.iloc[n1]
        id, x2, y2 = points_df.iloc[n2]

        ax.plot([x1, x2], [y1, y2], "k-", linewidth=1)

        if show_ids:
            xm = 0.5 * (x1 + x2)
            ym = 0.5 * (y1 + y2)
            ax.text(xm, ym, int(edge["edgeID"]), color="blue", fontsize=8)

    # Plot points
    ax.scatter(points_df["x"], points_df["y"], c="red", s=15)

    if show_ids:
        for _, p in points_df.iterrows():
            ax.text(p["x"], p["y"], int(p["pointID"]),
                    color="red", fontsize=8, verticalalignment="bottom")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Mesh")
    ax.grid(True)

    plt.show()

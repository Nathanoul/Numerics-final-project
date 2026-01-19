import numpy as np
import pandas as pd


def generate_square_mesh(N, M, X, Y, out_dir="Mesh_data/Part1"):
    """
    Generate a structured rectangular mesh for PDE solvers.

    Parameters
    ----------
    N, M : int
        Number of points in x and y directions
    X, Y : float
        Domain size
    out_dir : str
        Directory to save CSV files
    """

    # -------------------------
    # Generate points
    # -------------------------
    xs = np.linspace(0, X, N)
    ys = np.linspace(0, Y, M)

    points = []
    pid = 0
    point_id = {}

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            points.append([pid, x, y])
            point_id[(i, j)] = pid
            pid += 1

    points_df = pd.DataFrame(points, columns=["pointID", "x", "y"])

    # -------------------------
    # Generate cells (quads)
    # -------------------------
    cells = []
    cid = 0
    cell_id = {}

    for j in range(M - 1):
        for i in range(N - 1):
            p1 = point_id[(i, j)]
            p2 = point_id[(i + 1, j)]
            p3 = point_id[(i + 1, j + 1)]
            p4 = point_id[(i, j + 1)]

            cells.append([cid, p1, p2, p3, p4])
            cell_id[(i, j)] = cid
            cid += 1

    cells_df = pd.DataFrame(cells, columns=["cellID", "p1", "p2", "p3", "p4"])

    # -------------------------
    # Generate edges
    # -------------------------
    edges = []
    edge_map = {}
    eid = 0

    def add_edge(n1, n2, c_left, c_right):
        nonlocal eid
        key = tuple(sorted((n1, n2)))

        if key not in edge_map:
            id, x1, y1 = points_df.iloc[n1]
            id, x2, y2 = points_df.iloc[n2]
            length = np.hypot(x2 - x1, y2 - y1)

            edge_map[key] = eid
            edges.append([eid, n1, n2, c_left, c_right, length])
            eid += 1
        else:
            idx = edge_map[key]
            edges[idx][4] = c_left  # fill cellR

    for j in range(M - 1):
        for i in range(N - 1):
            c = cell_id[(i, j)]

            p1 = point_id[(i, j)]
            p2 = point_id[(i + 1, j)]
            p3 = point_id[(i + 1, j + 1)]
            p4 = point_id[(i, j + 1)]

            add_edge(p1, p2, c, -1)  # bottom
            add_edge(p2, p3, c, -1)  # right
            add_edge(p3, p4, c, -1)  # top
            add_edge(p4, p1, c, -1)  # left

    edges_df = pd.DataFrame(
        edges,
        columns=["edgeID", "n1", "n2", "cellL", "cellR", "length"]
    )

    # -------------------------
    # Save CSV files
    # -------------------------
    points_df.to_csv(f"{out_dir}/Points.csv", index=False)
    edges_df.to_csv(f"{out_dir}/Edges.csv", index=False)
    cells_df.to_csv(f"{out_dir}/Cells.csv", index=False)

    print("Mesh generated successfully:")
    print(f"  Points: {len(points_df)}")
    print(f"  Edges : {len(edges_df)}")
    print(f"  Cells : {len(cells_df)}")

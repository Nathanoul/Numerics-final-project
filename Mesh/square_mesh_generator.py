import numpy as np
import pandas as pd


def generate_square_mesh(x_nodes, y_nodes, x_span, y_span, out_dir):
    """
    Generate a structured rectangular quad mesh for FV solvers.

    Output CSVs (Points, Edges, Cells) use the same column conventions as
    the triangular MATLAB mesh so that build_fv_matrix works on both.

    Columns
    -------
    Points : pointID, x, y
    Cells  : cellID, cx, cy, n1, n2, n3, n4,
             nbr12, nbr23, nbr34, nbr41,
             n12x, n12y, n23x, n23y, n34x, n34y, n41x, n41y
    Edges  : edgeID, n1, n2, cellL, cellR, len, mx, my, nLx, nLy
    """

    xs = np.linspace(x_span[0], x_span[1], x_nodes)
    ys = np.linspace(y_span[0], y_span[1], y_nodes)

    # ------------------------------------------------------------------
    # Points
    # ------------------------------------------------------------------
    points = []
    pid = 0
    point_id = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            points.append([pid, x, y])
            point_id[(i, j)] = pid
            pid += 1
    points_df = pd.DataFrame(points, columns=["pointID", "x", "y"])

    # ------------------------------------------------------------------
    # Cells (quads)
    # ------------------------------------------------------------------
    extended_cells = []
    cell_id = {}
    cid = 0
    for j in range(y_nodes - 1):
        for i in range(x_nodes - 1):
            p1 = point_id[(i,     j    )]   # bottom-left
            p2 = point_id[(i + 1, j    )]   # bottom-right
            p3 = point_id[(i + 1, j + 1)]   # top-right
            p4 = point_id[(i,     j + 1)]   # top-left

            # centroid = midpoint of diagonal
            x1, y1 = points_df.iloc[p1][["x", "y"]]
            x3, y3 = points_df.iloc[p3][["x", "y"]]
            cx = (x1 + x3) / 2.0
            cy = (y1 + y3) / 2.0

            nbr12 = cell_id.get((i,     j - 1), -1)   # below
            nbr23 = cell_id.get((i + 1, j    ), -1)   # right
            nbr34 = cell_id.get((i,     j + 1), -1)   # above
            nbr41 = cell_id.get((i - 1, j    ), -1)   # left

            extended_cells.append([
                cid, cx, cy,
                p1, p2, p3, p4,
                nbr12, nbr23, nbr34, nbr41,
                0, -1,   # n12 normal: bottom face → outward = (0, -1)
                1,  0,   # n23 normal: right  face → outward = (1,  0)
                0,  1,   # n34 normal: top    face → outward = (0,  1)
               -1,  0,   # n41 normal: left   face → outward = (-1, 0)
            ])
            cell_id[(i, j)] = cid
            cid += 1

    cells_df = pd.DataFrame(extended_cells, columns=[
        "cellID", "cx", "cy",
        "n1", "n2", "n3", "n4",
        "nbr12", "nbr23", "nbr34", "nbr41",
        "n12x", "n12y", "n23x", "n23y", "n34x", "n34y", "n41x", "n41y",
    ])

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    edges    = []
    edge_map = {}
    eid      = 0

    def add_edge(n1, n2, c_left):
        nonlocal eid
        key = (min(n1, n2), max(n1, n2))

        if key not in edge_map:
            # BUG FIX: was  "id, x1, y1 = ..."  (3 vars from 2 values)
            x1, y1 = points_df.iloc[n1][["x", "y"]]
            x2, y2 = points_df.iloc[n2][["x", "y"]]
            mx     = (x1 + x2) / 2.0
            my     = (y1 + y2) / 2.0
            length = float(np.hypot(x2 - x1, y2 - y1))
            # Left-normal of directed edge n1→n2 (outward from cellL)
            nLx = float(np.sign(y2 - y1))
            nLy = float(np.sign(x1 - x2))
            edge_map[key] = eid
            edges.append([eid, n1, n2, c_left, -1, length, mx, my, nLx, nLy])
            eid += 1
        else:
            # Second cell claiming this edge → it becomes cellR
            edges[edge_map[key]][4] = c_left

    for j in range(y_nodes - 1):
        for i in range(x_nodes - 1):
            c  = cell_id[(i, j)]
            p1 = point_id[(i,     j    )]
            p2 = point_id[(i + 1, j    )]
            p3 = point_id[(i + 1, j + 1)]
            p4 = point_id[(i,     j + 1)]
            add_edge(p1, p2, c)   # bottom
            add_edge(p2, p3, c)   # right
            add_edge(p3, p4, c)   # top
            add_edge(p4, p1, c)   # left

    edges_df = pd.DataFrame(edges, columns=[
        "edgeID", "n1", "n2", "cellL", "cellR",
        "len", "mx", "my", "nLx", "nLy",
    ])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    import os
    os.makedirs(out_dir, exist_ok=True)
    points_df.to_csv(f"{out_dir}/Points.csv", index=False)
    edges_df .to_csv(f"{out_dir}/Edges.csv",  index=False)
    cells_df .to_csv(f"{out_dir}/Cells.csv",  index=False)

    print(f"Mesh generated: {x_nodes-1}×{y_nodes-1} quads  "
          f"({len(points_df)} nodes, {len(edges_df)} edges, {len(cells_df)} cells)")
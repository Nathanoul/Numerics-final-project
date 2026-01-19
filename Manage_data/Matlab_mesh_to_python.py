import pandas as pd
from .shift_ids import shift_ids_df

def Matlab_to_python(points_path, edges_path, tri_path, out_dir):
    edges_df = pd.read_csv(edges_path)
    points_df = pd.read_csv(points_path, header=None)
    cells_df = pd.read_csv(tri_path)

    id_columns_edges = ["edgeID", "n1", "n2", "triL", "triR"]
    edges_df = shift_ids_df(edges_df, id_columns_edges)

    id_columns_tri = ["triID", "v1", "v2", "v3", "nbr12", "nbr23", "nbr31"]
    cells_df = shift_ids_df(cells_df, id_columns_tri)
    cells_df = cells_df.rename(columns={"triID": "cellID",
                                        "v1": "p1", "v2": "p2", "v3": "p3"})

    points_df.insert(0, "pointsID", [i for i in range(len(points_df))])
    points_df.columns = ["pointsID", "x", "y"]
    points_df.to_csv(f"{out_dir}/Points.csv", index=False)

    edges_df.to_csv(f"{out_dir}/Edges.csv", index=False)
    cells_df.to_csv(f"{out_dir}/Cells.csv", index=False)

    return points_df, edges_df, cells_df




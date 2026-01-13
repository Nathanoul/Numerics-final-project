import pandas as pd

def read_data_csv(points_path, edges_path, tri_path):
    points_df = pd.read_csv(points_path)
    edges_df = pd.read_csv(edges_path)
    cells_df = pd.read_csv(tri_path)

    return points_df, edges_df, cells_df
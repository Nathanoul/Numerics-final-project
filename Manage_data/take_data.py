import pandas as pd

def csv_data_to_list(points_path, edges_path, tri_path):
    points_df = pd.read_csv(points_path)
    edges_df = pd.read_csv(edges_path)
    cells_df = pd.read_csv(tri_path)

    points = points_df.to_dict('records')
    edges = edges_df.to_dict('records')
    cells = cells_df.to_dict('records')

    return points, edges, cells
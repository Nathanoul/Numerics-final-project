import pandas as pd

def csv_data_to_dic(points_path, edges_path, tri_path):
    points_df = pd.read_csv(points_path)
    edges_df = pd.read_csv(edges_path)
    cells_df = pd.read_csv(tri_path)

    points = points_df.to_dict('dict')
    edges = edges_df.to_dict('dict')
    cells = cells_df.to_dict('dict')

    return points, edges, cells
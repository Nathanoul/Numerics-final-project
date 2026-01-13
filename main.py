from Mesh_generator.plot_mesh import plot_mesh
from Mesh_generator.read_data import read_data_csv

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # matlab_edges_path = 'Mesh_data/Matlab_mesh/EdgeTable10.csv'
    # matlab_cells_path = 'Mesh_data/Matlab_mesh/TriTable.csv'
    # matlab_points_path = 'Mesh_data/Matlab_mesh/p.csv'
    # out_path = 'Mesh_data/Part2/'
    # points, edges, triangles = Matlab_to_python(matlab_points_path,
    #                                             matlab_edges_path,
    #                                             matlab_cells_path,
    #                                             out_path)

    #generate_square_mesh(10, 10, 1, 1)

    edges_path = "Mesh_data/Part2/Edges.csv"
    cells_path = "Mesh_data/Part2/Cells.csv"
    points_path = "Mesh_data/Part2/Points.csv"
    points_df, edges_df, cells_df = read_data_csv(points_path, edges_path, cells_path)
    plot_mesh(points_df, edges_df)

    p = 1

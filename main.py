from Mesh_generator.plot_mesh import plot_mesh
from Mesh_generator.read_data import read_data_csv
from Mesh_generator.square_mesh_generator import generate_square_mesh
from Matlab_to_python.Matlab_mesh_to_python import Matlab_to_python

if __name__ == '__main__':
    project_directory = "Numerics-final-project/"
    matlab_mesh_path = f"{project_directory}Mesh_data/Matlab_mesh/"
    no_hole_mesh_path = f"{project_directory}Mesh_data/Square_no_hole_mesh/"
    hole_mesh_path = f"{project_directory}Mesh_data/Triangle_hole_mesh/"

    # matlab_edges_path = f'{matlab_mesh_path}EdgeTable10.csv'
    # matlab_cells_path = f'{matlab_mesh_path}TriTable.csv'
    # matlab_points_path = f'{matlab_mesh_path}p.csv'
    # points, edges, triangles = Matlab_to_python(matlab_points_path,
    #                                             matlab_edges_path,
    #                                             matlab_cells_path,
    #                                             out_dir=hole_mesh_path)


    # generate_square_mesh(x_nodes=20, y_nodes=20,
    #                      x_span=[-0.5, 0.5], y_span=[-0.5, 0.5],
    #                      out_dir=no_hole_mesh_path)


    edges_path = f"{hole_mesh_path}Edges.csv"
    cells_path = f"{hole_mesh_path}Cells.csv"
    points_path = f"{hole_mesh_path}Points.csv"
    points_df, edges_df, cells_df = read_data_csv(points_path, edges_path, cells_path)
    plot_mesh(points_df, edges_df)

    p = 1

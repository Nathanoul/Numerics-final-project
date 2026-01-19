from Mesh.plot_mesh import plot_mesh
from Manage_data.read_data import read_data_csv
from Manage_data.Matlab_mesh_to_python import Matlab_to_python
from Mesh.square_mesh_generator import generate_square_mesh

if __name__ == '__main__':
    project_directory = "Numerics-final-project/"
    matlab_mesh_path = f"{project_directory}Mesh_data/Matlab_mesh/"
    no_hole_mesh_path = f"{project_directory}Mesh_data/Square_no_hole_mesh/"
    hole_mesh_path = f"{project_directory}Mesh_data/Triangle_hole_mesh/"
    no_hole_edges_path = f"{no_hole_mesh_path}Edges.csv"
    no_hole_cells_path = f"{no_hole_mesh_path}Cells.csv"
    no_hole_points_path = f"{no_hole_mesh_path}Points.csv"
    hole_edges_path = f"{hole_mesh_path}Edges.csv"
    hole_cells_path = f"{hole_mesh_path}Cells.csv"
    hole_points_path = f"{hole_mesh_path}Points.csv"

    # matlab_edges_path = f'{matlab_mesh_path}EdgeTable10.csv'
    # matlab_cells_path = f'{matlab_mesh_path}TriTable.csv'
    # matlab_points_path = f'{matlab_mesh_path}p.csv'
    # points, edges, triangles = Matlab_to_python(matlab_points_path,
    #                                             matlab_edges_path,
    #                                             matlab_cells_path,
    #                                             out_dir=hole_mesh_path)
    #
    #
    # generate_square_mesh(x_nodes=20, y_nodes=20,
    #                      x_span=[-0.5, 0.5], y_span=[-0.5, 0.5],
    #                      out_dir=no_hole_mesh_path)



    points_df, edges_df, cells_df = read_data_csv(no_hole_points_path, no_hole_edges_path, no_hole_cells_path)
    points = points_df.to_dict('records')
    edges = edges_df.to_dict('records')
    cells = cells_df.to_dict('records')
    plot_mesh(points, edges)


    p = 1

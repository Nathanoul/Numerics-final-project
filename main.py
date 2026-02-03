from Mesh.plot_mesh import plot_mesh
from Manage_data.take_data import csv_data_to_list
from Manage_data.matlab_mesh_to_python import matlab_mesh_to_python
from Mesh.square_mesh_generator import generate_square_mesh

if __name__ == '__main__':
    # project_directory = "Numerics-final-project/"
    project_directory = ""

    matlab_mesh_path = f"{project_directory}Mesh_data/Matlab_mesh/"
    no_hole_mesh_path = f"{project_directory}Mesh_data/Square_no_hole_mesh/"
    hole_mesh_path = f"{project_directory}Mesh_data/Triangle_hole_mesh/"

    matlab_edges_path = f'{matlab_mesh_path}EdgeTable10.csv'
    matlab_cells_path = f'{matlab_mesh_path}TriTable.csv'
    matlab_points_path = f'{matlab_mesh_path}p.csv'
    no_hole_edges_path = f"{no_hole_mesh_path}Edges.csv"
    no_hole_cells_path = f"{no_hole_mesh_path}Cells.csv"
    no_hole_points_path = f"{no_hole_mesh_path}Points.csv"
    hole_edges_path = f"{hole_mesh_path}Edges.csv"
    hole_cells_path = f"{hole_mesh_path}Cells.csv"
    hole_points_path = f"{hole_mesh_path}Points.csv"


    no_hole_points, no_hole_edges, no_hole_cells =\
        csv_data_to_list(no_hole_points_path, no_hole_edges_path, no_hole_cells_path)

    hole_points_df, hole_edges_df, hole_cells_df =\
        read_data_csv(hole_points_path, hole_edges_path, hole_cells_path)
    hole_points = hole_points_df.to_dict('records')
    hole_edges = hole_edges_df.to_dict('records')
    hole_cells = hole_cells_df.to_dict('records')

    plot_mesh(hole_points, hole_edges)

    p = 1

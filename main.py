from Mesh.plot_mesh import plot_mesh
from Manage_data.get_data import csv_data_to_dic
from Manage_data.matlab_mesh_to_python import matlab_mesh_to_python
from Mesh.square_mesh_generator import generate_square_mesh
from Solver.boundary_condtitions import DirichletBC, NeumannBC
from Solver.gauss_seidel_solver import solve_gauss_seidel
from Solver.plotter import plot_steady_state

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
        csv_data_to_dic(no_hole_points_path, no_hole_edges_path, no_hole_cells_path)


    #plot_mesh(no_hole_points, no_hole_edges)

    bottom_bc: NeumannBC = NeumannBC(location = "bottom",
                                         flux_func = lambda x,y: 0,
                                         points = no_hole_points,
                                         edges = no_hole_edges,
                                         cells = no_hole_cells)
    top_bc: NeumannBC = NeumannBC(location = "top",
                                         flux_func = lambda x,y: 0,
                                         points = no_hole_points,
                                         edges = no_hole_edges,
                                         cells = no_hole_cells)
    right_bc: DirichletBC = DirichletBC(location = "right",
                                         value_func = lambda x,y: 1 - 4 * (y + 0.5)**2,
                                         points = no_hole_points,
                                         edges = no_hole_edges,
                                         cells = no_hole_cells)
    left_bc: DirichletBC = DirichletBC(location = "left",
                                         value_func = lambda x,y: 1 - 4 * (y + 0.5)**2,
                                         points = no_hole_points,
                                         edges = no_hole_edges,
                                         cells = no_hole_cells)

    bc = {"bottom bc": bottom_bc,
          "top bc": top_bc,
          "right bc": right_bc,
          "left bc": left_bc,
          "circle bc": None}

    dx, dy = no_hole_edges["len"][0], no_hole_edges["len"][1]

    k1 = 10**-3
    k2 = 100
    solution = solve_gauss_seidel(k1, k2, dx, dy, repetitions = 3, direction = "xy", **bc)

    plot_steady_state(no_hole_points, no_hole_edges, solution)

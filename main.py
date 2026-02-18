from Mesh import *
from Manage_data import *
from Solver import *
import numpy as np

if __name__ == '__main__':
    project_directory = "Numerics-final-project/"
    #project_directory = ""

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


    #generate_square_mesh(128, 128, [- 0.5, 0.5], [- 0.5, 0.5], out_dir=no_hole_mesh_path)


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
                                         value_func = lambda x,y: 1 - 4 * y**2,
                                         points = no_hole_points,
                                         edges = no_hole_edges,
                                         cells = no_hole_cells)
    left_bc: DirichletBC = DirichletBC(location = "left",
                                         value_func = lambda x,y: 1 - 4 * y**2,
                                         points = no_hole_points,
                                         edges = no_hole_edges,
                                         cells = no_hole_cells)
    zero_bottom_bc: DirichletBC = DirichletBC(location = "bottom",
                                         value_func = lambda x,y: 0,
                                         points = no_hole_points,
                                         edges = no_hole_edges,
                                         cells = no_hole_cells)
    zero_top_bc: DirichletBC = DirichletBC(location = "top",
                                         value_func = lambda x,y: 0,
                                         points = no_hole_points,
                                         edges = no_hole_edges,
                                         cells = no_hole_cells)
    zero_right_bc: DirichletBC = DirichletBC(location = "right",
                                         value_func = lambda x,y: 0,
                                         points = no_hole_points,
                                         edges = no_hole_edges,
                                         cells = no_hole_cells)
    zero_left_bc: DirichletBC = DirichletBC(location = "left",
                                         value_func = lambda x,y: 0,
                                         points = no_hole_points,
                                         edges = no_hole_edges,
                                         cells = no_hole_cells)

    bc = {"bottom bc": bottom_bc,
          "top bc": top_bc,
          "right bc": right_bc,
          "left bc": left_bc,
          "circle bc": None}

    zero_bc = {"zero bottom bc": zero_bottom_bc,
               "zero top bc": zero_top_bc,
               "zero right bc": zero_right_bc,
               "zero left bc": zero_left_bc,
               "zero circle bc": None}


    dx, dy = no_hole_edges["len"][0], no_hole_edges["len"][1]

    k1 = 10**-3
    k2 = 100
    #solution, final_rep, error = solve_gauss_seidel(k1, k2, dx, dy, direction = "x", **bc)

    solution, final_rep, error  = solve_multigrid(k1, k2, dx, dy, max_level=4, max_gauss_iter=10, **bc, **zero_bc)

    plot_steady_state(no_hole_points, no_hole_edges, solution.reshape(-1))

from Modules.Mesh import *
from Modules.Data_manager import *
from Modules.Solver import *

import time

import csv

output_file = "solver_runtime_log_2.csv"

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


def mesh2bc():
    no_hole_points, no_hole_edges, no_hole_cells =\
                    csv_data_to_dic(no_hole_points_path, no_hole_edges_path, no_hole_cells_path)
    
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
    zero_bottom_bc: NeumannBC = NeumannBC(location = "bottom",
                                        flux_func = lambda x,y: 0,
                                        points = no_hole_points,
                                        edges = no_hole_edges,
                                        cells = no_hole_cells)
    zero_top_bc: NeumannBC = NeumannBC(location = "top",
                                        flux_func = lambda x,y: 0,
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
    return no_hole_points, no_hole_edges, no_hole_cells, bc, zero_bc
    
def question_1(gird_shape):
    generate_square_mesh(gird_shape, gird_shape, [- 0.5, 0.5], [- 0.5, 0.5], out_dir=no_hole_mesh_path)
    no_hole_points, no_hole_edges, no_hole_cells, bc, zero_bc = mesh2bc()
    
    dx, dy = no_hole_edges["len"][0], no_hole_edges["len"][1]
    k1 = 10e-3
    k2 = 100
    solution, final_rep, error = solve_gauss_seidel(k1, k2, dx, dy, direction = "x", **bc)
    plot_steady_state(no_hole_points, no_hole_edges, solution.reshape(-1),gird_shape=f"{gird_shape}x{gird_shape}-gauss_seidel")
    print(f"final_rep: {final_rep}, error: {error:.6f}")

def question_2(gird_shape):
    generate_square_mesh(gird_shape, gird_shape, [- 0.5, 0.5], [- 0.5, 0.5], out_dir=no_hole_mesh_path)
    no_hole_points, no_hole_edges, no_hole_cells, bc, zero_bc = mesh2bc()
    
    dx, dy = no_hole_edges["len"][0], no_hole_edges["len"][1]
    k1 = 10e-3
    k2 = 100

    #solution, final_rep, error = solve_gauss_seidel(k1, k2, dx, dy, direction = "x", **bc)
    preconditioner: Multigrid_preconditioner = Multigrid_preconditioner(k1, k2, dx, dy, **bc, **zero_bc)

    A = preconditioner.get_A_operator()
    M = preconditioner.get_M_operator()
    
    start_time = time.perf_counter()
    solution, final_rep, error = solve_multigrid(k1, k2, dx, dy, **bc, **zero_bc)
    end_time = time.perf_counter()
    
    print(f"multigrid of {gird_shape} - run time: {end_time - start_time:.6f} s")
    plot_steady_state(no_hole_points, no_hole_edges, solution.reshape(-1),gird_shape=f"{gird_shape}x{gird_shape}_multigridSolver")
    

def question_3(gird_shape):
    generate_square_mesh(gird_shape, gird_shape, [- 0.5, 0.5], [- 0.5, 0.5], out_dir=no_hole_mesh_path)
    no_hole_points, no_hole_edges, no_hole_cells, bc, zero_bc = mesh2bc()

    dx, dy = no_hole_edges["len"][0], no_hole_edges["len"][1]
    k1 = 10e-3
    k2 = 100

    #solution, final_rep, error = solve_gauss_seidel(k1, k2, dx, dy, direction = "x", **bc)
    preconditioner: Multigrid_preconditioner = Multigrid_preconditioner(k1, k2, dx, dy, **bc, **zero_bc)

    A = preconditioner.get_A_operator()
    M = preconditioner.get_M_operator()

    start_time = time.perf_counter()
    solution_bicg, info = solve_bicgstab(A, M, **bc)
    end_time = time.perf_counter()

    print(f"bicg of {gird_shape} - run time: {end_time - start_time:.6f} s")
    plot_steady_state(no_hole_points, no_hole_edges, solution_bicg.reshape(-1),gird_shape=f"{gird_shape}x{gird_shape}_bicgSolver")

def question_4(gird_shape):
    multigrid_runtime_log_list = []
    bicg_runtime_log_list = []
    gird_shape_list = [32,48,64,128,192,256,384,512,768,1024,1536,2048]

    for gird_shape in gird_shape_list:
        generate_square_mesh(gird_shape, gird_shape, [- 0.5, 0.5], [- 0.5, 0.5], out_dir=no_hole_mesh_path)
        no_hole_points, no_hole_edges, no_hole_cells, bc, zero_bc = mesh2bc()
        
        dx, dy = no_hole_edges["len"][0], no_hole_edges["len"][1]
        k1 = 10e-3
        k2 = 100

        #solution, final_rep, error = solve_gauss_seidel(k1, k2, dx, dy, direction = "x", **bc)
        preconditioner: Multigrid_preconditioner = Multigrid_preconditioner(k1, k2, dx, dy, **bc, **zero_bc)

        A = preconditioner.get_A_operator()
        M = preconditioner.get_M_operator()
        
        start_time = time.perf_counter()
        solution, final_rep, error = solve_multigrid(k1, k2, dx, dy, **bc, **zero_bc)
        end_time = time.perf_counter()
        
        print(f"multigrid of {gird_shape} - run time: {end_time - start_time:.6f} s")
        multigrid_runtime_log_list.append(end_time - start_time)
        plot_steady_state(no_hole_points, no_hole_edges, solution.reshape(-1),gird_shape=f"{gird_shape}x{gird_shape}_multigridSolver")

        start_time = time.perf_counter()
        solution_bicg, info = solve_bicgstab(A, M, **bc)
        end_time = time.perf_counter()

        print(f"bicg of {gird_shape} - run time: {end_time - start_time:.6f} s")
        bicg_runtime_log_list.append(end_time - start_time)
        plot_steady_state(no_hole_points, no_hole_edges, solution_bicg.reshape(-1),gird_shape=f"{gird_shape}x{gird_shape}_bicgSolver")


    print("bicg runtime log list:", bicg_runtime_log_list)
    print("multigrid runtime log list:", multigrid_runtime_log_list)

    header = ['Grid_Shape', 'BiCGStab_Time_s', 'Multigrid_Time_s']
    rows = zip(gird_shape_list, bicg_runtime_log_list, multigrid_runtime_log_list)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Runtime log saved to {output_file}")



if __name__ == '__main__':
    question_2(200)


        
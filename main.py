from Mesh import *
from Manage_data import *
from Solver import *
from Animator import *

if __name__ == '__main__':
    """-------------------------------------------------------------------------------------------------
    ----------------------------------Project directories-----------------------------------------------
    -------------------------------------------------------------------------------------------------"""

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


    """-------------------------------------------------------------------------------------------------
    ------------------------------------------PART 1----------------------------------------------------
    -------------------------------------------------------------------------------------------------"""
    """
    #generate_square_mesh(256, 256, [- 0.5, 0.5], [- 0.5, 0.5], out_dir=no_hole_mesh_path)


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


    
    dx, dy = no_hole_edges["len"][0], no_hole_edges["len"][1]
    k1 = 10e-3
    k2 = 100

    solution, final_rep, error = solve_gauss_seidel(k1, k2, dx, dy, direction = "x", **bc)

    solution, final_rep, error  = solve_multigrid(k1, k2, dx, dy, **bc, **zero_bc)

    preconditioner: Multigrid_preconditioner = Multigrid_preconditioner(k1, k2, dx, dy, **bc, **zero_bc)
    A = preconditioner.get_A_operator()
    M = preconditioner.get_M_operator()
    start_time = time.perf_counter()
    solution, info = solve_bicgstab(A, M, **bc)
    solution, final_rep, error = solve_multigrid(k1, k2, dx, dy, **bc, **zero_bc)
    end_time = time.perf_counter()
    print(f"run time: {end_time - start_time:.6f} s")
    plot_steady_state(no_hole_points, no_hole_edges, solution.reshape(-1))
    """

    """-------------------------------------------------------------------------------------------------
    -----------------------------------------------PART 2-----------------------------------------------
    -------------------------------------------------------------------------------------------------"""
    """
    Question 5
    """
    """
    hole_points, hole_edges, hole_cells =\
        csv_data_to_dic(hole_points_path, hole_edges_path, hole_cells_path)

    #plot_mesh(no_hole_points, no_hole_edges)

    bottom_bc: NeumannBC = NeumannBC(location = "bottom",
                                         flux_func = lambda x,y: 0,
                                         points = hole_points,
                                         edges = hole_edges,
                                         cells = hole_cells)
    top_bc: NeumannBC = NeumannBC(location = "top",
                                         flux_func = lambda x,y: 0,
                                         points = hole_points,
                                         edges = hole_edges,
                                         cells = hole_cells)
    right_bc: DirichletBC = DirichletBC(location = "right",
                                         value_func = lambda x,y: 1 - 4 * y**2,
                                         points = hole_points,
                                         edges = hole_edges,
                                         cells = hole_cells)
    left_bc: DirichletBC = DirichletBC(location = "left",
                                         value_func = lambda x,y: 1 - 4 * y**2,
                                         points = hole_points,
                                         edges = hole_edges,
                                         cells = hole_cells)
    circle_bc: DirichletBC = DirichletBC(location = "circle",
                                         value_func = lambda x,y: 2,
                                         points = hole_points,
                                         edges = hole_edges,
                                         cells = hole_cells)
    bc = {"bottom bc": bottom_bc,
          "top bc": top_bc,
          "right bc": right_bc,
          "left bc": left_bc,
          "circle bc": circle_bc}

    k1 = 10e-3
    k2 = 100
    K, rhs_bc, areas, tri_id_to_idx = build_fv_matrix(points=hole_points,
                                                      edges=hole_edges,
                                                      cells=hole_cells, k1=k1, k2=k2,
                                                      **bc)
    results = solve_unsteady_LU(K, rhs_bc, areas, hole_cells, tri_id_to_idx,
                                dt=1e-3,
                                report_times=(1.0, 5.0),
                                tol_steady=1e-6,
                                snapshots_every=50)  # every 50 steps = one frame per 0.05s

    anim = FVAnimation(results, hole_points)
    anim.save("animations/", fmt="gif", fps=20)
    anim.preview(29)
    anim.preview(99)
    anim.preview(-1)
    """

    """
    Question 6
    """

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------

    generate_square_mesh(20, 20, [-0.5, 0.5], [-0.5, 0.5], out_dir=no_hole_mesh_path)

    sq_points, sq_edges, sq_cells =\
        csv_data_to_dic(no_hole_points_path, no_hole_edges_path, no_hole_cells_path)

    # ------------------------------------------------------------------
    # Outer boundary conditions  (circle handled by IBM — no circle_bc here)
    # ------------------------------------------------------------------
    bottom_bc_q6 = NeumannBC(location="bottom", flux_func=lambda x, y: 0,
                             points=sq_points, edges=sq_edges, cells=sq_cells)
    top_bc_q6 = NeumannBC(location="top", flux_func=lambda x, y: 0,
                          points=sq_points, edges=sq_edges, cells=sq_cells)
    right_bc_q6 = DirichletBC(location="right", value_func=lambda x, y: 1 - 4 * y ** 2,
                              points=sq_points, edges=sq_edges, cells=sq_cells)
    left_bc_q6 = DirichletBC(location="left", value_func=lambda x, y: 1 - 4 * y ** 2,
                             points=sq_points, edges=sq_edges, cells=sq_cells)

    bc_q6 = {
        "bottom bc": bottom_bc_q6,
        "top bc": top_bc_q6,
        "right bc": right_bc_q6,
        "left bc": left_bc_q6,
        "circle bc": None,  # circle is imposed via IBM, not as a BC here
    }

    # ------------------------------------------------------------------
    # FV matrix  (outer BCs only — circle interior handled by IBM)
    # ------------------------------------------------------------------
    k1, k2 = 1e-3, 100

    K_q6, rhs_bc_q6, areas_q6, tri_id_to_idx_q6 = build_fv_matrix(
        sq_cells, sq_edges, sq_points, k1, k2, **bc_q6
    )

    # ------------------------------------------------------------------
    # IBM setup
    # ------------------------------------------------------------------
    fluid_indices, ibm_indices = classify_cells(
        sq_cells, tri_id_to_idx_q6, cx_hole=0.0, cy_hole=0.0, R=0.2
    )

    T_ibm = build_ibm_forcing(
        sq_cells, tri_id_to_idx_q6, ibm_indices,
        value_func=lambda x, y: 2.0  # T = 2 on the hole boundary
    )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    results_q6 = solve_ibm_schur_LU(
        K_q6, rhs_bc_q6, areas_q6,
        sq_cells, tri_id_to_idx_q6,
        fluid_indices, ibm_indices, T_ibm,
        dt=1e-3,
        report_times=(1.0, 5.0),
        tol_steady=1e-6,
        snapshots_every=50,
    )

    # ------------------------------------------------------------------
    # Visualise  (identical API to Q5)
    # ------------------------------------------------------------------

    anim_q6 = FVAnimation(results_q6, title="Schur complement")
    anim_q6.save("animations/", filename="q6_ibm", fmt="gif", fps=20)
    anim_q6.preview(29)
    anim_q6.preview(99)
    anim_q6.preview(-1)
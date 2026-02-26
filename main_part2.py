from Modules import *

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

animations_path = f"{project_directory}animations/"

def question_6():
    generate_square_mesh(200, 200, [-0.5, 0.5], [-0.5, 0.5], out_dir=no_hole_mesh_path)

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
        tol_steady=1e-4,
        snapshots_every=50,
    )

    # ------------------------------------------------------------------
    # Visualise  (identical API to Q5)
    # ------------------------------------------------------------------

    anim_q6 = FVAnimation(results_q6, sq_points, title="Schur complement")
    anim_q6.save(animations_path, filename="q6_ibm", fmt="gif", fps=20)
    anim_q6.preview(19)
    anim_q6.preview(99)
    anim_q6.preview(-1)



def question_7():
    # ------------------------------------------------------------------
    # Mesh  (same 200×200 structured square mesh as Q6)
    # ------------------------------------------------------------------
    sq_points, sq_edges, sq_cells = \
        csv_data_to_dic(no_hole_points_path, no_hole_edges_path, no_hole_cells_path)

    # ------------------------------------------------------------------
    # Outer boundary conditions  (identical to Q6)
    # ------------------------------------------------------------------
    bottom_bc_q7 = NeumannBC(location="bottom", flux_func=lambda x, y: 0,
                              points=sq_points, edges=sq_edges, cells=sq_cells)
    top_bc_q7    = NeumannBC(location="top",    flux_func=lambda x, y: 0,
                              points=sq_points, edges=sq_edges, cells=sq_cells)
    right_bc_q7  = DirichletBC(location="right",
                                value_func=lambda x, y: 1 - 4 * y ** 2,
                                points=sq_points, edges=sq_edges, cells=sq_cells)
    left_bc_q7   = DirichletBC(location="left",
                                value_func=lambda x, y: 1 - 4 * y ** 2,
                                points=sq_points, edges=sq_edges, cells=sq_cells)

    bc_q7 = {
        "bottom bc": bottom_bc_q7,
        "top bc":    top_bc_q7,
        "right bc":  right_bc_q7,
        "left bc":   left_bc_q7,
        "circle bc": None,   # hole enforced via explicit IBM (zero flux)
    }

    # ------------------------------------------------------------------
    # FV matrix  (outer BCs only)
    # ------------------------------------------------------------------
    k1, k2 = 1e-3, 100

    K_q7, rhs_bc_q7, areas_q7, tri_id_to_idx_q7 = build_fv_matrix(
        sq_cells, sq_edges, sq_points, k1, k2, **bc_q7
    )

    # ------------------------------------------------------------------
    # IBM cell classification  (same circle as Q6)
    # ------------------------------------------------------------------
    fluid_indices_q7, ibm_indices_q7 = classify_cells(
        sq_cells, tri_id_to_idx_q7, cx_hole=0.0, cy_hole=0.0, R=0.2
    )

    # ------------------------------------------------------------------
    # Solve — EXPLICIT IBM, ADIABATIC hole (zero normal heat flux)
    #
    # Key idea: K_FI is discarded entirely from the fluid equations.
    # This enforces zero flux across the IBM interface by construction.
    # No T_ibm vector is needed (Neumann, not Dirichlet).
    # ------------------------------------------------------------------
    results_q7 = solve_ibm_explicit(
        K_q7, rhs_bc_q7, areas_q7,
        sq_cells, tri_id_to_idx_q7,
        fluid_indices_q7, ibm_indices_q7,
        dt=1e-3,
        report_times=(1.0, 5.0),
        tol_steady=1e-6,
        snapshots_every=50,
    )

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    anim_q7 = FVAnimation(results_q7, sq_points,
                           title="Explicit IBM – adiabatic hole")
    anim_q7.save(animations_path, filename="q7_ibm_explicit", fmt="gif", fps=20)
    anim_q7.preview(19)
    anim_q7.preview(99)
    anim_q7.preview(-1)


if __name__ == '__main__':
    question_7()


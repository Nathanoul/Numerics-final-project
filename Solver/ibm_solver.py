import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu


def solve_ibm_schur_LU(K, rhs_bc, areas, cells, tri_id_to_idx,
                        fluid_indices, ibm_indices, T_ibm,
                        dt=1e-3,
                        report_times=(1.0, 5.0),
                        tol_steady=1e-6,
                        T_init=None,
                        snapshots_every=None):
    """
    Fully implicit IBM with Schur complement (Question 6).

    Formulation
    -----------
    The full backward-Euler system is:

        A [T_F; T_I] = [b_F; b_I]      A = M/dt - K

    where F = fluid cells, I = IBM (inside circle) cells.

    Direct forcing replaces the IBM equations with:

        T_I^{n+1} = T_ibm   (prescribed, constant each step)

    Substituting into the fluid rows gives the Schur complement system:

        A_FF T_F^{n+1} = b_F^n - A_FI T_ibm
                                  ^^^^^^^^^^^
                                  constant correction, precomputed once

    where  A_FF = M_FF/dt - K_FF,  A_FI = -K_FI.

    A_FF is LU-factorised ONCE.  Each time step is one triangular solve.

    Steady state is obtained separately via:

        K_FF T_F = K_FI T_ibm - rhs_bc_F

    using a second LU factorisation of K_FF (also done once).

    Parameters
    ----------
    K : scipy sparse matrix
        FV diffusion matrix from build_fv_matrix.
    rhs_bc : ndarray (N,)
        Boundary RHS from build_fv_matrix.
    areas : ndarray (N,)
        Cell areas (diagonal of M).
    cells : dict
        Cell table from csv_data_to_dic.
    tri_id_to_idx : dict
        cellID → compact index from build_fv_matrix.
    fluid_indices : ndarray (int)
        Compact indices of fluid cells, from classify_cells.
    ibm_indices : ndarray (int)
        Compact indices of IBM cells, from classify_cells.
    T_ibm : ndarray
        Prescribed temperature at each IBM cell, from build_ibm_forcing.
        Must be in the same order as ibm_indices.
    dt : float
        Time step (unconditionally stable, use dt=1e-3 as per Q6).
    report_times : iterable of float
        Times to save  (e.g. t=1, t=5 for Q6).
    tol_steady : float
        Convergence criterion on fluid cells: max|T_F^{n+1} - T_F^n|.
    T_init : ndarray or None
        Initial condition for ALL cells.  Defaults to zeros.
    snapshots_every : int or None
        Save every N steps for animation.

    Returns
    -------
    results : dict  (same format as solve_unsteady_LU)
        "times"        – list[float]
        "snapshots"    – list[ndarray(N)]   full field including IBM cells
        "steady_state" – ndarray(N)         direct LU steady state
        "cx"           – ndarray(N)
        "cy"           – ndarray(N)
        "tri_vertices" – None  (not needed; FVAnimation uses cx/cy)
        "dt"           – float
    """

    K_csr = K.tocsr()
    N     = K_csr.shape[0]

    f_idx = np.asarray(fluid_indices, dtype=int)
    i_idx = np.asarray(ibm_indices,   dtype=int)

    # ------------------------------------------------------------------
    # 1. Centroid arrays for output
    # ------------------------------------------------------------------
    row_ids  = list(cells["cellID"].keys())
    cell_ids = [cells["cellID"][r] for r in row_ids]
    cx_by_idx = np.empty(N)
    cy_by_idx = np.empty(N)
    for r, cid in zip(row_ids, cell_ids):
        idx = tri_id_to_idx[cid]
        cx_by_idx[idx] = cells["cx"][r]
        cy_by_idx[idx] = cells["cy"][r]

    # ------------------------------------------------------------------
    # 2. Extract submatrices
    #    K_FF : fluid × fluid      (drives diffusion between fluid cells)
    #    K_FI : fluid × IBM        (coupling: IBM cells act as sources)
    # ------------------------------------------------------------------
    K_FF = K_csr[f_idx, :][:, f_idx]
    K_FI = K_csr[f_idx, :][:, i_idx]

    # ------------------------------------------------------------------
    # 3. Steady state:  K_FF T_F = K_FI T_ibm - rhs_bc_F
    #    (from  K T + rhs_bc = 0  →  K_FF T_F = -rhs_bc_F - K_FI T_ibm
    #     BUT sign: K_FI has positive off-diag entries, so source from
    #     IBM cells enters as  +K_FI T_ibm on the RHS when moved across)
    # ------------------------------------------------------------------
    print("Computing steady-state via Schur / LU(K_FF) ...")
    rhs_steady_F = K_FI.dot(T_ibm) - rhs_bc[f_idx]
    lu_K_FF      = splu(K_FF.tocsc())
    T_steady_F   = lu_K_FF.solve(rhs_steady_F)

    T_steady             = np.zeros(N)
    T_steady[f_idx]      = T_steady_F
    T_steady[i_idx]      = T_ibm
    print("  Done.")

    # ------------------------------------------------------------------
    # 4. Assemble A_FF = M_FF/dt - K_FF  and LU-factorise ONCE
    #
    #    A_FI = -K_FI   (off-diagonal mass is zero)
    #    constant correction = A_FI T_ibm = -K_FI T_ibm
    # ------------------------------------------------------------------
    M_FF      = diags(areas[f_idx] / dt, format="csr")
    A_FF      = M_FF - K_FF
    correction = -K_FI.dot(T_ibm)          # shape (N_F,), constant every step

    print("LU factorisation of A_FF (Schur complement, fluid block) ...")
    lu_A_FF = splu(A_FF.tocsc())
    print("  Done.  Starting time integration.")

    # ------------------------------------------------------------------
    # 5. Backward-Euler time loop
    #    A_FF T_F^{n+1} = (M_FF/dt) T_F^n + rhs_bc_F - correction
    # ------------------------------------------------------------------
    M_f_dt = areas[f_idx] / dt

    T_full = np.zeros(N) if T_init is None else T_init.copy()
    T_full[i_idx] = T_ibm                   # IBM cells fixed throughout

    times:     list = []
    snapshots: list = []

    def _save(t_val, T_vec):
        times.append(t_val)
        snapshots.append(T_vec.copy())

    report_set   = sorted(set(report_times))
    next_rep_idx = 0
    t    = 0.0
    step = 0

    while True:
        T_F     = T_full[f_idx]
        b_F     = M_f_dt * T_F + rhs_bc[f_idx]
        T_F_new = lu_A_FF.solve(b_F - correction)

        dT_inf = np.max(np.abs(T_F_new - T_F))
        T_full[f_idx] = T_F_new
        t    += dt
        step += 1

        # Periodic snapshots for animation
        if snapshots_every is not None and step % snapshots_every == 0:
            _save(t, T_full)

        # Requested report times
        if next_rep_idx < len(report_set):
            if t >= report_set[next_rep_idx] - 1e-10:
                if not times or abs(times[-1] - report_set[next_rep_idx]) > 1e-10:
                    _save(t, T_full)
                    print(f"  Saved t = {report_set[next_rep_idx]:.3f}  "
                          f"(step {step},  max|dT_F| = {dT_inf:.3e})")
                next_rep_idx += 1

        # Steady-state check (fluid cells only; IBM cells never change)
        if dT_inf < tol_steady:
            if not times or abs(times[-1] - t) > 1e-10:
                _save(t, T_full)
            print(f"  Steady state at t = {t:.5f}  "
                  f"(step {step},  max|dT_F| = {dT_inf:.2e})")
            break

        if step % 100 == 0:
            print(f"  step {step:7d},  t = {t:.4f},  max|dT_F| = {dT_inf:.3e}")

    for t_rep in report_set[next_rep_idx:]:
        _save(t_rep, T_full)

    return {
        "times":        times,
        "snapshots":    snapshots,
        "steady_state": T_steady,
        "cx":           cx_by_idx,
        "cy":           cy_by_idx,
        "tri_vertices": None,       # FVAnimation uses cx/cy; not needed
        "dt":           dt,
    }
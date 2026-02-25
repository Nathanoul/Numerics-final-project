import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu


def solve_unsteady_LU(K, rhs_bc, areas, cells, tri_id_to_idx,
                      dt=1e-3,
                      report_times=(1.0, 5.0),
                      tol_steady=1e-6,
                      T_init=None,
                      snapshots_every=None):
    """
    Integrate  M dT/dt = K T + rhs_bc  using backward-Euler time stepping.

    "Direct LU method" (Question 5): the system matrix

        A = M/dt - K

    is assembled ONCE, LU-factorised ONCE (scipy SuperLU), and then each
    time step is a single cheap forward/back substitution:

        A T^{n+1} = (M/dt) T^n + rhs_bc

    This is unconditionally stable for any dt, which is essential here
    because k2 = 100 makes explicit methods require dt ~ O(1e-6).

    Parameters
    ----------
    K : scipy sparse matrix (lil or csr)
        FV diffusion matrix from build_fv_matrix.
        Sign convention: K[i,i] < 0, K[i,j] > 0 (i != j).
    rhs_bc : np.ndarray, shape (N,)
        Boundary RHS from build_fv_matrix.
    areas : np.ndarray, shape (N,)
        Cell areas  (diagonal of mass matrix M).
    cells : dict
        Cell table from csv_data_to_dic.
    tri_id_to_idx : dict
        cellID -> compact 0-based index from build_fv_matrix.
    dt : float
        Time step.  Unconditionally stable -- choose based on temporal
        accuracy.  dt = 1e-3 as required by Question 5.
    report_times : iterable of float
        Times at which the solution is saved (t = 1 and t = 5 for Q5).
    tol_steady : float
        Steady-state criterion:  max|T^{n+1} - T^n| < tol_steady.
    T_init : np.ndarray or None
        Initial condition (length N).  Defaults to zeros.
    snapshots_every : int or None
        Save a snapshot every this many steps (for animation frames).
        None -> only save at report_times and steady state.

    Returns
    -------
    results : dict
        "times"        - list[float]           one entry per saved snapshot
        "snapshots"    - list[np.ndarray(N)]   T field at each saved time
        "steady_state" - np.ndarray(N)         steady-state via LU(K) T = rhs_bc
        "cx"           - np.ndarray(N)         cell centroid x
        "cy"           - np.ndarray(N)         cell centroid y
        "tri_vertices" - np.ndarray(Ntri, 3)   node indices for Triangulation
        "dt"           - float
    """

    K_csr = K.tocsr()
    N     = K_csr.shape[0]

    # ------------------------------------------------------------------
    # 1. Build centroid and connectivity arrays in compact-index order
    # ------------------------------------------------------------------
    row_ids = list(cells["cellID"].keys())
    tri_ids = [cells["cellID"][r] for r in row_ids]

    cx_by_idx = np.empty(N)
    cy_by_idx = np.empty(N)
    v1_by_idx = np.empty(N, dtype=int)
    v2_by_idx = np.empty(N, dtype=int)
    v3_by_idx = np.empty(N, dtype=int)

    for r, tid in zip(row_ids, tri_ids):
        idx = tri_id_to_idx[tid]
        cx_by_idx[idx] = cells["cx"][r]
        cy_by_idx[idx] = cells["cy"][r]
        v1_by_idx[idx] = int(cells["n1"][r])
        v2_by_idx[idx] = int(cells["n2"][r])
        v3_by_idx[idx] = int(cells["n3"][r])

    tri_vertices = np.stack([v1_by_idx, v2_by_idx, v3_by_idx], axis=1)

    # ------------------------------------------------------------------
    # 2. Steady-state: solve  K T_ss = rhs_bc  directly via LU(K)
    # ------------------------------------------------------------------
    print("Computing steady-state via LU(K) ...")
    lu_K     = splu(K_csr.tocsc())
    # Steady state:  K T + rhs_bc = 0  →  K T = -rhs_bc
    T_steady = lu_K.solve(-rhs_bc)
    print("  Done.")

    # ------------------------------------------------------------------
    # 3. Assemble system matrix  A = M/dt - K  and factorise ONCE.
    #
    #    Backward Euler discretisation of  M dT/dt = K T + rhs_bc:
    #
    #        (M/dt - K) T^{n+1} = (M/dt) T^n + rhs_bc
    #              A T^{n+1}    =       b^n
    #
    #    A is assembled and LU-factored here; each time step is then
    #    just one triangular solve -- O(nnz) work.
    # ------------------------------------------------------------------
    M_over_dt = diags(areas / dt, format="csr")
    A         = M_over_dt - K_csr

    print("LU factorisation of A = M/dt - K ...")
    lu_A = splu(A.tocsc())
    print("  Done.  Starting time integration.")

    # ------------------------------------------------------------------
    # 4. Backward-Euler time loop
    # ------------------------------------------------------------------
    M_dt_vec = areas / dt        # diagonal of M/dt as a plain vector

    T = np.zeros(N) if T_init is None else T_init.copy()

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
        rhs   = M_dt_vec * T + rhs_bc      # build RHS
        T_new = lu_A.solve(rhs)            # one triangular solve per step

        dT_inf = np.max(np.abs(T_new - T))
        T      = T_new
        t     += dt
        step  += 1

        # ---- Periodic snapshots for animation -----------------------
        if snapshots_every is not None and step % snapshots_every == 0:
            _save(t, T)

        # ---- Requested report times --------------------------------
        if next_rep_idx < len(report_set):
            if t >= report_set[next_rep_idx] - 1e-10:
                # avoid duplicate if snapshots_every already saved this step
                if not times or abs(times[-1] - report_set[next_rep_idx]) > 1e-10:
                    _save(t, T)
                    print(f"  Saved t = {report_set[next_rep_idx]:.3f}  "
                          f"(step {step},  max|dT| = {dT_inf:.3e})")
                next_rep_idx += 1

        # ---- Steady-state check ------------------------------------
        if dT_inf < tol_steady:
            if not times or abs(times[-1] - t) > 1e-10:
                _save(t, T)
            print(f"  Steady state at t = {t:.5f}  "
                  f"(step {step},  max|dT| = {dT_inf:.2e})")
            break

        if step % 1000 == 0:
            print(f"  step {step:7d},  t = {t:.4f},  max|dT| = {dT_inf:.3e}")

    # Ensure all report times are present even if loop ended early
    for t_rep in report_set[next_rep_idx:]:
        _save(t_rep, T)

    return {
        "times":        times,
        "snapshots":    snapshots,
        "steady_state": T_steady,
        "cx":           cx_by_idx,
        "cy":           cy_by_idx,
        "tri_vertices": tri_vertices,
        "dt":           dt,
    }
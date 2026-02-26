import numpy as np
from scipy.sparse import diags
from scipy.spatial import cKDTree


def solve_ibm_explicit(K, rhs_bc, areas, cells, tri_id_to_idx,
                       fluid_indices, ibm_indices,
                       dt=1e-3,
                       report_times=(1.0, 5.0),
                       tol_steady=1e-6,
                       T_init=None,
                       snapshots_every=None):
    """
    Explicit direct-forcing IBM for an INSULATED (zero normal heat flux)
    circular hole boundary — Question 7.

    Formulation
    -----------
    The governing PDE is discretised with a backward-Euler (implicit
    diffusion) scheme on ALL cells, but the IBM enforcement is done
    EXPLICITLY, i.e. using the previous time step's temperature to
    prescribe IBM values before solving the fluid system.

    Zero normal heat flux (Neumann, adiabatic) on the hole surface is
    enforced by the "ghost-cell / mirror" technique:

        T_I^n  =  T_{F,nearest}^{n-1}

    where T_{F,nearest} is the temperature of the closest fluid cell
    centroid to each IBM cell centroid.  This sets ∂T/∂n ≈ 0 across
    the IBM interface at the previous time step.

    The fluid system solved each step is:

        (M_FF/dt - K_FF) T_F^{n+1} = (M_FF/dt) T_F^n
                                      + rhs_bc_F
                                      - K_FI T_I^n          (explicit)

    Because T_I^n changes every step (it mirrors the evolving fluid
    field), the right-hand side is updated each step — unlike Q6 where
    T_ibm was constant and the correction could be precomputed.

    The system matrix A_FF = M_FF/dt - K_FF is CONSTANT, so it is
    LU-factorised ONCE for efficiency.

    Steady state
    ------------
    Obtained by running the time loop until convergence, then also
    solving the direct steady-state system:

        K_FF T_F = -K_FI T_I^ss - rhs_bc_F

    where T_I^ss is derived from the converged fluid field (same
    mirror rule applied to the steady solution).

    Parameters
    ----------
    K : scipy sparse matrix
        FV diffusion matrix from build_fv_matrix.
    rhs_bc : ndarray (N,)
        Boundary RHS from build_fv_matrix.
    areas : ndarray (N,)
        Cell areas.
    cells : dict
        Cell table from csv_data_to_dic.
    tri_id_to_idx : dict
        cellID → compact index from build_fv_matrix.
    fluid_indices : ndarray (int)
        Compact indices of fluid cells.
    ibm_indices : ndarray (int)
        Compact indices of IBM cells.
    dt : float
        Time step.
    report_times : iterable of float
        Times at which to save snapshots.
    tol_steady : float
        Convergence criterion: max|T_F^{n+1} - T_F^n| < tol_steady.
    T_init : ndarray or None
        Initial condition.  Defaults to zeros.
    snapshots_every : int or None
        Save a snapshot every N steps for animation.

    Returns
    -------
    results : dict
        "times"        – list[float]
        "snapshots"    – list[ndarray(N)]
        "steady_state" – ndarray(N)
        "cx"           – ndarray(N)
        "cy"           – ndarray(N)
        "tri_vertices" – None
        "dt"           – float
    """
    from scipy.sparse.linalg import splu

    K_csr = K.tocsr()
    N     = K_csr.shape[0]

    f_idx = np.asarray(fluid_indices, dtype=int)
    i_idx = np.asarray(ibm_indices,   dtype=int)

    # ------------------------------------------------------------------
    # 1. Centroid arrays
    # ------------------------------------------------------------------
    row_ids  = list(cells["cellID"].keys())
    cell_ids = [cells["cellID"][r] for r in row_ids]
    cx_all = np.empty(N)
    cy_all = np.empty(N)
    for r, cid in zip(row_ids, cell_ids):
        idx = tri_id_to_idx[cid]
        cx_all[idx] = cells["cx"][r]
        cy_all[idx] = cells["cy"][r]

    # ------------------------------------------------------------------
    # 2. Build KD-tree over FLUID centroids for mirror-cell lookup
    #
    #    For each IBM cell we find the nearest FLUID cell centroid.
    #    This defines the explicit "mirror" to enforce zero normal flux.
    # ------------------------------------------------------------------
    fluid_pts   = np.column_stack([cx_all[f_idx], cy_all[f_idx]])
    ibm_pts     = np.column_stack([cx_all[i_idx], cy_all[i_idx]])

    tree        = cKDTree(fluid_pts)
    _, nn_local = tree.query(ibm_pts, k=1)  # nn_local[j] = index in f_idx
    mirror_f_idx = nn_local                 # local index into f_idx array

    print(f"Mirror-cell mapping built:  {len(i_idx)} IBM cells "
          f"→ {len(np.unique(mirror_f_idx))} unique fluid mirrors.")

    # ------------------------------------------------------------------
    # 3. Extract submatrices
    # ------------------------------------------------------------------
    K_FF = K_csr[f_idx, :][:, f_idx]
    K_FI = K_csr[f_idx, :][:, i_idx]

    # ------------------------------------------------------------------
    # 4. LU factorisation of A_FF  (done ONCE — matrix is constant)
    # ------------------------------------------------------------------
    M_FF  = diags(areas[f_idx] / dt, format="csr")
    A_FF  = M_FF - K_FF
    M_f_dt = areas[f_idx] / dt

    print("LU factorisation of A_FF (explicit IBM, fluid block) ...")
    lu_A_FF = splu(A_FF.tocsc())
    print("  Done.  Starting time integration.")

    # ------------------------------------------------------------------
    # 5. Initial condition
    # ------------------------------------------------------------------
    T_full = np.zeros(N) if T_init is None else T_init.copy()

    # Initialise IBM cells by mirroring the (zero) initial fluid field
    T_full[i_idx] = T_full[f_idx][mirror_f_idx]

    times:     list = []
    snapshots: list = []

    def _save(t_val, T_vec):
        times.append(t_val)
        snapshots.append(T_vec.copy())

    report_set   = sorted(set(report_times))
    next_rep_idx = 0
    t    = 0.0
    step = 0

    # ------------------------------------------------------------------
    # 6. Time loop
    # ------------------------------------------------------------------
    while True:
        T_F = T_full[f_idx]

        # --- Explicit IBM: set IBM temperatures from PREVIOUS fluid field
        T_I_explicit       = T_full[f_idx][mirror_f_idx]   # shape (N_I,)
        T_full[i_idx]      = T_I_explicit

        # --- Compute explicit correction (changes every step)
        correction = K_FI.dot(T_I_explicit)   # subtract from RHS

        # --- Build RHS and solve
        b_F     = M_f_dt * T_F + rhs_bc[f_idx] - correction
        T_F_new = lu_A_FF.solve(b_F)

        dT_inf = np.max(np.abs(T_F_new - T_F))
        T_full[f_idx] = T_F_new
        t    += dt
        step += 1

        # Periodic animation snapshots
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

        # Convergence check
        if dT_inf < tol_steady:
            if not times or abs(times[-1] - t) > 1e-10:
                _save(t, T_full)
            print(f"  Steady state at t = {t:.5f}  "
                  f"(step {step},  max|dT_F| = {dT_inf:.2e})")
            break

        if step % 1000 == 0:
            print(f"  step {step:7d},  t = {t:.4f},  max|dT_F| = {dT_inf:.3e}")

    # Save any remaining requested report times
    for t_rep in report_set[next_rep_idx:]:
        _save(t_rep, T_full)

    # ------------------------------------------------------------------
    # 7. Steady-state solution (direct solve using converged IBM values)
    # ------------------------------------------------------------------
    print("Computing steady-state via direct LU(K_FF) ...")
    T_I_ss        = T_full[f_idx][mirror_f_idx]
    rhs_steady_F  = -K_FI.dot(T_I_ss) - rhs_bc[f_idx]
    from scipy.sparse.linalg import spsolve
    T_steady_F    = spsolve(K_FF.tocsc(), rhs_steady_F)

    T_steady        = np.zeros(N)
    T_steady[f_idx] = T_steady_F
    T_steady[i_idx] = T_I_ss
    print("  Done.")

    return {
        "times":        times,
        "snapshots":    snapshots,
        "steady_state": T_steady,
        "cx":           cx_all,
        "cy":           cy_all,
        "tri_vertices": None,
        "dt":           dt,
    }

import numpy as np
from scipy.sparse import lil_matrix


def get_conductivity(x, y, k1, k2):
    """
    Checkerboard conductivity: k1 where x·y ≥ 0, k2 where x·y < 0.
    Matches the four-quadrant layout in Fig. 1 of the project.
    """
    return k2 if x>=0 and y>=0 else k1


def _harmonic_mean(a, b):
    return 2.0 * a * b / (a + b) if (a + b) != 0 else 0.0


def _cell_area(cells, row_id, points):
    """
    Compute triangle area from its three vertices using the cross-product formula.
    |2A| = |(x2-x1)(y3-y1) - (y2-y1)(x3-x1)|
    """
    px = points["x"]
    py = points["y"]
    v1 = cells["n1"][row_id]
    v2 = cells["n2"][row_id]
    v3 = cells["n3"][row_id]
    x1, y1 = px[v1], py[v1]
    x2, y2 = px[v2], py[v2]
    x3, y3 = px[v3], py[v3]
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))


def build_fv_matrix(cells, edges, points, k1, k2, **boundary_conditions):
    """
    Assemble the FV diffusion matrix K and boundary RHS vector for:

        ∇·(k ∇T) = 0   (steady),  or  ∂T/∂t = ∇·(k ∇T)  (unsteady).

    The caller is responsible for time-stepping (e.g. forming  M/dt - K
    and solving the linear system with LU).

    Non-orthogonality on the triangular mesh is handled with the
    Minimum Correction Method (MCM, as in Lecture 2 recitation).

    The sign convention for K follows FV:
        ∑_f  Flux_{i,f} = 0   →   K T = 0  at steady state.
    Diagonal entries K[i,i] < 0,  off-diagonal K[i,j] > 0.

    Parameters
    ----------
    cells : dict
        Triangle (cell) table from csv_data_to_dic.
        Required columns: triID, cx, cy, v1, v2, v3
                          (nbr* and n*x/y are NOT used – we use the edge table).
    edges : dict
        Unique edge table from csv_data_to_dic.
        Required columns: edgeID, triL, triR, len, mx, my, nLx, nLy
    points : dict
        Node table from csv_data_to_dic.  Required columns: x, y
    k1, k2 : float
        Thermal conductivities for the two checkerboard regions.
    dirichlet_bcs : list of DirichletBC
        Each must have  contains_midpoint(mx, my)  and
        get_value_at_midpoint(mx, my)  (add these via bc_additions.py).
    neumann_bcs : list of NeumannBC
        Each must have  contains_midpoint(mx, my)  and
        get_flux_at_midpoint(mx, my)  (add these via bc_additions.py).

    Returns
    -------
    K : scipy lil_matrix, shape (N, N)
        FV diffusion matrix (convert to csr for solving).
    rhs_bc : np.ndarray, shape (N,)
        Boundary contribution to the RHS coming from Dirichlet BCs.
        For Neumann BCs the flux is already embedded in K via the
        boundary face flux (zero flux → no contribution; non-zero flux
        is added to rhs_bc).
    areas : np.ndarray, shape (N,)
        Cell areas (diagonal of mass matrix M).
    tri_id_to_idx : dict
        Maps triID (from the CSV) to the 0-based row/column index in K.
    """

    # ------------------------------------------------------------------
    # 1. Index map:  triID → compact 0-based index
    # ------------------------------------------------------------------
    row_ids = list(cells["cellID"].keys())
    tri_ids = [cells["cellID"][r] for r in row_ids]
    tri_id_to_idx = {tid: idx for idx, tid in enumerate(tri_ids)}
    N = len(tri_ids)

    # Cell centroids (indexed by compact idx)
    cx_arr = np.array([cells["cx"][r] for r in row_ids])
    cy_arr = np.array([cells["cy"][r] for r in row_ids])

    # Cell areas (needed by the caller for M)
    areas = np.array([_cell_area(cells, r, points) for r in row_ids])

    # ------------------------------------------------------------------
    # 2. Allocate K and rhs_bc
    # ------------------------------------------------------------------
    K      = lil_matrix((N, N), dtype=float)
    rhs_bc = np.zeros(N)

    # ------------------------------------------------------------------
    # 3. Edge-based assembly loop
    # ------------------------------------------------------------------
    for r in edges["edgeID"].keys():

        triL_id = int(edges["cellL"][r])
        triR_id = int(edges["cellR"][r])   # 0 → boundary edge
        Lf      = edges["len"][r]
        mx      = edges["mx"] [r]
        my      = edges["my"] [r]
        nLx     = edges["nLx"][r]         # outward unit normal w.r.t. triL
        nLy     = edges["nLy"][r]

        i    = tri_id_to_idx[triL_id]
        cx_i = cx_arr[i];  cy_i = cy_arr[i]
        n_f  = np.array([nLx, nLy])       # outward from cell i

        # ==============================================================
        # INTERIOR EDGE
        # ==============================================================
        if triR_id != -1:
            j    = tri_id_to_idx[triR_id]
            cx_j = cx_arr[j];  cy_j = cy_arr[j]

            # Face conductivity: harmonic mean of the two cells
            ki  = get_conductivity(cx_i, cy_i, k1, k2)
            kj  = get_conductivity(cx_j, cy_j, k1, k2)
            k_f = _harmonic_mean(ki, kj)

            # Centroid-to-centroid vector and its projection on n_f
            d_ij   = np.array([cx_j - cx_i, cy_j - cy_i])
            d_n    = np.dot(d_ij, n_f)          # scalar: |d_ij| cos θ

            # ---- Minimum Correction Method (MCM) ---------------------
            #
            #  Decompose  n_f  =  alpha · d_ij  +  t
            #  where  alpha = d_n / |d_ij|²   (so alpha·d_ij is the
            #  component of n_f parallel to d_ij).
            #
            #  Orthogonal flux (two-point):
            #    F_ortho = k_f · Lf · alpha · (T_j - T_i)
            #
            #  Non-orthogonal correction vector:
            #    t = n_f - alpha · d_ij
            #
            #  The MCM approximation of  ∇T · t  at the face uses the
            #  same two-point difference rescaled along d_ij:
            #    ∇T · t  ≈  (T_j - T_i) / |d_ij| · (d̂ · t)
            #
            #  Combined coefficient:
            #    coeff = k_f · Lf · (alpha  +  (d̂ · t) / |d_ij|)
            #          = k_f · Lf · d_n / |d_ij|²            [1]
            #    (the two terms combine to  d_n / |d_ij|²; see below)
            #
            #  Derivation of [1]:
            #    alpha + (d̂·t)/|d_ij|
            #    = d_n/|d_ij|²  +  (d_ij/|d_ij|)·(n_f - alpha·d_ij) / |d_ij|
            #    = d_n/|d_ij|²  +  (d_n/|d_ij|² - alpha·|d_ij|/|d_ij|)/1
            #    ... simplifies to  d_n / |d_ij|²
            #
            d_ij_sq = np.dot(d_ij, d_ij)
            # d_n should be positive: centroid of triR lies in the direction
            # of n_f (outward from triL).  A negative value signals an
            # inconsistent normal in the mesh; abs() keeps the matrix
            # diagonally dominant regardless.
            coeff   = k_f * Lf * abs(d_n) / d_ij_sq

            # Contribution to cell i and cell j (equal-and-opposite)
            K[i, i] -= coeff
            K[i, j] += coeff
            K[j, j] -= coeff
            K[j, i] += coeff

        # ==============================================================
        # BOUNDARY EDGE
        # ==============================================================
        else:
            k_f = get_conductivity(cx_i, cy_i, k1, k2)

            # Distance from cell centroid to face midpoint along n_f.
            # Must be positive (midpoint lies outside the cell).
            # If the stored normal points slightly inward on some boundary
            # faces (can happen in body-fitted meshes), d_if becomes
            # negative and flips all flux signs -- clamp with abs().
            m_f  = np.array([mx, my])
            c_i  = np.array([cx_i, cy_i])
            d_if = abs(np.dot(m_f - c_i, n_f))

            # ---- Check Dirichlet BCs ---------------------------------
            for bc_name, bc in boundary_conditions.items():
                if bc.type=="Dirichlet":
                    if bc.contains_midpoint(mx, my):
                        T_b   = bc.get_value_at_midpoint(mx, my)
                        coeff = k_f * Lf / d_if
                        # Flux = coeff * (T_b - T_i)
                        #   → moves coeff * T_i to LHS (K diagonal)
                        #   → moves coeff * T_b to RHS
                        K[i, i] -= coeff
                        rhs_bc[i] += coeff * T_b
                        break
                elif bc.type=="Neumann":
                    if bc.contains_midpoint(mx, my):
                        g = bc.get_flux_at_midpoint(mx, my)
                        # Prescribed normal flux: Flux = g · Lf
                        # Zero flux → nothing to do; non-zero → add to RHS
                        if g != 0.0:
                            rhs_bc[i] += g * Lf
                        break

            # Unmatched boundary edges are silently skipped
            # (can happen for internal artefacts in some mesh generators)

    return K, rhs_bc, areas, tri_id_to_idx
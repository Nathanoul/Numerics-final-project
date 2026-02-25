import numpy as np
from scipy.sparse import lil_matrix


def get_conductivity(x, y, k1, k2):
    """
    k2 (high conductivity) in upper-right quadrant (x>=0, y>=0).
    k1 everywhere else.  Matches Fig. 1 of the project.
    """
    return k2 if x >= 0 and y >= 0 else k1


def _harmonic_mean(a, b):
    return 2.0 * a * b / (a + b) if (a + b) != 0 else 0.0


def _cell_area(cells, row_id, points):
    """
    Area via cross-product (triangles) or shoelace (quads).
    Handles both triangular (n1,n2,n3) and quad (n1,n2,n3,n4) cells.
    """
    px = points["x"]
    py = points["y"]
    n1 = cells["n1"][row_id]
    n2 = cells["n2"][row_id]
    n3 = cells["n3"][row_id]
    x1, y1 = px[n1], py[n1]
    x2, y2 = px[n2], py[n2]
    x3, y3 = px[n3], py[n3]

    if "n4" in cells:
        # Quad: shoelace formula over all four vertices
        n4 = cells["n4"][row_id]
        x4, y4 = px[n4], py[n4]
        return 0.5 * abs(
            (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) +
            (x3 * y4 - x4 * y3) + (x4 * y1 - x1 * y4)
        )

    # Triangle: cross-product
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))


def build_fv_matrix(cells, edges, points, k1, k2, **boundary_conditions):
    """
    Assemble FV diffusion matrix K and boundary RHS for:

        ∂T/∂t = ∇·(k ∇T)    (or steady: ∇·(k ∇T) = 0)

    Works on both triangular (Q5 MATLAB mesh) and quad (Q6 square mesh) grids.
    Non-orthogonality handled via Minimum Correction Method (MCM).

    Sign convention:  K[i,i] < 0,  K[i,j] > 0  (i≠j)
    Steady state:     K T = -rhs_bc   →   K T + rhs_bc = 0

    Parameters
    ----------
    cells, edges, points : dicts from csv_data_to_dic
    k1, k2 : conductivities
    **boundary_conditions : DirichletBC / NeumannBC instances
        Each must expose  .type, .contains_midpoint(mx,my),
        .get_value_at_midpoint(mx,my) / .get_flux_at_midpoint(mx,my)

    Returns
    -------
    K          : lil_matrix (N×N)
    rhs_bc     : ndarray (N,)
    areas      : ndarray (N,)   cell areas (diagonal of mass matrix M)
    tri_id_to_idx : dict  cellID → compact 0-based index
    """

    # ------------------------------------------------------------------
    # 1. Index map  cellID → 0-based compact index
    # ------------------------------------------------------------------
    row_ids = list(cells["cellID"].keys())
    cell_ids = [cells["cellID"][r] for r in row_ids]
    tri_id_to_idx = {cid: idx for idx, cid in enumerate(cell_ids)}
    N = len(cell_ids)

    cx_arr = np.array([cells["cx"][r] for r in row_ids])
    cy_arr = np.array([cells["cy"][r] for r in row_ids])
    areas  = np.array([_cell_area(cells, r, points) for r in row_ids])

    # ------------------------------------------------------------------
    # 2. Allocate
    # ------------------------------------------------------------------
    K      = lil_matrix((N, N), dtype=float)
    rhs_bc = np.zeros(N)

    # ------------------------------------------------------------------
    # 3. Edge loop
    # ------------------------------------------------------------------
    for r in edges["edgeID"].keys():

        cellL_id = int(edges["cellL"][r])
        cellR_id = int(edges["cellR"][r])   # -1 → boundary edge
        n1 = int(edges["n1"][r])
        n2 = int(edges["n2"][r])
        Lf  = edges["len"][r]
        mx  = edges["mx"] [r]
        my  = edges["my"] [r]
        nLx = edges["nLx"][r]
        nLy = edges["nLy"][r]

        i    = tri_id_to_idx[cellL_id]
        cx_i = cx_arr[i];  cy_i = cy_arr[i]
        n_f  = np.array([nLx, nLy])            # outward from cellL

        # ==============================================================
        # INTERIOR EDGE
        # ==============================================================
        if cellR_id != -1:
            j    = tri_id_to_idx[cellR_id]
            cx_j = cx_arr[j];  cy_j = cy_arr[j]

            ki  = get_conductivity(cx_i, cy_i, k1, k2)
            kj  = get_conductivity(cx_j, cy_j, k1, k2)
            k_f = _harmonic_mean(ki, kj)

            d_ij    = np.array([cx_j - cx_i, cy_j - cy_i])
            d_n     = np.dot(d_ij, n_f)
            d_ij_sq = np.dot(d_ij, d_ij)

            # MCM: combined coeff = k_f * Lf * d_n / |d_ij|²
            # abs(d_n) guards against inconsistent normals in unstructured mesh
            coeff = k_f * Lf * abs(d_n) / d_ij_sq

            K[i, i] -= coeff;  K[i, j] += coeff
            K[j, j] -= coeff;  K[j, i] += coeff

        # ==============================================================
        # BOUNDARY EDGE
        # ==============================================================
        else:
            k_f = get_conductivity(cx_i, cy_i, k1, k2)

            m_f  = np.array([mx, my])
            c_i  = np.array([cx_i, cy_i])
            # abs: guard against normals pointing slightly inward
            d_if = abs(np.dot(m_f - c_i, n_f))

            for bc_name, bc in boundary_conditions.items():
                if bc is None:
                    continue
                if bc.type == "Dirichlet" and bc.on_boundary(n1) and bc.on_boundary(n2):
                    T_b   = bc.get_value_at_midpoint(mx, my)
                    coeff = k_f * Lf / d_if
                    K[i, i]   -= coeff
                    rhs_bc[i] += coeff * T_b
                    break
                elif bc.type == "Neumann" and bc.on_boundary(n1) and bc.on_boundary(n2):
                    g = bc.get_flux_at_midpoint(mx, my)
                    # g = ∂T/∂n;  flux contribution = k_f * g * Lf
                    if g != 0.0:
                        rhs_bc[i] += k_f * g * Lf
                    break

    return K, rhs_bc, areas, tri_id_to_idx


def pin_reference_cell(K, rhs_bc, tri_id_to_idx, pin_tid=None, T_ref=0.0):
    """
    Pin one cell to T_ref.  Fixes the null-space of a pure-Neumann system.
    Modifies K and rhs_bc in-place.
    """
    pin_idx = 0 if pin_tid is None else tri_id_to_idx[pin_tid]
    K[pin_idx, :]       = 0.0
    K[pin_idx, pin_idx] = 1.0
    rhs_bc[pin_idx]     = T_ref
    return K, rhs_bc
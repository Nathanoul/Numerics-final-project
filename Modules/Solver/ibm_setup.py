import numpy as np
from scipy.spatial import cKDTree

def classify_cells(cells, tri_id_to_idx, cx_hole=0.0, cy_hole=0.0, R=0.2):
    """
    Partition all cells into FLUID (outside circle) and IBM (inside circle).

    A cell is IBM when its centroid lies inside or on the circle:
        (cx - cx_hole)² + (cy - cy_hole)² ≤ R²

    This is the direct-forcing IBM strategy: IBM cells are not solved —
    their temperature is simply prescribed by the boundary condition.

    Parameters
    ----------
    cells : dict
        Cell table from csv_data_to_dic.
    tri_id_to_idx : dict
        cellID → compact index, from build_fv_matrix.
    cx_hole, cy_hole : float
        Circle centre coordinates.
    R : float
        Circle radius.

    Returns
    -------
    fluid_indices : np.ndarray (int)
        Compact indices of fluid cells (outside circle).
    ibm_indices : np.ndarray (int)
        Compact indices of IBM cells (inside / on circle boundary).
    """
    row_ids  = list(cells["cellID"].keys())
    cell_ids = [cells["cellID"][r] for r in row_ids]

    fluid_indices = []
    ibm_indices   = []

    for r, cid in zip(row_ids, cell_ids):
        idx = tri_id_to_idx[cid]
        cx  = cells["cx"][r]
        cy  = cells["cy"][r]
        if (cx - cx_hole) ** 2 + (cy - cy_hole) ** 2 <= R ** 2:
            ibm_indices.append(idx)
        else:
            fluid_indices.append(idx)

    fluid_indices = np.array(fluid_indices, dtype=int)
    ibm_indices   = np.array(ibm_indices,   dtype=int)

    print(f"IBM classification:  {len(fluid_indices)} fluid cells, "
          f"{len(ibm_indices)} IBM cells  "
          f"(total {len(fluid_indices) + len(ibm_indices)})")
    return fluid_indices, ibm_indices


def build_ibm_forcing(cells, tri_id_to_idx, ibm_indices, value_func=lambda x, y: 2.0):
    """
    Compute the prescribed temperature at each IBM cell.

    The value is evaluated at the cell centroid.  For a constant BC
    (T=2 on the circle) this just fills a vector of 2s, but the interface
    supports spatially varying BCs for generality.

    Parameters
    ----------
    cells : dict
    tri_id_to_idx : dict
    ibm_indices : np.ndarray
        Compact indices of IBM cells, from classify_cells.
    value_func : callable  (x, y) → float
        Prescribed temperature at position (x, y).

    Returns
    -------
    T_ibm : np.ndarray, shape (len(ibm_indices),)
        Prescribed temperature for each IBM cell in the same order
        as ibm_indices.
    """
    # Build reverse map: compact idx → CSV row_id
    row_ids  = list(cells["cellID"].keys())
    cell_ids = [cells["cellID"][r] for r in row_ids]
    idx_to_row = {tri_id_to_idx[cid]: r for r, cid in zip(row_ids, cell_ids)}

    T_ibm = np.array([
        value_func(cells["cx"][idx_to_row[idx]],
                   cells["cy"][idx_to_row[idx]])
        for idx in ibm_indices
    ])
    return T_ibm


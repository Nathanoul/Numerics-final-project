from scipy.sparse.linalg import bicgstab
from numpy import zeros

def solve_bicgstab(A, M, b=0,tol=1e-3, **boundary_condition):
    if b == 0:
        num_x = boundary_condition["top bc"].bc_points_num
        num_y = boundary_condition["right bc"].bc_points_num
        b = zeros((num_y*num_x,))

    for bc_name, bc in boundary_condition.items():
        if bc is not None:
            b = bc.apply_rhs(b)

    return bicgstab(A=A, b=b, M=M, rtol=tol)
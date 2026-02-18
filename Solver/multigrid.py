import numpy as np
from .thomas_matrix_solver import thomas_solver
from .general_functions import *
from .gauss_seidel_solver import solve_gauss_seidel

def restrict(r):
    return 0.25*(r[::2,::2]
        +r[1::2,::2]
        +r[::2,1::2]
        +r[1::2,1::2])

def prolong(e):
    num_x, num_y = e.shape
    fine = np.zeros((2*num_x, 2*num_y))
    fine[::2,::2] = e
    fine[1::2,::2] = e
    fine[::2,1::2] = e
    fine[1::2,1::2] = e
    return fine

def residual(T, k1, k2, dx, dy):
    num_y, num_x = T.shape
    r=np.zeros_like(T)

    for j in range(1, num_x - 1):
        for i in range(1, num_y - 1):

            k_up, k_down, k_left, k_right, k_center_x, k_center_y = get_k_star(i, j, num_x, num_y, k1, k2)

            r[i,j]=(k_right * (T[i+1,j] - T[i,j])
                   -k_left * (T[i,j] - T[i-1,j])) / dx**2 \
                 +(k_up * (T[i,j+1] - T[i,j])
                   -k_down * (T[i,j] - T[i,j-1])) / dy**2

    return r

def vcycle(T, k1, k2, dx, dy, source=None, level=0, max_level=1, **boundary_condition):

    #update bc to fit the size
    for bc_name, bc in boundary_condition.items():
        if bc is not None:
            bc.resize_for_square_mesh(*T.shape)

    T, *_ = solve_gauss_seidel(k1, k2, dx, dy, direction="x", max_rep=1, init_guess=T, source=source, **boundary_condition)

    r = residual(T, k1, k2, dx, dy)

    if level==max_level:
        return T

    rc = restrict(r)
    ec = np.zeros_like(rc)

    ec = vcycle(ec, k1, k2, 2*dx, 2*dy, level=level+1, max_level=max_level, source=rc, **boundary_condition)

    T += prolong(ec)

    # update bc to fit the size once more
    for bc_name, bc in boundary_condition.items():
        if bc is not None:
            bc.resize_for_square_mesh(*T.shape)

    T, *_ = solve_gauss_seidel(k1, k2, dx, dy, direction="y", max_rep=1, init_guess=T, source=source, **boundary_condition)

    return T


def multigrid_solver(k1, k2, dx, dy, max_level, init_guess=None, max_rep = 0, tol=10**-3, **boundary_condition):
    top_bc = boundary_condition["top bc"]
    bottom_bc = boundary_condition["bottom bc"]
    right_bc = boundary_condition["right bc"]
    left_bc = boundary_condition["left bc"]

    num_x = len(bottom_bc.boundary_points_ids)
    num_y = len(right_bc.boundary_points_ids)

    if init_guess is None:
        init_guess = np.zeros((num_y, num_x))

    old_solution = init_guess
    max_error = 1
    rep = 0

    solution = old_solution.copy()
    while max_error >= tol and (max_rep == 0 or rep < max_rep):
        solution = vcycle(solution, k1, k2, dx, dy, max_level=max_level, **boundary_condition)

        error = abs(old_solution - solution)
        max_error = max(max(row) for row in error)
        old_solution = solution.copy()

        rep += 1
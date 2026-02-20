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
    num_y, num_x = e.shape
    fine = np.zeros((2 * num_y, 2 * num_x))

    # Inject coarse values at even indices
    fine[::2, ::2] = e

    # Interpolate along x (only interior odd columns)
    fine[::2, 1:-1:2] = 0.5 * (e[:, :-1] + e[:, 1:])
    # Copy boundary column
    fine[::2, -1] = e[:, -1]

    # Interpolate along y (only interior odd rows)
    fine[1:-1:2, ::2] = 0.5 * (e[:-1, :] + e[1:, :])
    # Copy boundary row
    fine[-1, ::2] = e[-1, :]

    # Interpolate at interior cell centers
    fine[1:-1:2, 1:-1:2] = 0.25 * (e[:-1, :-1] + e[:-1, 1:] + e[1:, :-1] + e[1:, 1:])
    # Copy boundary corners/edges
    fine[-1, 1:-1:2] = 0.5 * (e[-1, :-1] + e[-1, 1:])
    fine[1:-1:2, -1] = 0.5 * (e[:-1, -1] + e[1:, -1])
    fine[-1, -1] = e[-1, -1]

    return fine

def residual(T, k1, k2, dx, dy):
    num_y, num_x = T.shape
    r=np.zeros_like(T)

    for j in range(1, num_x - 1):
        for i in range(1, num_y - 1):

            k_up, k_down, k_left, k_right, k_center_x, k_center_y = get_k_star(i, j, num_x, num_y, k1, k2)

            r[i,j]=(k_up * (T[i+1,j] - T[i,j])
                   -k_down * (T[i,j] - T[i-1,j])) / dy**2 \
                 +(k_right * (T[i,j+1] - T[i,j])
                   -k_left * (T[i,j] - T[i,j-1])) / dx**2

    return r

def vcycle(T, k1, k2, dx, dy, source=None, level=0, max_level=1, max_gauss_iter = 1,
           use_zero_bc=False, **boundary_condition):

    #choose right bc and update it's size
    if use_zero_bc:
        top_bc = boundary_condition["zero top bc"]
        bottom_bc = boundary_condition["zero bottom bc"]
        right_bc = boundary_condition["zero right bc"]
        left_bc = boundary_condition["zero left bc"]
    else:
        top_bc = boundary_condition["top bc"]
        bottom_bc = boundary_condition["bottom bc"]
        right_bc = boundary_condition["right bc"]
        left_bc = boundary_condition["left bc"]
    relevant_bc = {"bottom bc": bottom_bc,
                   "top bc": top_bc,
                   "right bc": right_bc,
                   "left bc": left_bc,
                   "circle bc": None}
    for bc_name, bc in relevant_bc.items():
        if bc is not None:
            bc.resize_for_square_mesh(*T.shape)

    T, *_ = solve_gauss_seidel(k1, k2, dx, dy, direction="x", max_rep=max_gauss_iter,
                               init_guess=T, source=source, **relevant_bc)

    r = residual(T, k1, k2, dx, dy)
    if source is not None:
        r = r + source

    if level==max_level:
        T, *_ = solve_gauss_seidel(k1, k2, dx, dy, direction="xy", max_rep=50,
                                   init_guess=T, source=source, **relevant_bc)
        return T

    rc = restrict(r)
    rc[0, :] = 0
    rc[-1, :] = 0
    rc[:, 0] = 0
    rc[:, -1] = 0
    ec = np.zeros_like(rc)

    ec = vcycle(ec, k1, k2, 2*dx, 2*dy, level=level+1, max_level=max_level, source=rc, max_gauss_iter=max_gauss_iter,
                use_zero_bc=True, **boundary_condition)

    T += prolong(ec)

    # update bc to fit the size once more
    for bc_name, bc in relevant_bc.items():
        if bc is not None:
            bc.resize_for_square_mesh(*T.shape)

    T, *_ = solve_gauss_seidel(k1, k2, dx, dy, direction="y", max_rep=max_gauss_iter,
                               init_guess=T, source=source, **relevant_bc)
    return T


def solve_multigrid(k1, k2, dx, dy, max_level, max_gauss_iter, use_zero_bc=False,
                    init_guess=None, source=None, max_rep=0, tol=10**-3, **boundary_condition):
    num_x = boundary_condition["bottom bc"].bc_points_num
    num_y = boundary_condition["right bc"].bc_points_num

    if init_guess is None:
        init_guess = np.zeros((num_y, num_x))
    if source is None:
        source = np.zeros((num_y, num_x))
    init_guess = init_guess.copy()
    old_solution = init_guess
    max_error = 1
    rep = 0

    solution = old_solution.copy()
    while max_error >= tol and (max_rep == 0 or rep < max_rep):
        solution = vcycle(solution, k1, k2, dx, dy, source=source, max_level=max_level, max_gauss_iter=max_gauss_iter,
                          use_zero_bc=use_zero_bc, **boundary_condition)

        error = abs(old_solution - solution)
        max_error = max(max(row) for row in error)
        old_solution = solution.copy()

        rep += 1

    return solution, rep, max_error
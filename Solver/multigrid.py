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
    num_x, num_y = T.shape
    r=np.zeros_like(T)

    for j in range(1, num_x - 1):
        for i in range(1, num_y - 1):

            k_up, k_down, k_left, k_right, k_center_x, k_center_y = get_k_star(i, j, num_x, num_y, k1, k2)

            r[i,j]=(k_right * (T[i+1,j] - T[i,j])
                   -k_left * (T[i,j] - T[i-1,j])) / dx**2 \
                 +(k_up * (T[i,j+1] - T[i,j])
                   -k_down*(T[i,j] - T[i,j-1])) / dy**2

    return r

def vcycle(T, k1, k2, dx, dy, level=0, max_level=4, **boundary_condition):

    T, *_ = solve_gauss_seidel(k1, k2, dx, dy, direction="x", max_rep=1, init_guess=T, **boundary_condition)

    r = residual(T, k1, k2, dx, dy)

    if level==max_level:
        return T

    rc = restrict(r)
    ec = np.zeros_like(rc)

    ec = vcycle(ec, k1, k2, 2*dx, 2*dy, level+1, max_level, **boundary_condition)

    T += prolong(ec)

    T, *_ = solve_gauss_seidel(k1, k2, dx, dy, direction="y", max_rep=1, init_guess=T, **boundary_condition)

    return T
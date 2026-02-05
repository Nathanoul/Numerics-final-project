import numpy as np
from .thomas_matrix_solver import thomas_solver

def get_ij(id, num_x):
    return id // num_x, id % num_x


def get_id(i, j, num_x):
    return i * num_x + j


def get_k(i, j, num_x, num_y, k1, k2):
    if (i >= num_y // 2) and (j >= num_x // 2):
        return k2
    else:
        return k1


def build_matrix_and_rhs_for_line(top_bc, bottom_bc, left_bc, right_bc, k1, k2, dx, dy, num_x, num_y,
                                  next_line, prev_line, direction, line_index):
    if direction == "y":
        A = np.zeros((num_y, num_y))
        rhs = np.zeros((num_y, 1))

        if ((line_index == 1) and (left_bc.type == "Neumann")) or \
           ((line_index == num_x - 2) and (right_bc.type == "Neumann")):
            for i in range(1, num_y - 1):
                j = line_index
                k = get_k(i, j, num_x, num_y, k1, k2)

                A[i, i] = -2 * (k / dx ** 2 + k / dy ** 2) + k / dx ** 2
                A[i, i - 1] = k / dy ** 2
                A[i, i + 1] = k / dy ** 2

                if line_index == 1:
                    flux = left_bc.get_flux_at_boundary_id(get_id(i - 1, j, num_x))
                else:
                    flux = -right_bc.get_flux_at_boundary_id(get_id(i + 1, j, num_x))
                rhs[i] = -k / dx ** 2 * next_line[i] + k / dx * flux

        else:
            for i in range(1, num_y - 1):
                j = line_index
                k = get_k(i, j, num_x, num_y, k1, k2)

                A[i, i] = -2 * (k / dx ** 2 + k / dy ** 2)
                A[i, i - 1] = k / dy ** 2
                A[i, i + 1] = k / dy ** 2

                rhs[i] = -k / dx ** 2 * next_line[i] - k / dx ** 2 * prev_line[i]

        if bottom_bc.type == "Dirichlet":
            A[0, 0] = 1
            i = 0
            j = line_index
            rhs[i] = bottom_bc.get_value_at_boundary_id(get_id(i, j, num_x))

        if bottom_bc.type == "Neumann":
            A[0, 0] = 1
            A[0, 1] = -1
            i = 0
            j = line_index
            rhs[i] = dy * bottom_bc.get_flux_at_boundary_id(get_id(i, j, num_x))

        if top_bc.type == "Dirichlet":
            A[num_y - 1, num_y - 1] = 1
            i = num_y - 1
            j = line_index
            rhs[i] = top_bc.get_value_at_boundary_id(get_id(i, j, num_x))

        if top_bc.type == "Neumann":
            A[num_y - 1, num_y - 1] = -1
            A[num_y - 1, num_y - 2] = 1
            i = num_y - 1
            j = line_index
            rhs[i] = dy * top_bc.get_flux_at_boundary_id(get_id(i, j, num_x))

    if direction == "x":
        A = np.zeros((num_x, num_x))
        rhs = np.zeros(num_x)

        if ((line_index == 1) and (bottom_bc.type == "Neumann")) or \
           ((line_index == num_y - 2) and (top_bc.type == "Neumann")):
            for j in range(1, num_x - 1):
                i = line_index
                k = get_k(i, j, num_x, num_y, k1, k2)

                A[j, j] = -2 * (k / dx ** 2 + k / dy ** 2) + k / dy ** 2
                A[j, j - 1] = k / dx ** 2
                A[j, j + 1] = k / dx ** 2

                if line_index == 1:
                    flux = bottom_bc.get_flux_at_boundary_id(get_id(i - 1, j, num_x))
                else:
                    flux = -top_bc.get_flux_at_boundary_id(get_id(i + 1, j, num_x))
                rhs[j] = -k / dy ** 2 * next_line[j] + k / dy * flux
        else:
            for j in range(1, num_x - 1):
                i = line_index
                k = get_k(i, j, num_x, num_y, k1, k2)

                A[j, j] = -2 * (k / dx ** 2 + k / dy ** 2)
                A[j, j - 1] = k / dx ** 2
                A[j, j + 1] = k / dx ** 2

                rhs[j] = -k / dy ** 2 * next_line[j] - k / dy ** 2 * prev_line[j]

        if left_bc.type == "Dirichlet":
            A[0, 0] = 1
            i = line_index
            j = 0
            rhs[j] = left_bc.get_value_at_boundary_id(get_id(i, j, num_x))

        if left_bc.type == "Neumann":
            A[0, 0] = 1
            A[0, 1] = -1
            i = line_index
            j = 0
            rhs[j] = dx * left_bc.get_flux_at_boundary_id(get_id(i, j, num_x))

        if right_bc.type == "Dirichlet":
            A[num_x - 1, num_x - 1] = 1
            i = line_index
            j = num_x - 1
            rhs[j] = right_bc.get_value_at_boundary_id(get_id(i, j, num_x))

        if right_bc.type == "Neumann":
            A[num_x - 1, num_x - 1] = -1
            A[num_x - 1, num_x - 2] = 1
            i = line_index
            j = num_x - 1
            rhs[j] = dx * right_bc.get_flux_at_boundary_id(get_id(i, j, num_x))

    return A, rhs


def solve_gauss_seidel(k1, k2, dx, dy, direction, max_rep=0, init_guess=None, tol=10**-3, **boundary_condition):
    top_bc = boundary_condition["top bc"]
    bottom_bc = boundary_condition["bottom bc"]
    right_bc = boundary_condition["right bc"]
    left_bc = boundary_condition["left bc"]

    num_x = len(bottom_bc.boundary_points_ids)
    num_y = len(right_bc.boundary_points_ids)

    if not init_guess:
        init_guess = np.zeros((num_y, num_x))
    for bc_name, bc in boundary_condition.items():
        if bc is not None:
            if bc.type == "Dirichlet":
                for bc_id in bc.boundary_points_ids:
                    i, j = get_ij(bc_id, num_x)
                    init_guess[i, j] = bc.get_value_at_boundary_id(bc_id)
    old_solution = init_guess
    max_error = 1
    rep = 0

    solution = old_solution.copy()
    while max_error >= tol and (max_rep is 0 or rep < max_rep):
        rep += 1
        if direction == "x" or direction == "xy":
            for i in range(1, num_y - 1):
                prev_line = solution[i - 1, :]
                next_line = solution[i + 1, :]
                A, rhs = build_matrix_and_rhs_for_line(top_bc, bottom_bc, left_bc, right_bc, k1, k2, dx, dy, num_x, num_y,
                                              next_line, prev_line, direction = "x", line_index = i)
                c = [A[j, j + 1] for j in range(num_x - 1)]
                b = [A[j, j] for j in range(num_x)]
                a = [A[j + 1, j] for j in range(num_x - 1)]
                solution[i, :] = thomas_solver(a, b, c, rhs)

            if bottom_bc.type == "Neumann":
                boundary_flux = np.array([flux for id, flux in bottom_bc.flux_at_boundary.items()])
                solution[0, :] = solution[1, :] + dy * boundary_flux

            if top_bc.type == "Neumann":
                boundary_flux = np.array([flux for id, flux in top_bc.flux_at_boundary.items()])
                solution[-1, :] = solution[-2, :] - dy * boundary_flux


        if direction == "y" or direction == "xy":
            for j in range(1, num_x - 1):
                prev_line = solution[:, j - 1]
                next_line = solution[:, j + 1]
                A, rhs = build_matrix_and_rhs_for_line(top_bc, bottom_bc, left_bc, right_bc, k1, k2, dx, dy, num_x,
                                                       num_y,
                                                       next_line, prev_line, direction = "y", line_index = j)
                c = [A[i, i + 1] for i in range(num_y - 1)]
                b = [A[i, i] for i in range(num_y)]
                a = [A[i + 1, i] for i in range(num_y - 1)]
                solution[:, j] = thomas_solver(a, b, c, rhs)

            if left_bc.type == "Neumann":
                boundary_flux = np.array([flux for id, flux in left_bc.flux_at_boundary.items()])
                solution[0, :] = solution[1, :] + dx * boundary_flux

            if right_bc.type == "Neumann":
                boundary_flux = np.array([flux for id, flux in right_bc.flux_at_boundary.items()])
                solution[-1, :] = solution[-2, :] - dx * boundary_flux

        error = abs(old_solution - solution)
        max_error = max(max(row) for row in error)
        old_solution = solution.copy()


    return solution.reshape(-1), rep, max_error
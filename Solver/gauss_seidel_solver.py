import numpy as np

def get_ij(id, num_x):
    return id // num_x, id % num_x

def get_id(i, j, num_x):
    return i * num_x + j

def build_matrix_for_line(bc_top, bc_bottom, bc_left, bc_right, k1, k2, dx, dy, num_x, num_y,
                          next_line, prev_line, direction, line_index):
    A = np.zeros(num_y)
    if direction == "y":
        if ((line_index == 1) and (bc_left.type == "Neumann")) or \
           ((line_index == num_x - 2) and (bc_right.type == "Neumann")):
            for i in range(1, num_y - 1):
                A[i, i] = -2 * (k1 / dx ** 2 + k1 / dy ** 2) + k1 / dx
                A[i + 1, i] = k1 / dy ** 2
                A[i, i + 1] = k1 / dy ** 2

        else:
            for i in range(1, num_y - 1):
                A[i, i] = -2 * (k1 / dx ** 2 + k1 / dy ** 2)
                A[i + 1, i] = k1 / dy ** 2
                A[i, i + 1] = k1 / dy ** 2

        if bc_bottom.type == "Dirichlet":
            A[0, 0] = 1

        if bc_bottom.type == "Neumann":
            A[0, 0] = 1
            A[0, 1] = -1

        if bc_top.type == "Dirichlet":
            A[num_y - 1, num_y - 1] = 1

        if bc_top.type == "Neumann":
            A[num_y - 1, num_y - 1] = -1
            A[num_y - 1, num_y - 2] = 1

    if direction == "x":
        if ((line_index == 1) and (bc_bottom.type == "Neumann")) or \
           ((line_index == num_y - 2) and (bc_top.type == "Neumann")):
            for j in range(1, num_x - 1):
                A[j, j] = -2 * (k1 / dx ** 2 + k1 / dy ** 2) + k1 / dy
                A[j + 1, j] = k1 / dx ** 2
                A[j, j + 1] = k1 / dx ** 2
        else:
            for j in range(1, num_x - 1):
                A[j, j] = -2 * (k1 / dx ** 2 + k1 / dy ** 2)
                A[j + 1, j] = k1 / dx ** 2
                A[j, j + 1] = k1 / dx ** 2

        if bc_left.type == "Dirichlet":
            A[0, 0] = 1

        if bc_left.type == "Neumann":
            A[0, 0] = 1
            A[0, 1] = -1

        if bc_right.type == "Dirichlet":
            A[num_x - 1, num_x - 1] = 1

        if bc_right.type == "Neumann":
            A[num_x - 1, num_x - 1] = -1
            A[num_x - 1, num_x - 2] = 1

    return A

def solve_gauss_seidel(k1, k2, dx, dy,  **boundary_condition):
    top_bc = boundary_condition["top bc"]
    bottom_bc = boundary_condition["bottom bc"]
    right_bc = boundary_condition["right bc"]
    left_bc = boundary_condition["left bc"]

    num_x = len(bottom_bc.boundary_points_ids)
    num_y = len(right_bc.boundary_points_ids)

    init_guess = np.zeros((num_x, num_y))
    for bc_name, bc in boundary_condition.items():
        if bc.type == "Dirichlet":
            for bc_id in bc.boundary_points_ids:
                i, j = get_ij(bc_id, num_x)
                init_guess[i, j] = bc.get_value_at_boundary_id(bc_id)
    solution = init_guess

    return solution
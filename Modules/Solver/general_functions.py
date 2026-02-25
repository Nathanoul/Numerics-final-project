def get_ij(id, num_x):
    return id // num_x, id % num_x


def get_id(i, j, num_x):
    return i * num_x + j


def get_k(i, j, num_x, num_y, k1, k2):
    if (i >= num_y // 2) and (j >= num_x // 2):
        return k2
    else:
        return k1

def get_k_star(i, j, num_x, num_y, k1, k2):
    def harmonic_mean(a, b):
        return 2 * a * b / (b + a)

    k_left = harmonic_mean( get_k(i, j, num_x, num_y, k1, k2), get_k(i, j - 1, num_x, num_y, k1, k2))
    k_right = harmonic_mean( get_k(i, j, num_x, num_y, k1, k2), get_k(i, j + 1, num_x, num_y, k1, k2))
    k_down = harmonic_mean( get_k(i - 1, j, num_x, num_y, k1, k2), get_k(i, j, num_x, num_y, k1, k2))
    k_up = harmonic_mean( get_k(i + 1, j, num_x, num_y, k1, k2), get_k(i, j, num_x, num_y, k1, k2))

    k_center_x = (k_left + k_right) / 2
    k_center_y = (k_up + k_down) / 2
    return k_up, k_down, k_left, k_right, k_center_x, k_center_y


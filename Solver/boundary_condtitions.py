import numpy as np

class DirichletBC():

    def __init__(self, location, value_func, points, edges, cells, lettice,
                 eps=10**-5, cx=0.0, cy=0.0, R=0.2):
        self.location = location
        self.value_func = value_func
        self.eps = eps
        self.cx = cx
        self.cy = cy
        self.R = R

        x, y = points["x"], points["y"]
        top_x = max(x)
        bottom_x = min(x)
        top_y = max(y)
        bottom_y = min(y)
        if self.location == "left":
            boundary = x - bottom_x <= 0.5 * self.eps
        if self.location == "right":
            boundary = top_x - x <= 0.5 * self.eps
        if self.location == "bottom":
            boundary = y - bottom_y <= 0.5 * self.eps
        if self.location == "top":
            boundary = top_y - y <= 0.5 * self.eps
        if self.location == "circle":
            boundary = (x - self.cx) ** 2 + (y - self.cy) ** 2 - self.R ** 2 <= 0.25 * self.eps ** 2
        self.boundary_points_ids = [i for i, b in enumerate(boundary) if b == 1]

        self.boundary_edges_ids = []
        for edge_id, edge in edges.items():
            n1 = edge["n1"]
            n2 = edge["n2"]
            if n1 in self.boundary_points_ids and n2 in self.boundary_points_ids:
                self.boundary_edges_ids.append(edge_id)

        self.boundary_cells_ids = []
        for cell_id, cell in cells.items():
            if lettice == "homogeneous square":
                n1 = cell["n1"]
                n2 = cell["n2"]
                n3 = cell["n3"]
                n4 = cell["n4"]

                if n1 in self.boundary_points_ids or n2 in self.boundary_points_ids \
                        or n3 in self.boundary_points_ids or n4 in self.boundary_points_ids:
                    self.boundary_cells_ids.append(cell_id)


    def on_boundary(self, point_id):
        return any(self.boundary_points_ids == point_id)

    # Don't use this function, it doesn't have a purpose yet
    def apply(self, A, b, n, points):
        x, y = points["x"][n], points["y"][n]
        A[n, :] = 0
        A[n, n] = 1
        b[n] = self.value_func(x, y)



class NeumannBC():

    def __init__(self, location, value_func, points, edges, cells, lettice,
                 eps=10 ** -5, cx=0.0, cy=0.0, R=0.2):
        self.location = location
        self.value_func = value_func
        self.lattice = lettice
        self.eps = eps
        self.cx = cx
        self.cy = cy
        self.R = R

        x, y = points["x"], points["y"]
        top_x = max(x)
        bottom_x = min(x)
        top_y = max(y)
        bottom_y = min(y)
        if self.location == "left":
            boundary = x - bottom_x <= 0.5 * self.eps
        if self.location == "right":
            boundary = top_x - x <= 0.5 * self.eps
        if self.location == "bottom":
            boundary = y - bottom_y <= 0.5 * self.eps
        if self.location == "top":
            boundary = top_y - y <= 0.5 * self.eps
        if self.location == "circle":
            boundary = (x - self.cx) ** 2 + (y - self.cy) ** 2 - self.R ** 2 <= 0.25 * self.eps ** 2
        self.boundary_points_ids = [i for i, b in enumerate(boundary) if b == 1]

        self.boundary_edges_ids = []
        for edge_id, edge in edges.items():
            n1 = edge["n1"]
            n2 = edge["n2"]
            if n1 in self.boundary_points_ids and n2 in self.boundary_points_ids:
                self.boundary_edges_ids.append(edge_id)

        self.boundary_cells_ids = []
        for cell_id, cell in cells.items():
            if lettice == "homogeneous rectangular":
                n1 = cell["n1"]
                n2 = cell["n2"]
                n3 = cell["n3"]
                n4 = cell["n4"]

                if n1 in self.boundary_points_ids or n2 in self.boundary_points_ids \
                        or n3 in self.boundary_points_ids or n4 in self.boundary_points_ids:
                    self.boundary_cells_ids.append(cell_id)

    def on_boundary(self, point_id):
        return any(self.boundary_points_ids == point_id)


    #Don't use this function, it doesn't have a purpose yet
    def apply(self, A, b, n, points, edges):
        if self.lattice == "homogeneous square":
            x, y = points["x"][n], points["y"][n]

            N_row = np.sqrt(points["x"][-1] + 1)
            i = n % N_row
            j = n // N_row
            dx, dy = edges["len"][0], edges["len"][1]
            q = self.flux_func(x, y)

            if self.location == "bottom":
                north = i * N_row + j + 1
                A[n, n] += 1.0 / dy ** 2
                A[n, north] -= 1.0 / dy ** 2
                b[n] += q / dy
            if self.location == "top":
                south = i * N_row + j - 1
                A[n, n] += 1.0 / dy ** 2
                A[n, south] -= 1.0 / dy ** 2
                b[n] += q / dy
            if self.location == "left":
                east = (i + 1) * N_row + j
                A[n, n] += 1.0 / dy ** 2
                A[n, east] -= 1.0 / dy ** 2
                b[n] += q / dy
            if self.location == "left":
                west = (i - 1) * N_row + j
                A[n, n] += 1.0 / dy ** 2
                A[n, west] -= 1.0 / dy ** 2
                b[n] += q / dy


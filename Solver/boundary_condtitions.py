import numpy as np

class DirichletBC():

    def __init__(self, location, value_func, points, edges, cells,
                 eps=10**-5, cx=0.0, cy=0.0, R=0.2):
        self.type = "Dirichlet"
        self.location = location
        self.value_func = value_func
        self.eps = eps
        self.cx = cx
        self.cy = cy
        self.R = R

        x, y = np.array(list(points["x"].values())), np.array(list(points["y"].values()))
        self.top_x = max(x)
        self.bottom_x = min(x)
        self.top_y = max(y)
        self.bottom_y = min(y)
        if self.location == "left":
            boundary = x - self.bottom_x <= 0.5 * self.eps
        if self.location == "right":
            boundary = self.top_x - x <= 0.5 * self.eps
        if self.location == "bottom":
            boundary = y - self.bottom_y <= 0.5 * self.eps
        if self.location == "top":
            boundary = self.top_y - y <= 0.5 * self.eps
        if self.location == "circle":
            boundary = (x - self.cx) ** 2 + (y - self.cy) ** 2 - self.R ** 2 <= 0.25 * self.eps ** 2
        self.boundary_points_ids = [i for i, b in enumerate(boundary) if b == 1]
        bc_ids = self.boundary_points_ids

        self.boundary_edges_ids = []
        for edge_id in edges["edgeID"].values():
            n1 = edges["n1"][edge_id]
            n2 = edges["n2"][edge_id]
            if n1 in bc_ids and n2 in bc_ids:
                self.boundary_edges_ids.append(edge_id)

        self.boundary_cells_ids = []
        if "n4" in cells:
            for cell_id in cells["cellID"].values():
                n1 = cells["n1"][cell_id]
                n2 = cells["n2"][cell_id]
                n3 = cells["n3"][cell_id]
                n4 = cells["n4"][cell_id]

                if n1 in bc_ids or n2 in bc_ids or n3 in bc_ids or n4 in bc_ids:
                    self.boundary_cells_ids.append(cell_id)

        else:
            for cell_id in cells["cellID"].values():
                n1 = cells["n1"][cell_id]
                n2 = cells["n2"][cell_id]
                n3 = cells["n3"][cell_id]

                bc_ids = self.boundary_points_ids
                if (n1 in bc_ids and n2 in bc_ids) or (n1 in bc_ids and n3 in bc_ids)\
                    or (n2 in bc_ids and n3 in bc_ids):
                    self.boundary_cells_ids.append(cell_id)

        self.values_at_boundary = {id: self.value_func(x[id], y[id]) for id in bc_ids}

    #updates just point ids
    def resize_for_square_mesh(self, new_num_x, new_num_y):
        bc_ids = []
        bc_values = {}
        bottom_x = self.bottom_x
        bottom_y = self.bottom_y
        top_x = self.top_x
        top_y = self.top_y
        y_span = np.linspace(bottom_y, top_y, new_num_y)
        x_span = np.linspace(bottom_x, top_x, new_num_x)

        if self.location == "left":
            for i in range(new_num_y):
                new_bc_id = i * new_num_y
                bc_ids.append(new_bc_id)
                bc_values[new_bc_id] = self.value_func(bottom_x, y_span[i])
        if self.location == "right":
            for i in range(new_num_y):
                new_bc_id = (i + 1) * new_num_y - 1
                bc_ids.append(new_bc_id)
                bc_values[new_bc_id] = self.value_func(top_x, y_span[i])
        if self.location == "top":
            for j in range(new_num_x):
                new_bc_id = new_num_x * (new_num_y - 1) + j
                bc_ids.append(new_bc_id)
                bc_values[new_bc_id] = self.value_func(x_span[j], top_y)
        if self.location == "bottom":
            for j in range(new_num_x):
                new_bc_id = j
                bc_ids.append(new_bc_id)
                bc_values[new_bc_id] = self.value_func(x_span[j], bottom_y)
        self.values_at_boundary = bc_values
        self.boundary_points_ids = bc_ids

    def get_value_at_boundary_id(self, id):
        return self.values_at_boundary[id]

    def on_boundary(self, point_id):
        return any(self.boundary_points_ids == point_id)


    # Don't use this function, it doesn't have a purpose yet
    def apply(self, A, b, n, points):
        x, y = points["x"][n], points["y"][n]
        A[n, :] = 0
        A[n, n] = 1
        b[n] = self.value_func(x, y)



class NeumannBC():

    def __init__(self, location, flux_func, points, edges, cells,
                 eps=10 ** -5, cx=0.0, cy=0.0, R=0.2):
        self.type = "Neumann"
        self.location = location
        self.flux_func = flux_func
        self.eps = eps
        self.cx = cx
        self.cy = cy
        self.R = R

        x, y = np.array(list(points["x"].values())), np.array(list(points["y"].values()))
        self.top_x = max(x)
        self.bottom_x = min(x)
        self.top_y = max(y)
        self.bottom_y = min(y)
        if self.location == "left":
            boundary = x - self.bottom_x <= 0.5 * self.eps
        if self.location == "right":
            boundary = self.top_x - x <= 0.5 * self.eps
        if self.location == "bottom":
            boundary = y - self.bottom_y <= 0.5 * self.eps
        if self.location == "top":
            boundary = self.top_y - y <= 0.5 * self.eps
        if self.location == "circle":
            boundary = (x - self.cx) ** 2 + (y - self.cy) ** 2 - self.R ** 2 <= 0.25 * self.eps ** 2
        self.boundary_points_ids = [i for i, b in enumerate(boundary) if b == 1]
        bc_ids = self.boundary_points_ids

        self.boundary_edges_ids = []
        for edge_id in edges["edgeID"].values():
            n1 = edges["n1"][edge_id]
            n2 = edges["n2"][edge_id]
            if n1 in bc_ids and n2 in bc_ids:
                self.boundary_edges_ids.append(edge_id)

        self.boundary_cells_ids = []
        if "n4" in cells:
            for cell_id in cells["cellID"].values():
                n1 = cells["n1"][cell_id]
                n2 = cells["n2"][cell_id]
                n3 = cells["n3"][cell_id]
                n4 = cells["n4"][cell_id]

                if n1 in bc_ids or n2 in bc_ids or n3 in bc_ids or n4 in bc_ids:
                    self.boundary_cells_ids.append(cell_id)

        else:
            for cell_id in cells["cellID"].values():
                n1 = cells["n1"][cell_id]
                n2 = cells["n2"][cell_id]
                n3 = cells["n3"][cell_id]

                bc_ids = self.boundary_points_ids
                if (n1 in bc_ids and n2 in bc_ids) or (n1 in bc_ids and n3 in bc_ids)\
                    or (n2 in bc_ids and n3 in bc_ids):
                    self.boundary_cells_ids.append(cell_id)

        self.flux_at_boundary = {id: self.flux_func(x[id], y[id]) for id in bc_ids}

    #updates just point ids
    def resize_for_square_mesh(self, new_num_y, new_num_x):
        bc_ids = []
        bc_values = {}
        bottom_x = self.bottom_x
        bottom_y = self.bottom_y
        top_x = self.top_x
        top_y = self.top_y
        y_span = np.linspace(bottom_y, top_y, new_num_y)
        x_span = np.linspace(bottom_x, top_x, new_num_x)

        if self.location == "left":
            for i in range(new_num_y):
                new_bc_id = i * new_num_y
                bc_ids.append(new_bc_id)
                bc_values[new_bc_id] = self.flux_func(bottom_x, y_span[i])
        if self.location == "right":
            for i in range(new_num_y):
                new_bc_id = (i + 1) * new_num_y - 1
                bc_ids.append(new_bc_id)
                bc_values[new_bc_id] = self.flux_func(top_x, y_span[i])
        if self.location == "top":
            for j in range(new_num_x):
                new_bc_id = new_num_x * (new_num_y - 1) + j
                bc_ids.append(new_bc_id)
                bc_values[new_bc_id] = self.flux_func(x_span[j], top_y)
        if self.location == "bottom":
            for j in range(new_num_x):
                new_bc_id = j
                bc_ids.append(new_bc_id)
                bc_values[new_bc_id] = self.flux_func(x_span[j], bottom_y)
        self.flux_at_boundary = bc_values
        self.boundary_points_ids = bc_ids

    def get_flux_at_boundary_id(self, id):
        return self.flux_at_boundary[id]


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


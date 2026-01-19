import numpy as np

class Grid:
    def __init__(self, points, edges, tol=1e-12):
        self.points = np.asarray(points)        # (N, 2)
        self.edges  = np.asarray(edges, int)    # (E, 2)

        self.Np = self.points.shape[0]
        self.Ne = self.edges.shape[0]

        self._build_adjacency()
        self._infer_spacing(tol)

    def _build_adjacency(self):
        self.neighbors = [[] for _ in range(self.Np)]

        for i, j in self.edges:
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

    def _infer_spacing(self, tol):
        xs = np.unique(np.round(self.points[:,0], 12))
        ys = np.unique(np.round(self.points[:,1], 12))

        xs.sort()
        ys.sort()

        dxs = np.diff(xs)
        dys = np.diff(ys)

        self.dx = np.min(dxs) if len(dxs) else None
        self.dy = np.min(dys) if len(dys) else None

        self.Nx = len(xs)
        self.Ny = len(ys)

        self.structured = (
            np.allclose(dxs, self.dx, atol=tol) and
            np.allclose(dys, self.dy, atol=tol)
        )

    def build_index_map(self):
        xs = np.unique(self.points[:,0])
        ys = np.unique(self.points[:,1])

        xs.sort()
        ys.sort()

        self.x_to_i = {x: i for i, x in enumerate(xs)}
        self.y_to_j = {y: j for j, y in enumerate(ys)}

        self.ij_of_node = np.zeros((self.Np, 2), dtype=int)

        for k, (x, y) in enumerate(self.points):
            self.ij_of_node[k] = [
                self.x_to_i[x],
                self.y_to_j[y]
            ]

    def find_boundary_nodes(self):
        counts = np.array([len(n) for n in self.neighbors])
        self.boundary_nodes = np.where(counts < 4)[0]

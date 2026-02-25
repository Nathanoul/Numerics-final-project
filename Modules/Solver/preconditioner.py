from .multigrid import *
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .general_functions import *

class Multigrid_preconditioner():
    def __init__(self, k1, k2, dx, dy, max_rep=1, max_gauss_iter=1, max_level=None, **boundary_condition):
        self.k1 = k1
        self.k2 = k2
        self.dx = dx
        self.dy = dy
        self.max_level=max_level
        self.max_rep = max_rep
        self.max_gauss_iter = max_gauss_iter

        self.bc = boundary_condition
        self.num_x = self.bc["bottom bc"].bc_points_num
        self.num_y = self.bc["right bc"].bc_points_num
        self.actual_bc = {"bottom bc": self.bc["bottom bc"],
                          "top bc": self.bc["top bc"],
                          "right bc": self.bc["right bc"],
                          "left bc": self.bc["left bc"],
                          "circle bc": self.bc["circle bc"]}

        self.zero_bc = {"zero bottom bc": self.bc["zero bottom bc"],
                       "zero top bc": self.bc["zero top bc"],
                       "zero right bc": self.bc["zero right bc"],
                       "zero left bc": self.bc["zero left bc"],
                       "zero circle bc": self.bc["zero circle bc"]}

    def mg_preconditioner(self, r):
        z0 = np.zeros_like(r).astype(float)
        z, *_ = solve_multigrid(
                init_guess=z0,
                k1=self.k1,
                k2=self.k2,
                dx=self.dx,
                dy=self.dy,
                source=r,
                max_level=self.max_level,
                max_gauss_iter=self.max_gauss_iter,
                max_rep=self.max_rep,
                use_zero_bc=True,
                **self.bc
        )
        return z

    def A_operator(self, T):
        AT_no_bc = -residual(T=T, k1=self.k1, k2=self.k2, dx=self.dx, dy=self.dy)
        get_ij_wrapper = lambda id: get_ij(id, self.num_x)

        for bc_name, bc in self.actual_bc.items():
            if bc is not None:
                AT = bc.apply_AT(AT_no_bc, T, get_ij_wrapper)

        return AT

    def matvec(self, x):
        T = x.reshape((self.num_y, self.num_x))
        AT = self.A_operator(T)
        return AT.ravel()

    def precond(self, x):
        r = x.reshape(self.num_y, self.num_x)
        z = self.mg_preconditioner(r)
        return z.ravel()

    def get_M_operator(self):
        N = self.num_x * self.num_y
        return LinearOperator((N, N), matvec=self.precond)

    def get_A_operator(self):
        N = self.num_x * self.num_y
        return LinearOperator((N, N), matvec=self.matvec)

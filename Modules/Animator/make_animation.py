import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


class FVAnimation:
    """
    Creates and saves a matplotlib animation from solve_unsteady_LU output.

    Two rendering modes:
    - Triangulation mode (Q5): uses tri_vertices + node positions for
      tripcolor(shading='flat').  One colour value per triangle.
    - Scatter/centroid mode (Q6 IBM / quad mesh): used when
      results["tri_vertices"] is None.  Plots cell centroids from
      results["cx"] / results["cy"] as a scatter with marker size
      scaled to the cell size.

    Usage
    -----
    anim = FVAnimation(results, points)    # Q5 triangular mesh
    anim = FVAnimation(results, points)    # Q6 quad/IBM mesh (tri_vertices=None)
    anim.save("output_dir/")               # gif by default
    anim.save("output_dir/", fmt="mp4")
    anim.preview()                         # show steady-state frame
    """

    def __init__(self, results, points,
                 cmap="inferno", figsize=(7, 6),
                 title="Unsteady heat conduction"):
        """
        Parameters
        ----------
        results : dict
            Output of solve_unsteady_LU or solve_ibm_schur_LU.
        points : dict
            Node table from csv_data_to_dic  (keys 'x', 'y').
            Used only in triangulation mode; may be None for centroid mode.
        cmap : str
            Matplotlib colormap.
        figsize : tuple
            Figure size in inches.
        title : str
            Animation super-title.
        """
        self.cmap    = cmap
        self.figsize = figsize
        self.title   = title

        # ------------------------------------------------------------------
        # Determine rendering mode based on whether tri_vertices is provided.
        # FIX: Q6 (IBM / quad mesh) returns tri_vertices=None.  Original code
        # crashed with  np.vectorize(id_to_pos.get)(None).
        # We fall back to centroid-scatter rendering in that case.
        # ------------------------------------------------------------------
        self._use_scatter = (results.get("tri_vertices") is None)

        if not self._use_scatter:
            # ---- Triangulation mode (Q5) --------------------------------
            node_keys = list(points["x"].keys())
            px = np.array([points["x"][k] for k in node_keys])
            py = np.array([points["y"][k] for k in node_keys])

            # Remap node IDs to contiguous 0-based if needed
            id_to_pos = {k: i for i, k in enumerate(node_keys)}
            raw_verts = results["tri_vertices"]          # shape (N_tri, 3)
            verts = np.vectorize(id_to_pos.get)(raw_verts)

            self.triang = mtri.Triangulation(px, py, verts)
        else:
            # ---- Centroid-scatter mode (Q6 / IBM / quad mesh) -----------
            self.cx = results["cx"]
            self.cy = results["cy"]

        # ------------------------------------------------------------------
        # Frame list: transient snapshots  +  steady-state as final frame
        # ------------------------------------------------------------------
        self.frame_times = list(results["times"]) + ["steady"]
        self.frame_data  = (list(results["snapshots"])
                            + [results["steady_state"]])

        # Global colour scale across ALL frames
        all_T       = np.concatenate(self.frame_data)
        self.vmin   = float(np.nanmin(all_T))
        self.vmax   = float(np.nanmax(all_T))

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_time(t):
        return "t = steady state" if t == "steady" else f"t = {t:.4f}"

    # ------------------------------------------------------------------
    def _build_figure(self):
        """Create figure, axes, and the initial plot collection."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        T0 = self.frame_data[0]

        if not self._use_scatter:
            # Triangulation: one colour per triangle (cell-centred)
            tpc = ax.tripcolor(self.triang, T0,
                               cmap=self.cmap,
                               vmin=self.vmin,
                               vmax=self.vmax)
        else:
            # Centroid scatter: each cell is a square marker
            # Marker size is scaled so cells visually tile the domain
            x_span = self.cx.max() - self.cx.min()
            n_approx = int(np.sqrt(len(self.cx)))
            pt_size  = (fig.get_size_inches()[0] * fig.dpi / n_approx) ** 2 * 0.9
            tpc = ax.scatter(self.cx, self.cy, c=T0,
                             cmap=self.cmap,
                             vmin=self.vmin, vmax=self.vmax,
                             s=pt_size, marker="s",
                             linewidths=0)

        fig.colorbar(tpc, ax=ax, label="Temperature")
        ttl = ax.set_title(
            f"{self.title}\n{self._fmt_time(self.frame_times[0])}")
        fig.tight_layout()
        return fig, ax, tpc, ttl

    # ------------------------------------------------------------------
    def _update_frame(self, tpc, ttl, fi):
        """Update the plot collection for frame fi."""
        T = self.frame_data[fi]
        if not self._use_scatter:
            tpc.set_array(T)
        else:
            tpc.set_array(T)
        ttl.set_text(
            f"{self.title}\n{self._fmt_time(self.frame_times[fi])}")
        return tpc, ttl

    # ------------------------------------------------------------------
    def save(self, output_dir, filename="heat_animation",
             fmt="gif", fps=15, dpi=120):
        """
        Render and save the animation.

        Parameters
        ----------
        output_dir : str   Directory (created if absent).
        filename   : str   File stem without extension.
        fmt        : str   'gif' or 'mp4'.
        fps        : int   Frames per second.
        dpi        : int   Resolution.

        Returns
        -------
        out_path : str   Absolute path of the saved file.
        """
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{filename}.{fmt}")

        fig, ax, tpc, ttl = self._build_figure()
        n_frames = len(self.frame_data)

        def _update(fi):
            return self._update_frame(tpc, ttl, fi)

        anim = FuncAnimation(fig, _update,
                             frames=n_frames,
                             interval=max(1, 1000 // fps),
                             blit=False)

        writer = FFMpegWriter(fps=fps) if fmt == "mp4" else PillowWriter(fps=fps)

        print(f"Saving animation -> {out_path}  "
              f"({n_frames} frames @ {fps} fps) ...")
        anim.save(out_path, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"  Saved: {out_path}")
        return out_path

    # ------------------------------------------------------------------
    def preview(self, frame_idx=-1):
        """
        Display a single frame without saving.
        Default frame_idx=-1 shows the steady-state (last frame).
        """
        fig, ax, tpc, ttl = self._build_figure()
        fi = frame_idx % len(self.frame_data)
        self._update_frame(tpc, ttl, fi)
        plt.tight_layout()
        plt.show()
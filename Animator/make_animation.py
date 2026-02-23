import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


class FVAnimation:
    """
    Creates and saves a matplotlib animation from solve_unsteady_LU output.

    Cell-centred FV data (one value per triangle) is visualised with
    tripcolor(shading='flat'), which correctly maps one colour per triangle.
    shading='gouraud' is NOT used here because it requires per-node values.

    Usage
    -----
    anim = FVAnimation(results, points)
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
            Output of solve_unsteady_LU.
        points : dict
            Node table from csv_data_to_dic  (keys 'x', 'y').
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
        # Build matplotlib Triangulation from mesh NODE coordinates.
        # tri_vertices (N_tri x 3) indexes into the node arrays.
        # ------------------------------------------------------------------
        node_keys = list(points["x"].keys())
        px = np.array([points["x"][k] for k in node_keys])
        py = np.array([points["y"][k] for k in node_keys])

        # If node IDs are not contiguous 0-based integers, remap them.
        id_to_pos = {k: i for i, k in enumerate(node_keys)}
        raw_verts = results["tri_vertices"]          # shape (N_tri, 3)
        verts = np.vectorize(id_to_pos.get)(raw_verts)

        self.triang = mtri.Triangulation(px, py, verts)

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
        """Create figure, axes, and the initial tripcolor collection."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        T0  = self.frame_data[0]
        # shading='flat'  ->  one colour value per triangle  (cell-centred)
        tpc = ax.tripcolor(self.triang, T0,
                           cmap=self.cmap,
                           vmin=self.vmin,
                           vmax=self.vmax)
        fig.colorbar(tpc, ax=ax, label="Temperature")
        ttl = ax.set_title(
            f"{self.title}\n{self._fmt_time(self.frame_times[0])}")
        fig.tight_layout()
        return fig, ax, tpc, ttl

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
            # tripcolor flat: update the facecolor array
            tpc.set_array(self.frame_data[fi])
            ttl.set_text(
                f"{self.title}\n{self._fmt_time(self.frame_times[fi])}")
            return tpc, ttl

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
        tpc.set_array(self.frame_data[fi])
        ttl.set_text(
            f"{self.title}\n{self._fmt_time(self.frame_times[fi])}")
        plt.tight_layout()
        plt.show()


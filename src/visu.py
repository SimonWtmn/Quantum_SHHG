"""
=============================================================================
HBT Grid / Single-Plot Visualisation
=============================================================================

Builds the three diagnostic figures of the HBT pipeline, each as a single
detector-pair plot or as the full comparison grid:

* coherence (correlation-histogram) spectra,
* g^(2)(0) integration sweeps,
* Cauchy-Schwarz R sweeps.

Design
------
Every ``plot_*`` method is a **pure figure builder**: it constructs a Matplotlib
figure and *returns* it (``fig, ax`` or ``fig, axes``) WITHOUT calling
``plt.show()`` or saving. The orchestration layer (:mod:`src.report`) decides
whether to display (interactive ipympl zoom), save to disk, or both. This keeps a
single code path for on-screen zooming and for the downloaded PNGs.

Titles are intentionally simple: a short descriptive title on top and one compact
metadata line beneath it (sample, filter, polariser, power, date). Pass
``show_details=True`` for the extra acquisition line (laser, binning, ...).

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquee (LOA), Ecole Polytechnique
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "figure.titlesize": 18,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# Grid geometry shared by coherence / g2 (5x3) and R (4x3).
CROSS_ROWS = ["TT", "TR", "RT", "RR"]
CROSS_COLS = [("3", "4"), ("3", "5"), ("4", "5")]
AUTO_COLS = ["3", "4", "5"]


class GridVisualizer:
    """Render coherence / g^(2) / R as single plots or full comparison grids.

    Parameters
    ----------
    measurements : HBTMeasurement | list[HBTMeasurement]
        One run, or several to overlay (comparison).
    labels : list[str], optional
        Legend labels (default: each run's ``legend_tag()``).
    comparison_variable : str, optional
        What changes across the overlaid runs (shown in the grid title).
    show_details : bool
        Add the extra acquisition metadata line to titles.
    """

    def __init__(self, measurements, labels=None, comparison_variable=None,
                 show_details=False):
        if not isinstance(measurements, (list, tuple)):
            self.runs = [measurements]
        else:
            self.runs = list(measurements)
        self.labels = labels if labels else self._auto_labels(self.runs)
        self.comparison_variable = comparison_variable
        self.show_details = show_details

        self.histogram_source = (
            "physical"
            if self.runs and all(getattr(r, "has_physical_histograms", False) for r in self.runs)
            else "virtual"
        )
        delay_kind = "Physical" if self.histogram_source == "physical" else "Virtual"
        self.method_colors = {"direct": "#e74c3c", "delay": "#2980b9", "heralded": "#27ae60"}
        self.method_labels = {
            "direct": "Physical (direct)",
            "delay": f"{delay_kind} (delay)",
            "heralded": "Virtual (heralded)",
        }
        self.run_colors = self._make_run_colors(len(self.runs))

    @property
    def is_comparison(self):
        return len(self.runs) > 1

    @staticmethod
    def _auto_labels(runs):
        """Legend labels that show only what VARIES across runs: split each run's
        ``legend_tag`` into ' | ' tokens and drop tokens shared by every run, so a
        scan that only changes the angle is labelled by the angle alone. Falls back
        to the full tag (or folder name) for any run left empty."""
        tags = [r.legend_tag() for r in runs]
        if len(runs) <= 1:
            return tags
        token_lists = [t.split(" | ") for t in tags]
        common = set(token_lists[0])
        for toks in token_lists[1:]:
            common &= set(toks)
        labels = []
        for toks, full, r in zip(token_lists, tags, runs):
            kept = [tk for tk in toks if tk not in common]
            labels.append(" | ".join(kept) if kept else (full or r.run_dir.name))
        return labels

    @staticmethod
    def _make_run_colors(n):
        if n <= 10:
            return [plt.cm.tab10(i) for i in range(n)]
        if n <= 20:
            return [plt.cm.tab20(i) for i in range(n)]
        return [plt.cm.turbo(i / (n - 1)) for i in range(max(n, 2))]

    # =====================================================================
    # Title / layout
    # =====================================================================

    def _subtitle(self):
        r0 = self.runs[0]
        if not self.is_comparison:
            return r0.subtitle(details=self.show_details)
        # Comparison: only the shared physical context (sample), plus what varies.
        bits = []
        if r0.sample:
            bits.append(rf"Sample: {r0.sample}")
        if self.comparison_variable:
            bits.append(rf"Varying: {self.comparison_variable}")
        if r0.date and r0.date != "Unknown":
            bits.append(r0.date)
        return "  |  ".join(bits)

    def _decorate_single(self, fig, ax, base):
        """Title + subtitle (fig-level) and an in-axes legend for a single plot."""
        fig.suptitle(base, y=0.985, fontsize=15)
        sub = self._subtitle()
        if sub:
            fig.text(0.5, 0.915, sub, ha="center", va="top", fontsize=10.5,
                     color="#555555")
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(loc="best", framealpha=0.92, edgecolor="0.7", fontsize=10)
        fig.subplots_adjust(top=0.86, bottom=0.12, left=0.12, right=0.96)
        return fig

    def _decorate_grid(self, fig, axes, base):
        """Title + subtitle and one global legend above a subplot grid.

        Vertical budget (figure fraction, top-down): suptitle, subtitle, legend
        strip, then the axes. ``top`` is kept low enough that the first-row subplot
        titles sit clear below the legend."""
        fig.suptitle(base, y=0.993, fontsize=18)
        sub = self._subtitle()
        if sub:
            fig.text(0.5, 0.975, sub, ha="center", va="top", fontsize=12,
                     color="#555555")

        handles, labels = [], []
        for ax in np.asarray(axes).flatten():
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            ncol = max(1, min(len(labels), 6))
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, 0.962), ncol=ncol, frameon=True,
                       fontsize=12)
        fig.subplots_adjust(top=0.91, bottom=0.05, hspace=0.32, wspace=0.23)
        return fig

    # =====================================================================
    # Window helper
    # =====================================================================

    @staticmethod
    def _x_bounds(time_window_ns, xlim):
        if xlim is None:
            return (-time_window_ns, time_window_ns)
        if isinstance(xlim, (int, float)):
            return (-xlim, xlim)
        return tuple(xlim)

    # =====================================================================
    # 1. Coherence
    # =====================================================================

    @staticmethod
    def _normalize_coh(delta_t, y, x_bounds, normalize):
        if normalize in ("peak", "max", "area"):
            inwin = (delta_t >= x_bounds[0]) & (delta_t <= x_bounds[1])
            sel = y[inwin]
            ref = (float(np.sum(sel)) if normalize == "area"
                   else (float(np.max(sel)) if sel.size else 0.0))
            return y / ref if ref > 0 else y
        return y * 1e-3

    @staticmethod
    def _coh_ylabel(normalize):
        return {
            "peak": r"Normalised counts (peak $=1$)",
            "max": r"Normalised counts (peak $=1$)",
            "area": r"Normalised counts (unit area)",
        }.get(normalize, r"Counts $N \times 10^3$")

    def _draw_coh(self, ax, ch1_ref, ch2_ref, x_bounds, normalize,
                  integration_window_ns, logy, ylim):
        for idx, run in enumerate(self.runs):
            try:
                ch1 = run.get_ch(self.runs[0].channel_map[ch1_ref])
                ch2 = run.get_ch(self.runs[0].channel_map[ch2_ref])
            except KeyError:
                continue
            x, y = run.get_correlation_trace(ch1, ch2)
            if x is None:
                continue
            t0 = run.calculate_t0_shift(ch1, ch2)
            delta_t = x - t0
            mask = (delta_t >= x_bounds[0] - 1.0) & (delta_t <= x_bounds[1] + 1.0)
            yv = self._normalize_coh(delta_t[mask], y[mask], x_bounds, normalize)
            lw = 1.2 if len(self.runs) <= 6 else 1.0
            ax.step(delta_t[mask], yv, where="mid", color=self.run_colors[idx], lw=lw,
                    alpha=0.85 if self.is_comparison else 1.0, label=self.labels[idx])
            if not self.is_comparison:
                ax.fill_between(delta_t[mask], yv, step="mid",
                                color=self.run_colors[idx], alpha=0.3)
        if integration_window_ns is not None:
            ax.axvspan(-integration_window_ns / 2, integration_window_ns / 2,
                       color="#ea4432", alpha=0.3, label=r"$\tau_{in}$ window")
        ax.set_xlim(x_bounds)
        if logy:
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

    def plot_coherence(self, c1=None, c2=None, time_window_ns=5.0, xlim=None,
                       integration_window_ns=None, normalize=None, logy=False,
                       ylim=None):
        """Coherence (correlation-histogram) spectrum. Single pair if (c1, c2)
        given, else the 5x3 master grid. Returns ``(fig, ax)`` / ``(fig, axes)``."""
        x_bounds = self._x_bounds(time_window_ns, xlim)
        if normalize is None and self.is_comparison:
            normalize = "peak"
        ylabel = self._coh_ylabel(normalize)

        if c1 is not None and c2 is not None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
            phys_name = self.runs[0]._get_physical_name(c1, c2)
            self._draw_coh(ax, c1, c2, x_bounds, normalize,
                           integration_window_ns, logy, ylim)
            ax.set_xlabel(r"$\Delta t$ (ns)")
            ax.set_ylabel(ylabel)
            self._decorate_single(
                fig, ax, rf"Coherence Spectrum ({self.histogram_source}): {phys_name}")
            return fig, ax

        fig, axes = plt.subplots(5, 3, figsize=(18, 20), dpi=100)
        for i in range(5):
            for j in range(3):
                ax = axes[i, j]
                try:
                    if i == 0:
                        ch1_ref = self.runs[0].get_ch(f"H{AUTO_COLS[j]}R")
                        ch2_ref = self.runs[0].get_ch(f"H{AUTO_COLS[j]}T")
                        title = rf"Coherence $H_{{{AUTO_COLS[j]}{AUTO_COLS[j]}}}$ (RT)"
                    else:
                        hA, hB = CROSS_COLS[j]
                        ch1_ref = self.runs[0].get_ch(f"H{hA}{CROSS_ROWS[i - 1][0]}")
                        ch2_ref = self.runs[0].get_ch(f"H{hB}{CROSS_ROWS[i - 1][1]}")
                        title = rf"Coherence $H_{{{hA}{hB}}}$ ({CROSS_ROWS[i - 1]})"
                except KeyError:
                    ax.set_visible(False)
                    continue
                self._draw_coh(ax, ch1_ref, ch2_ref, x_bounds, normalize,
                               integration_window_ns, logy, ylim)
                ax.set_title(title)
                if i == 4:
                    ax.set_xlabel(r"$\Delta t$ (ns)")
                if j == 0:
                    ax.set_ylabel(ylabel)
        self._decorate_grid(
            fig, axes, rf"Coherence Spectrum Matrix ({self.histogram_source} channels)")
        return fig, axes

    # =====================================================================
    # 2. g^(2)
    # =====================================================================

    def plot_g2(self, c1=None, c2=None, methods=None, tau_min=0.3, tau_max=30.0,
                step=0.6, xlim=None, ylim=(1, 3), num_side_peaks=3):
        """g^(2)(0) integration-window sweep. Single pair if (c1, c2) given, else
        the 5x3 grid. ``num_side_peaks`` sets how many side peaks (each side) the
        ``delay`` method averages as the uncorrelated reference. Returns
        ``(fig, ax)`` / ``(fig, axes)``."""
        if methods is None:
            methods = ["direct"]
        tau_in_ns = np.arange(tau_min, tau_max, step)

        if c1 is not None and c2 is not None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
            phys_name = self.runs[0]._get_physical_name(c1, c2)
            self._fill_ax_g2(ax, c1, c2, tau_in_ns, methods, xlim=xlim, ylim=ylim,
                             num_side_peaks=num_side_peaks)
            ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
            ax.set_ylabel(r"$g^{(2)}(0)$")
            self._decorate_single(fig, ax, rf"$g^{{(2)}}$ Integration Sweep: {phys_name}")
            return fig, ax

        fig, axes = plt.subplots(5, 3, figsize=(18, 20), dpi=100)
        for i in range(5):
            for j in range(3):
                ax = axes[i, j]
                try:
                    if i == 0:
                        ch1_ref = self.runs[0].get_ch(f"H{AUTO_COLS[j]}R")
                        ch2_ref = self.runs[0].get_ch(f"H{AUTO_COLS[j]}T")
                        title = rf"Auto $g^{{(2)}}_{{{AUTO_COLS[j]}{AUTO_COLS[j]}}}$ (RT)"
                    else:
                        hA, hB = CROSS_COLS[j]
                        ch1_ref = self.runs[0].get_ch(f"H{hA}{CROSS_ROWS[i - 1][0]}")
                        ch2_ref = self.runs[0].get_ch(f"H{hB}{CROSS_ROWS[i - 1][1]}")
                        title = rf"Cross $g^{{(2)}}_{{{hA}{hB}}}$ ({CROSS_ROWS[i - 1]})"
                except KeyError:
                    ax.set_visible(False)
                    continue
                self._fill_ax_g2(ax, ch1_ref, ch2_ref, tau_in_ns, methods,
                                 xlim=xlim, ylim=ylim, num_side_peaks=num_side_peaks)
                ax.set_title(title)
                if i == 4:
                    ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
                if j == 0:
                    ax.set_ylabel(r"$g^{(2)}(0)$")
        self._decorate_grid(fig, axes, r"$g^{(2)}$ Sweeping Matrix")
        return fig, axes

    def _fill_ax_g2(self, ax, ch1_ref, ch2_ref, tau_in_ns, methods, xlim=None,
                    ylim=(1, 3), num_side_peaks=3):
        max_y = 2.5
        has_data = False
        for run_idx, run in enumerate(self.runs):
            try:
                c1 = run.get_ch(self.runs[0].channel_map[ch1_ref])
                c2 = run.get_ch(self.runs[0].channel_map[ch2_ref])
            except KeyError:
                continue
            for m_idx, method in enumerate(methods):
                vals = []
                for tau in tau_in_ns:
                    if method == "delay":
                        vals.append(run.compute_g2_delay(c1, c2, tau,
                                                         num_side_peaks=num_side_peaks))
                    elif method == "heralded":
                        vals.append(run.compute_g2_heralded(c1, c2, tau))
                    else:
                        vals.append(run.compute_g2_direct(c1, c2, tau))
                valid = [v for v in vals if not np.isnan(v)]
                if valid:
                    has_data = True
                    max_y = min(max(max_y, max(valid) * 1.1), 4.5)
                color = (self.method_colors.get(method, "#333333")
                         if not self.is_comparison
                         else self.run_colors[run_idx % len(self.run_colors)])
                m_name = self.method_labels.get(method, method.capitalize())
                lbl = (m_name if not self.is_comparison
                       else f"{self.labels[run_idx]}"
                       + (f" ({m_name})" if len(methods) > 1 else ""))
                marker = "o" if method == "direct" else ("s" if method == "delay" else "^")
                linestyle = "-" if not self.is_comparison else ("-" if m_idx == 0 else "--")
                ax.plot(tau_in_ns, vals, marker=marker, linestyle=linestyle,
                        color=color, markersize=4, alpha=0.85, label=lbl)
        if not has_data:
            return
        final_max_y = max(1.2, max_y)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlim(xlim if xlim is not None else (tau_in_ns[0], tau_in_ns[-1]))
        ax.axhspan(0, 1, color="#2ecc71", alpha=0.05)
        ax.axhspan(1, 2, color="#f1c40f", alpha=0.05)
        ax.axhspan(2, 4, color="#e67e22", alpha=0.08)
        ax.axhspan(4, 100, color="#e74c3c", alpha=0.1)
        ax.axhline(y=1, color="#313131", linestyle="--", lw=1.5, alpha=0.6)
        if final_max_y > 2.0:
            ax.axhline(y=2, color="#313131", linestyle="--", lw=1.5, alpha=0.6)
        if final_max_y > 4.0:
            ax.axhline(y=4, color="#313131", linestyle="--", lw=1.5, alpha=0.6)
        ax.grid(True, alpha=0.3)

    # =====================================================================
    # 3. Cauchy-Schwarz R
    # =====================================================================

    def plot_R(self, cross_pair=None, auto_pair_1=None, auto_pair_2=None,
               methods=None, tau_min=0.3, tau_max=30.0, step=0.6, xlim=None,
               ylim=(0.8, 1.2), num_side_peaks=3):
        """Cauchy-Schwarz R integration-window sweep. Single R if the three pairs
        are given, else the 4x3 cross grid. ``num_side_peaks`` is forwarded to the
        ``delay`` g^(2). Returns ``(fig, ax)`` / ``(fig, axes)``."""
        if methods is None:
            methods = ["direct"]
        tau_in_ns = np.arange(tau_min, tau_max, step)

        if cross_pair is not None and auto_pair_1 is not None and auto_pair_2 is not None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
            phys_name = self.runs[0]._get_physical_name(cross_pair[0], cross_pair[1])
            self._fill_ax_R(ax, cross_pair, auto_pair_1, auto_pair_2, tau_in_ns,
                            methods, xlim=xlim, ylim=ylim, num_side_peaks=num_side_peaks)
            ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
            ax.set_ylabel(r"$R$ parameter")
            self._decorate_single(fig, ax, rf"Cauchy-Schwarz $R$ Sweep: {phys_name}")
            return fig, ax

        fig, axes = plt.subplots(4, 3, figsize=(18, 16), dpi=100)
        for i, row in enumerate(CROSS_ROWS):
            for j, (hA, hB) in enumerate(CROSS_COLS):
                ax = axes[i, j]
                try:
                    autoA_ref = (self.runs[0].get_ch(f"H{hA}R"), self.runs[0].get_ch(f"H{hA}T"))
                    autoB_ref = (self.runs[0].get_ch(f"H{hB}R"), self.runs[0].get_ch(f"H{hB}T"))
                    cross_ref = (self.runs[0].get_ch(f"H{hA}{row[0]}"),
                                 self.runs[0].get_ch(f"H{hB}{row[1]}"))
                except KeyError:
                    ax.set_visible(False)
                    continue
                self._fill_ax_R(ax, cross_ref, autoA_ref, autoB_ref, tau_in_ns,
                                methods, xlim=xlim, ylim=ylim,
                                num_side_peaks=num_side_peaks)
                ax.set_title(rf"$R_{{{hA}{hB}}}$ ({row})")
                if i == 3:
                    ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
                if j == 0:
                    ax.set_ylabel(r"$R$ parameter")
        self._decorate_grid(fig, axes, r"Cauchy-Schwarz $R$ Sweeping Matrix")
        return fig, axes

    def _fill_ax_R(self, ax, cross_ref, autoA_ref, autoB_ref, tau_in_ns, methods,
                   xlim=None, ylim=(0.8, 1.2), num_side_peaks=3):
        has_data = False
        for run_idx, run in enumerate(self.runs):
            try:
                autoA = (run.get_ch(self.runs[0].channel_map[autoA_ref[0]]),
                         run.get_ch(self.runs[0].channel_map[autoA_ref[1]]))
                autoB = (run.get_ch(self.runs[0].channel_map[autoB_ref[0]]),
                         run.get_ch(self.runs[0].channel_map[autoB_ref[1]]))
                cross = (run.get_ch(self.runs[0].channel_map[cross_ref[0]]),
                         run.get_ch(self.runs[0].channel_map[cross_ref[1]]))
            except KeyError:
                continue
            for m_idx, method in enumerate(methods):
                R_vals = []
                for tau in tau_in_ns:
                    if method == "delay":
                        c = run.compute_g2_delay(cross[0], cross[1], tau, num_side_peaks=num_side_peaks)
                        aA = run.compute_g2_delay(autoA[0], autoA[1], tau, num_side_peaks=num_side_peaks)
                        aB = run.compute_g2_delay(autoB[0], autoB[1], tau, num_side_peaks=num_side_peaks)
                    elif method == "heralded":
                        c = run.compute_g2_heralded(cross[0], cross[1], tau)
                        aA = run.compute_g2_heralded(autoA[0], autoA[1], tau)
                        aB = run.compute_g2_heralded(autoB[0], autoB[1], tau)
                    else:
                        c = run.compute_g2_direct(cross[0], cross[1], tau)
                        aA = run.compute_g2_direct(autoA[0], autoA[1], tau)
                        aB = run.compute_g2_direct(autoB[0], autoB[1], tau)
                    R_vals.append(run.compute_R_parameter(c, aA, aB))
                if any(not np.isnan(r) for r in R_vals):
                    has_data = True
                color = (self.method_colors.get(method, "#333333")
                         if not self.is_comparison
                         else self.run_colors[run_idx % len(self.run_colors)])
                m_name = self.method_labels.get(method, method.capitalize())
                lbl = (m_name if not self.is_comparison
                       else f"{self.labels[run_idx]}"
                       + (f" ({m_name})" if len(methods) > 1 else ""))
                marker = "o" if method == "direct" else ("s" if method == "delay" else "^")
                linestyle = "-" if not self.is_comparison else ("-" if m_idx == 0 else "--")
                ax.plot(tau_in_ns, R_vals, marker=marker, linestyle=linestyle,
                        color=color, markersize=5, alpha=0.85, label=lbl)
        if not has_data:
            return
        ax.axhline(y=1, color="#7f8c8d", linestyle="--", lw=1.5, alpha=0.6)
        ax.axhspan(0, 1, color="#9b59b6", alpha=0.08)
        ax.axhspan(1, 100, color="#dd76b4", alpha=0.15)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlim(xlim if xlim is not None else (tau_in_ns[0], tau_in_ns[-1]))
        ax.grid(True, alpha=0.3)

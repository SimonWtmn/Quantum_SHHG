"""
=============================================================================
HBT Power-Scan Analyzer
=============================================================================

Tools to study how the harmonics of a high-harmonic-generation source behave when the driving (pump) intensity I_0 is swept. 
One acquisition file per power; this module turns that family of runs into the three diagnostic plots 
used to test the intensity-fluctuation model of the harmonic photon statistics.

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquee (LOA), Ecole Polytechnique
Date: 15/06/2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
    "figure.titlesize": 18,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# Two clearly distinct, colour-blind-friendly palettes: one for the auto-correlations
# (solid + circles) and a separate one for the cross-correlations (dashed + squares), so the
# two families never share a colour and stay readable even in greyscale.
_AUTO_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
_CROSS_COLORS = ['#ff7f0e', '#17becf', '#bcbd22', '#7f7f7f', '#1a9850', '#9e0142']
_IDEAL_GREY = '0.45'


class PowerScanAnalyzer:
    """Analyse a power scan: one HBTMeasurement per pump power, same filter/polarisation.

    Parameters
    ----------
    runs : list[HBTMeasurement]
        The runs of the scan (any order; runs without a known power are dropped).
    harmonics : tuple[int]
        Harmonic orders to analyse (default (3, 4, 5)).
    tau_in_ns : float
        Integration window for the delay g^(2) (the recommended method).
    g2_method : {'delay', 'direct', 'heralded'}
        How g^(2)(0) is computed at each power. 'delay' (default) uses the physical histogram
        when available.
    g2_source : {'auto', 'physical', 'virtual'}
        Histogram source for the delay method.
    intensity : {'countrate', 'counts'}
        Observable used as the harmonic intensity <I_n> (default count rate, counts/s).
    """

    def __init__(self, runs, harmonics=(3, 4, 5), tau_in_ns=4.0,
                 g2_method='delay', g2_source='auto', intensity='countrate'):
        self.runs = sorted([r for r in runs if r.power_mw is not None], key=lambda r: r.power_mw)
        if len(self.runs) < 2:
            raise ValueError("A power scan needs at least two runs with a known power.")
        self.harmonics = tuple(harmonics)
        self.tau_in_ns = tau_in_ns
        self.g2_method = g2_method
        self.g2_source = g2_source
        self.intensity = intensity
        self.hcolor = {n: _AUTO_COLORS[i % len(_AUTO_COLORS)] for i, n in enumerate(self.harmonics)}

        self._build()





    # ---------------- data assembly ----------------

    def _g2_pair(self, run, a_name, b_name):
        c1, c2 = run.get_ch(a_name), run.get_ch(b_name)
        if self.g2_method == 'direct':
            return run.compute_g2_direct(c1, c2)
        if self.g2_method == 'heralded':
            return run.compute_g2_heralded(c1, c2, self.tau_in_ns)
        return run.compute_g2_delay(c1, c2, self.tau_in_ns, source=self.g2_source)


    def _build(self):
        self.I0 = np.array([r.pump_intensity() for r in self.runs], dtype=float)
        log_I0 = np.log(self.I0)

        self.In, self.K, self.g2_auto = {}, {}, {}
        for n in self.harmonics:
            In = np.array([r.harmonic_intensity(n, self.intensity) for r in self.runs], dtype=float)
            self.In[n] = In
            with np.errstate(divide='ignore', invalid='ignore'):
                self.K[n] = np.gradient(np.log(In), log_I0)
            self.g2_auto[n] = np.array(
                [self._g2_pair(r, f"H{n}T", f"H{n}R") for r in self.runs], dtype=float)

        self.g2_cross = {}
        for i, m in enumerate(self.harmonics):
            for n in self.harmonics[i + 1:]:
                self.g2_cross[(m, n)] = np.array(
                    [self._g2_pair(r, f"H{m}T", f"H{n}T") for r in self.runs], dtype=float)
        self.ccolor = {pair: _CROSS_COLORS[i % len(_CROSS_COLORS)]
                       for i, pair in enumerate(self.g2_cross)}

        self.header = self._build_header()


    def _build_header(self):
        r0 = self.runs[0]
        bits = []
        if r0.material and r0.material != 'Unknown':
            bits.append(rf"Sample: {r0.material}")
        if r0.wavelength_nm:
            bits.append(rf"$\lambda_L = {r0.wavelength_nm:g}$ nm")
        if r0.filter_label:
            bits.append(rf"Filter: {r0.filter_label}")
        if r0.polarization:
            bits.append(r0.polarization)
        bits.append(rf"$P = {self.I0.min():g}$--${self.I0.max():g}$ mW ({len(self.I0)} pts)")
        method = {'delay': 'delay', 'direct': 'direct', 'heralded': 'heralded'}[self.g2_method]
        bits.append(rf"$g^{{(2)}}$: {method} ($\tau_{{in}} = {self.tau_in_ns:g}$ ns)")
        return "  |  ".join(bits)




    # ---------------- numeric helpers ----------------

    def perturbative_slope(self, n, n_fit=None):
        """Power-law exponent K(n) from a straight-line fit of log<I_n> vs log I_0 over the
        lowest `n_fit` powers (the perturbative regime). Returns (slope, intercept)."""
        x, y = np.log(self.I0), np.log(self.In[n])
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        if n_fit is None:
            n_fit = max(3, len(x) // 2)
        n_fit = min(n_fit, len(x))
        slope, intercept = np.polyfit(x[:n_fit], y[:n_fit], 1)
        return slope, intercept


    def collapse_auto(self, n, slope='local', n_fit=None):
        """(g2_nn - 1) / K^2(n) for harmonic n (should equal g2_0 - 1 for every harmonic)."""
        k = self.K[n] if slope == 'local' else self.perturbative_slope(n, n_fit)[0]
        return (self.g2_auto[n] - 1.0) / np.asarray(k) ** 2


    def collapse_cross(self, m, n, slope='local', n_fit=None):
        """(g2_mn - 1) / (K(m) K(n)) for a cross pair (should also equal g2_0 - 1)."""
        if slope == 'local':
            km, kn = self.K[m], self.K[n]
        else:
            km, kn = self.perturbative_slope(m, n_fit)[0], self.perturbative_slope(n, n_fit)[0]
        return (self.g2_cross[(m, n)] - 1.0) / n*m
        # return (self.g2_cross[(m, n)] - 1.0) / (np.asarray(km) * np.asarray(kn))


    def inferred_g2_0(self, slope='local', n_fit=None):
        """Indirect estimate of the pump excess g2_0 - 1 = sigma^2, as the mean over all
        harmonics/pairs of the collapsed curves (returns mean and std arrays vs power)."""
        stack = [self.collapse_auto(n, slope, n_fit) for n in self.harmonics]
        stack += [self.collapse_cross(m, n, slope, n_fit) for (m, n) in self.g2_cross]
        arr = np.vstack(stack)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


    def summary_table(self):
        """Print one row per power: I_0, <I_n>, K(n), g2_nn and the collapsed (g2-1)/K^2."""
        h = self.harmonics
        cols = ["P(mW)"] + [f"I{n}" for n in h] + [f"K{n}" for n in h] \
               + [f"g2_{n}{n}" for n in h] + [f"(g-1)/K2_{n}" for n in h]
        widths = [8] + [11] * len(h) + [7] * len(h) + [8] * len(h) + [12] * len(h)
        hdr = "".join(f"{c:>{w}s}" for c, w in zip(cols, widths))
        print(hdr); print("-" * len(hdr))
        coll = {n: self.collapse_auto(n) for n in h}
        for i in range(len(self.I0)):
            row = [f"{self.I0[i]:8.1f}"]
            row += [f"{self.In[n][i]:11.4g}" for n in h]
            row += [f"{self.K[n][i]:7.2f}" for n in h]
            row += [f"{self.g2_auto[n][i]:8.3f}" for n in h]
            row += [f"{coll[n][i]:12.4f}" for n in h]
            print("".join(row))




    # ---------------- plots (each accepts an optional ax for composition) ----------------

    @staticmethod
    def _ax(ax, figsize=(9, 6.5)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=300)
            return fig, ax, True
        return ax.figure, ax, False


    def _finish(self, fig, ax, own, title, handles=None, labels=None, legend_kw=None):
        """Title block + legend for a standalone figure (own=True).

        The descriptive title sits on top (suptitle); the acquisition metadata is the smaller
        grey sub-title placed just above the axes (so the order reads title -> metadata -> plot,
        not the other way round). For composed panels (own=False) only the legend is drawn.
        """
        lk = dict(framealpha=0.92, edgecolor='0.7', fontsize=10)
        if legend_kw:
            lk.update(legend_kw)
        if handles is not None:
            ax.legend(handles, labels, **lk)
        else:
            ax.legend(**lk)
        if own:
            fig.suptitle(title, fontsize=16, y=0.94)
            ax.set_title(self.header, fontsize=9.5, color='#555555', loc="center")
            fig.tight_layout()
            plt.show()
        return fig, ax


    def plot_intensity_scaling(self, ax=None, n_fit=None, show_ideal=True):
        """Plot 3: log <I_n> vs log I_0. Straight (slope ~ n) then bends at saturation.

        Markers = data, solid line = perturbative power-law fit (slope K reported per harmonic),
        thin grey dashes = the ideal slope K=n anchored on the lowest-power point.
        """
        fig, ax, own = self._ax(ax)
        xf = np.array([self.I0.min(), self.I0.max()])
        for n in self.harmonics:
            c = self.hcolor[n]
            In = self.In[n]
            slope, intercept = self.perturbative_slope(n, n_fit)
            ax.loglog(self.I0, In, 'o', color=c, ms=8,
                      label=rf"$H_{n}$  ($K_{{\mathrm{{fit}}}}={slope:.2f}$, ideal {n})")
            ax.loglog(xf, np.exp(intercept) * xf ** slope, '-', color=c, lw=2.0)
            if show_ideal:
                ax.loglog(xf, In[0] * (xf / self.I0[0]) ** n, '--', color=_IDEAL_GREY, lw=1.1)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"Harmonic intensity $\langle I_n\rangle$ (counts/s)")
        ax.grid(True, which='both', alpha=0.25)
        h, lab = ax.get_legend_handles_labels()
        if show_ideal:
            h.append(Line2D([0], [0], color=c, ls='--', lw=1.1))
            lab.append(r"ideal slope $K=n$")
        return self._finish(fig, ax, own, r"Harmonic intensity scaling $\langle I_n\rangle \sim I_0^{K(n)}$",
                            h, lab, legend_kw=dict(loc='lower right'))


    def plot_local_slope(self, ax=None):
        """Local exponent K(n) = d ln<I_n>/d ln I_0 vs power; departs from n at saturation."""
        fig, ax, own = self._ax(ax, figsize=(9, 6))
        for n in self.harmonics:
            c = self.hcolor[n]
            ax.semilogx(self.I0, self.K[n], 'o-', color=c, ms=7, label=rf"$K({n})$ measured")
            ax.axhline(n, color=c, ls=':', lw=1.4, alpha=0.7)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"Local exponent $K(n) = \mathrm{d}\ln\langle I_n\rangle/\mathrm{d}\ln I_0$")
        ax.grid(True, which='both', alpha=0.25)
        h, lab = ax.get_legend_handles_labels()
        h.append(Line2D([0], [0], color=_IDEAL_GREY, ls=':', lw=1.4))
        lab.append(r"ideal order $K=n$")
        return self._finish(fig, ax, own, r"Effective nonlinearity $K(n)$",
                            h, lab, legend_kw=dict(loc='best'))


    def plot_g2_vs_power(self, ax=None, include_cross=False):
        """Plot 1: g^(2) vs I_0 — a family of distinct curves (one per harmonic / pair)."""
        fig, ax, own = self._ax(ax)
        for n in self.harmonics:
            ax.plot(self.I0, self.g2_auto[n], 'o-', color=self.hcolor[n], ms=7,
                    label=rf"$g^{{(2)}}_{{{n}{n}}}$ auto")
        if include_cross:
            for (m, n) in self.g2_cross:
                ax.plot(self.I0, self.g2_cross[(m, n)], 's--', color=self.ccolor[(m, n)],
                        ms=6, lw=1.6, label=rf"$g^{{(2)}}_{{{m}{n}}}$ cross")
        ax.axhline(1.0, color='#313131', ls='--', lw=1.3, alpha=0.7)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$g^{(2)}(0)$")
        ax.set_ylim(1, 1.4)
        h, lab = ax.get_legend_handles_labels()
        h.append(Line2D([0], [0], color='#313131', ls='--', lw=1.3))
        lab.append(r"uncorrelated ($g^{(2)}=1$)")
        return self._finish(fig, ax, own, r"$g^{(2)}(0)$ vs pump power",
                            h, lab, legend_kw=dict(ncol=2, loc='best'))


    def plot_g2_collapse(self, ax=None, slope='local', include_cross=True, show_mean=True):
        """Plot 2: (g^(2)_n - 1)/K^2(n) vs I_0 — all harmonics should COLLAPSE onto g2_0 - 1.

        Auto pairs are divided by K^2(n), cross pairs by K(m)K(n). The black line is the mean
        over all pairs (the inferred pump excess g2_0 - 1), the grey band its 1-sigma spread.
        """
        fig, ax, own = self._ax(ax)
        for n in self.harmonics:
            ax.plot(self.I0, self.collapse_auto(n, slope), 'o-', color=self.hcolor[n], ms=7,
                    label=rf"$H_{n}$ auto $/K^2$")
        if include_cross:
            for (m, n) in self.g2_cross:
                ax.plot(self.I0, self.collapse_cross(m, n, slope), 's--', color=self.ccolor[(m, n)],
                        ms=6, lw=1.6, label=rf"$H_{m}H_{n}$ cross $/K_mK_n$")
        extra_h, extra_l = [], []
        if show_mean:
            mean, std = self.inferred_g2_0(slope)
            ax.plot(self.I0, mean, 'k-', lw=2.8, label=r"mean $= g^{(2)}_0 - 1$")
            ax.fill_between(self.I0, mean - std, mean + std, color='k', alpha=0.13)
            extra_h.append(Patch(facecolor='k', alpha=0.13))
            extra_l.append(r"$\pm1\sigma$ across harmonics")
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$\left(g^{(2)}_n - 1\right)/K^2(n)$")
        ax.set_ylim(-0.05, 0.05)
        h, lab = ax.get_legend_handles_labels()
        h += extra_h; lab += extra_l
        return self._finish(fig, ax, own,
                            r"Rescaled collapse $\;\to\; g^{(2)}_0 - 1 = \sigma^2$",
                            h, lab, legend_kw=dict(ncol=2, loc='upper right'))


    def plot_overview(self, slope='local', n_fit=None):
        """2x2 dashboard mirroring the model: g^(2) vs power, the rescaled collapse, the
        intensity scaling (log-log) and the local exponent K(n)."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
        self.plot_g2_vs_power(ax=axes[0, 0], include_cross=True)
        axes[0, 0].set_title(r"(1) $g^{(2)}(0)$ vs pump power", fontsize=15)
        self.plot_g2_collapse(ax=axes[0, 1], slope=slope)
        axes[0, 1].set_title(r"(2) Rescaled collapse $\to g^{(2)}_0 - 1 = \sigma^2$", fontsize=15)
        self.plot_intensity_scaling(ax=axes[1, 0], n_fit=n_fit)
        axes[1, 0].set_title(r"(3) Intensity scaling $\langle I_n\rangle \sim I_0^{K(n)}$", fontsize=15)
        self.plot_local_slope(ax=axes[1, 1])
        axes[1, 1].set_title(r"(4) Local exponent $K(n)$  (dotted = ideal $n$)", fontsize=15)
        fig.suptitle("Power scan — harmonic fluctuation model", fontsize=19, y=0.96)
        fig.text(0.5, 0.93, self.header, ha='center', fontsize=11, color='#555555')
        fig.subplots_adjust(top=0.91, hspace=0.20, wspace=0.20,
                            left=0.07, right=0.97, bottom=0.06)
        plt.show()
        return fig, axes

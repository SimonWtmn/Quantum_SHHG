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

        # Per-detector (physical channel) intensities, so the scaling can be shown arm-by-arm
        # (H3T, H3R, H4T, ...) and not only as the merged harmonic Hn.
        self.In_ch, self.chan_names, self.chcolor = {}, [], {}
        for hi, n in enumerate(self.harmonics):
            for arm in ('T', 'R'):
                name = f"H{n}{arm}"
                vals, ok = [], True
                for r in self.runs:
                    try:
                        ch = r.get_ch(name)
                    except KeyError:
                        ok = False
                        break
                    vals.append(r.channel_intensity(ch, self.intensity))
                if ok:
                    self.In_ch[name] = np.array(vals, dtype=float)
                    self.chan_names.append(name)
                    # T arm shares the harmonic's auto colour; R arm uses the cross palette.
                    palette = _AUTO_COLORS if arm == 'T' else _CROSS_COLORS
                    self.chcolor[name] = palette[hi % len(palette)]

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
        bits.append(rf"$g^{{(2)}}$: {self.g2_method} ($\tau_{{in}} = {self.tau_in_ns:g}$ ns)")
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


    @staticmethod
    def _line_sse(x, y):
        """Sum of squared residuals of a straight-line (log-log) fit; 0 for <2 points."""
        if len(x) < 2:
            return 0.0
        slope, intercept = np.polyfit(x, y, 1)
        return float(np.sum((y - (slope * x + intercept)) ** 2))


    def _auto_breakpoint(self, x, y, min_pts=2):
        """Index `b` splitting (x, y) into a low-power segment [:b] and a high-power
        (saturation) segment [b:] so that the combined two-line residual is minimal.
        Requires at least `min_pts` points on each side; returns None if impossible."""
        best_b, best_sse = None, np.inf
        for b in range(min_pts, len(x) - min_pts + 1):
            sse = self._line_sse(x[:b], y[:b]) + self._line_sse(x[b:], y[b:])
            if sse < best_sse:
                best_sse, best_b = sse, b
        return best_b


    def two_segment_fit(self, P, y, n_fit=None, split=None, min_pts=2):
        """Two free-slope power-law fits of y vs P in log-log: a low-power *perturbative* branch
        and a high-power *saturation* branch (each slope K is fitted, not imposed).

        `split` chooses where the perturbative regime ends and saturation begins:
          * None     -> auto-detected breakpoint (minimal combined residual); `n_fit`, if given,
                        is used instead as the number of perturbative (low-power) points,
          * int      -> fixed index (first point of the saturation branch).

        Returns a dict with 'lo'/'hi' = (slope, intercept) (each None if absent), the breakpoint
        index 'b', the saturation onset power 'P_sat', and the masked (P, y) actually used.
        """
        P, y = np.asarray(P, float), np.asarray(y, float)
        good = np.isfinite(P) & np.isfinite(y) & (y > 0)
        Pg, yg = P[good], y[good]
        lx, ly = np.log(Pg), np.log(yg)

        out = {'lo': None, 'hi': None, 'b': None, 'P_sat': None, 'P': Pg, 'y': yg}
        if len(Pg) < 2:
            return out
        if len(Pg) < 2 * min_pts:
            out['lo'] = tuple(np.polyfit(lx, ly, 1))
            return out

        if split is not None:
            b = int(split)
        elif n_fit is not None:
            b = int(n_fit)
        else:
            b = self._auto_breakpoint(lx, ly, min_pts)
        if b is not None:
            b = max(min_pts, min(int(b), len(Pg) - min_pts))

        if b is None:
            out['lo'] = tuple(np.polyfit(lx, ly, 1))
            return out

        out['b'] = b
        out['P_sat'] = float(Pg[b])
        out['lo'] = tuple(np.polyfit(lx[:b], ly[:b], 1))
        out['hi'] = tuple(np.polyfit(lx[b:], ly[b:], 1))
        return out


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
        return (self.g2_cross[(m, n)] - 1.0) / (np.asarray(km) * np.asarray(kn))


    def inferred_g2_0(self, slope='local', n_fit=None, harmonics=None, pairs=None):
        """Indirect estimate of the pump excess g2_0 - 1 = sigma^2, as the mean (and std) over the
        collapsed curves vs power. `harmonics`/`pairs` restrict which curves enter the average
        (default: all); exclude cross pairs entirely with `pairs=[]`. Useful to drop a degenerate
        harmonic (e.g. one with K~0, whose 1/K^2 collapse blows up). Returns NaN arrays if empty."""
        keep_h, keep_p = self._resolve_keep(harmonics, pairs)
        stack = [self.collapse_auto(n, slope, n_fit) for n in self.harmonics if n in keep_h]
        stack += [self.collapse_cross(m, n, slope, n_fit) for (m, n) in self.g2_cross
                  if self._keep_pair(m, n, keep_h, keep_p)]
        if not stack:
            nan = np.full(len(self.I0), np.nan)
            return nan, nan
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


    def plot_intensity_scaling(self, ax=None, n_fit=None, split=None, per_channel=True,
                               harmonics=None, channels=None, show_ideal=True,
                               xlim=None, ylim=None):
        """Plot 3: intensity vs I_0 (log-log) with two fitted power laws per curve.

        Markers = data, solid line = perturbative (low-power) fit, dashed line = saturation
        (high-power) fit (both slopes K are fitted), and an open ring marks the breakpoint where
        the scaling rolls over into saturation. Thin grey dots = the ideal slope K=n anchored on
        the lowest-power point.

        * per_channel=True (default): one curve per physical detector (H3T, H3R, H4T, ...).
        * per_channel=False: one curve per merged harmonic (H3, H4, H5).

        Toggle what is shown (handy when a cut produces no Hn, e.g. GaAs100 has no H4):
          * `harmonics` -> iterable of orders to keep, e.g. (3, 5) (default: all),
          * `channels`  -> explicit list of channel names to keep, e.g. ['H3T', 'H5R']
                           (per-channel mode only; takes precedence over `harmonics`).

        The breakpoint is auto-detected; override with `split` (int index, or a dict keyed by the
        curve name / harmonic) or `n_fit` (number of perturbative points). `xlim`/`ylim` set the
        axis ranges (each a (lo, hi) tuple); None keeps autoscale.
        """
        fig, ax, own = self._ax(ax)
        keep_h, _ = self._resolve_keep(harmonics, None)
        keep_c = set(channels) if channels is not None else None
        if per_channel:
            series = [(name, int(name[1:-1]), self.chcolor[name], self.In_ch[name])
                      for name in self.chan_names]
            if keep_c is not None:
                series = [s for s in series if s[0] in keep_c]
            else:
                series = [s for s in series if s[1] in keep_h]
            label_of = lambda name, n: rf"$H_{{{n}}}${name[len(str(n)) + 1:]}"
        else:
            series = [(rf"H{n}", n, self.hcolor[n], self.In[n])
                      for n in self.harmonics if n in keep_h]
            label_of = lambda name, n: rf"$H_{{{n}}}$"

        any_sat = False
        for name, n, c, y in series:
            sp = split.get(name, split.get(n)) if isinstance(split, dict) else split
            f = self.two_segment_fit(self.I0, y, n_fit=n_fit, split=sp)
            P, lo, hi, b = f['P'], f['lo'], f['hi'], f['b']
            if len(P) == 0:
                continue
            ax.loglog(P, f['y'], 'o', color=c, ms=7)
            lab = label_of(name, n)
            if lo is not None and hi is not None and b is not None:
                any_sat = True
                x_lo = np.array([P[0], P[b]])
                x_hi = np.array([P[b], P[-1]])
                ax.loglog(x_lo, np.exp(lo[1]) * x_lo ** lo[0], '-', color=c, lw=2.0)
                ax.loglog(x_hi, np.exp(hi[1]) * x_hi ** hi[0], '--', color=c, lw=2.0)
                ax.loglog([f['P_sat']], [f['y'][b]], 'o', mfc='none', mec=c, mew=2.0, ms=14)
                lab += rf": $K={lo[0]:.2f}\!\to\!{hi[0]:.2f}$ (ideal {n})"
            elif lo is not None:
                xf = np.array([P.min(), P.max()])
                ax.loglog(xf, np.exp(lo[1]) * xf ** lo[0], '-', color=c, lw=2.0)
                lab += rf": $K={lo[0]:.2f}$ (ideal {n})"
            if show_ideal:
                xf = np.array([P.min(), P.max()])
                ax.loglog(xf, f['y'][0] * (xf / P[0]) ** n, ':', color=_IDEAL_GREY, lw=1.0)
            ax.plot([], [], '-', color=c, label=lab)

        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"Intensity (counts/s)")
        ax.grid(True, which='both', alpha=0.25)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        h, lab = ax.get_legend_handles_labels()
        if any_sat:
            h.append(Line2D([0], [0], marker='o', mfc='none', mec='0.3', mew=2.0, ls='none', ms=11))
            lab.append(r"saturation onset")
        if show_ideal:
            h.append(Line2D([0], [0], color=_IDEAL_GREY, ls=':', lw=1.0))
            lab.append(r"ideal slope $K=n$")
        ncol = 2 if len(series) > 3 else 1
        return self._finish(fig, ax, own, r"Harmonic intensity scaling $\langle I_n\rangle \sim I_0^{K(n)}$",
                            h, lab, legend_kw=dict(loc='lower right', ncol=ncol))


    def plot_local_slope(self, ax=None, xlim=None, ylim=None):
        """Local exponent K(n) = d ln<I_n>/d ln I_0 vs power; departs from n at saturation.
        `xlim`/`ylim` set the axis ranges ((lo, hi) tuples); None keeps autoscale."""
        fig, ax, own = self._ax(ax, figsize=(9, 6))
        for n in self.harmonics:
            c = self.hcolor[n]
            ax.semilogx(self.I0, self.K[n], 'o-', color=c, ms=7, label=rf"$K({n})$ measured")
            ax.axhline(n, color=c, ls=':', lw=1.4, alpha=0.7)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"Local exponent $K(n) = \mathrm{d}\ln\langle I_n\rangle/\mathrm{d}\ln I_0$")
        ax.grid(True, which='both', alpha=0.25)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        h, lab = ax.get_legend_handles_labels()
        h.append(Line2D([0], [0], color=_IDEAL_GREY, ls=':', lw=1.4))
        lab.append(r"ideal order $K=n$")
        return self._finish(fig, ax, own, r"Effective nonlinearity $K(n)$",
                            h, lab, legend_kw=dict(loc='best'))


    def _resolve_keep(self, harmonics, pairs):
        """Normalise the selection arguments shared by the power-scan plots into a harmonic set
        `keep_h` (all harmonics when `harmonics` is None) and an explicit cross-pair set `keep_p`
        (None when `pairs` is not given)."""
        keep_h = set(self.harmonics if harmonics is None else harmonics)
        keep_p = {tuple(sorted(p)) for p in pairs} if pairs is not None else None
        return keep_h, keep_p


    @staticmethod
    def _keep_pair(m, n, keep_h, keep_p):
        """Whether a cross pair (m, n) should be shown given the harmonic set `keep_h` and the
        optional explicit pair set `keep_p` (which takes precedence)."""
        if keep_p is not None:
            return tuple(sorted((m, n))) in keep_p
        return m in keep_h and n in keep_h


    def plot_g2_vs_power(self, ax=None, include_cross=False, harmonics=None, pairs=None,
                         xlim=None, ylim=(0.9, 1.6)):
        """Plot 1: g^(2) vs I_0 — a family of distinct curves (one per harmonic / pair).

        Toggle what is shown (e.g. to drop a harmonic a given cut does not produce):
          * `harmonics` -> iterable of orders to keep (default: all); also restricts which cross
                           pairs appear (both members must be kept),
          * `pairs`     -> explicit list of cross pairs to keep, e.g. [(3, 5)] (takes precedence).

        `ylim` defaults to (0.9, 1.6) to zoom on the fluctuations; pass another (lo, hi) tuple
        to rescale or None to autoscale. `xlim` works the same way on the power axis.
        """
        fig, ax, own = self._ax(ax)
        keep_h, keep_p = self._resolve_keep(harmonics, pairs)
        for n in self.harmonics:
            if n not in keep_h:
                continue
            ax.plot(self.I0, self.g2_auto[n], 'o-', color=self.hcolor[n], ms=7,
                    label=rf"$g^{{(2)}}_{{{n}{n}}}$ auto")
        if include_cross:
            for (m, n) in self.g2_cross:
                if not self._keep_pair(m, n, keep_h, keep_p):
                    continue
                ax.plot(self.I0, self.g2_cross[(m, n)], 's--', color=self.ccolor[(m, n)],
                        ms=6, lw=1.6, label=rf"$g^{{(2)}}_{{{m}{n}}}$ cross")
        ax.axhline(1.0, color='#313131', ls='--', lw=1.3, alpha=0.7)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$g^{(2)}(0)$")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        h, lab = ax.get_legend_handles_labels()
        h.append(Line2D([0], [0], color='#313131', ls='--', lw=1.3))
        lab.append(r"uncorrelated ($g^{(2)}=1$)")
        return self._finish(fig, ax, own, r"$g^{(2)}(0)$ vs pump power",
                            h, lab, legend_kw=dict(ncol=2, loc='best'))


    def plot_g2_collapse(self, ax=None, slope='local', include_cross=True, show_mean=True,
                         harmonics=None, pairs=None, xlim=None, ylim=(0, 0.015)):
        """Plot 2: (g^(2)_n - 1)/K^2(n) vs I_0 — all harmonics should COLLAPSE onto g2_0 - 1.

        Auto pairs are divided by K^2(n), cross pairs by K(m)K(n). The black line is the mean
        over the shown pairs (the inferred pump excess g2_0 - 1), the grey band its 1-sigma spread.

        Toggle what is shown (and what the mean is taken over):
          * `harmonics` -> iterable of orders to keep (default: all); also restricts cross pairs,
          * `pairs`     -> explicit list of cross pairs to keep, e.g. [(3, 5)] (takes precedence).

        `xlim`/`ylim` set the axis ranges ((lo, hi) tuples); pass None to autoscale (handy when
        the collapsed values fall outside the default zoom window).
        """
        fig, ax, own = self._ax(ax)
        keep_h, keep_p = self._resolve_keep(harmonics, pairs)
        for n in self.harmonics:
            if n not in keep_h:
                continue
            ax.plot(self.I0, self.collapse_auto(n, slope), 'o-', color=self.hcolor[n], ms=7,
                    label=rf"$H_{n}$ auto $/K^2$")
        if include_cross:
            for (m, n) in self.g2_cross:
                if not self._keep_pair(m, n, keep_h, keep_p):
                    continue
                ax.plot(self.I0, self.collapse_cross(m, n, slope), 's--', color=self.ccolor[(m, n)],
                        ms=6, lw=1.6, label=rf"$H_{m}H_{n}$ cross $/K_mK_n$")
        extra_h, extra_l = [], []
        if show_mean:
            mean, std = self.inferred_g2_0(slope, harmonics=harmonics,
                                           pairs=(pairs if include_cross else []))
            ax.plot(self.I0, mean, 'k-', lw=2.8, label=r"mean $= g^{(2)}_0 - 1$")
            ax.fill_between(self.I0, mean - std, mean + std, color='k', alpha=0.13)
            extra_h.append(Patch(facecolor='k', alpha=0.13))
            extra_l.append(r"$\pm1\sigma$ across harmonics")
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$\left(g^{(2)}_n - 1\right)/K^2(n)$")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        h, lab = ax.get_legend_handles_labels()
        h += extra_h; lab += extra_l
        return self._finish(fig, ax, own,
                            r"Rescaled collapse $\;\to\; g^{(2)}_0 - 1 = \sigma^2$",
                            h, lab, legend_kw=dict(ncol=2, loc='upper right'))


    def plot_overview(self, slope='local', n_fit=None, split=None, per_channel=True,
                      harmonics=None, channels=None, pairs=None,
                      g2_ylim=(0.9, 1.6), collapse_ylim=(0, 0.015)):
        """2x2 dashboard mirroring the model: g^(2) vs power, the rescaled collapse, the
        intensity scaling (log-log) and the local exponent K(n).

        `g2_ylim`/`collapse_ylim` zoom panels (1) and (2) (None autoscales). `harmonics`/`pairs`
        select which harmonics/cross pairs are shown across panels (1)-(3); `channels`/`split`/
        `n_fit`/`per_channel` further control the intensity-scaling fit of panel (3)."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
        self.plot_g2_vs_power(ax=axes[0, 0], include_cross=True, harmonics=harmonics,
                              pairs=pairs, ylim=g2_ylim)
        axes[0, 0].set_title(r"(1) $g^{(2)}(0)$ vs pump power", fontsize=15)
        self.plot_g2_collapse(ax=axes[0, 1], slope=slope, harmonics=harmonics, pairs=pairs,
                              ylim=collapse_ylim)
        axes[0, 1].set_title(r"(2) Rescaled collapse $\to g^{(2)}_0 - 1 = \sigma^2$", fontsize=15)
        self.plot_intensity_scaling(ax=axes[1, 0], n_fit=n_fit, split=split,
                                    per_channel=per_channel, harmonics=harmonics, channels=channels)
        axes[1, 0].set_title(r"(3) Intensity scaling $\langle I_n\rangle \sim I_0^{K(n)}$", fontsize=15)
        self.plot_local_slope(ax=axes[1, 1])
        axes[1, 1].set_title(r"(4) Local exponent $K(n)$  (dotted = ideal $n$)", fontsize=15)
        fig.suptitle("Power scan — harmonic fluctuation model", fontsize=19, y=0.96)
        fig.text(0.5, 0.93, self.header, ha='center', fontsize=11, color='#555555')
        fig.subplots_adjust(top=0.91, hspace=0.20, wspace=0.20,
                            left=0.07, right=0.97, bottom=0.06)
        plt.show()
        return fig, axes

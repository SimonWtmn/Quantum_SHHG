"""
=============================================================================
HBT Power-Scan Analyzer
=============================================================================

Tools to study how the harmonics of a high-harmonic-generation source behave
when the driving (pump) intensity ``I_0`` is swept. One full acquisition file per
power; this module turns that family of runs into the diagnostic plots used to
test the intensity-fluctuation model of the harmonic photon statistics:

* ``g^(2)(0)`` vs power,
* the rescaled *collapse* ``(g^(2)_n - 1)/K^2(n) -> g^(2)_0 - 1 = sigma^2``,
* the intensity scaling ``<I_n> ~ I_0^{K(n)}`` (log-log), and
* the effective nonlinearity ``K(n) = d ln<I_n>/d ln I_0``.

The exponent ``K(n)`` is central to the whole model. By default it is the local
log-log slope of the few power-scan points, but a *dense intensity scan* (many
extra count-rate-only points taken by sweeping a half-wave-plate angle, see
:class:`~src.measurement.CountrateRecorder`) can be supplied to compute a far
smoother ``K(n)`` while the ``g^(2)`` plots still use only the selected powers.

All plotting methods are **pure figure builders**: they create (or fill) the axes
and return ``(fig, ax)`` without calling ``plt.show()`` so an orchestrator
(:class:`src.report.PowerScanReport`) controls saving and interactive display.

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquee (LOA), Ecole Polytechnique
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "text.usetex": False,
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
_SCAN_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b']
_IDEAL_GREY = '0.45'

_INTERACTIVE_DPI = 110

# Detector-pair grid geometry (mirrors src.visu): the cross rows are the four
# arm combinations, with the auto-correlations sitting on their own header row.
_CROSS_ROWS = ('TT', 'TR', 'RT', 'RR')


def malus_power(angle_deg, p_max, theta0=0.0, offset=0.0, factor=2.0, branch=None):
    """Pump power transmitted by a rotating wave-plate / polariser at ``angle_deg``.

    ``P(theta) = offset + p_max * cos^2(factor * (theta - theta0))`` (Malus' law).

    ``factor`` encodes the optic: use ``2`` for a half-wave plate in front of a fixed
    polariser (a 45 deg plate rotation gives a full min->max swing) and ``1`` for a
    rotating polariser / analyser. ``angle_deg`` may be a scalar or an array. ``branch``
    is accepted (and ignored) so the same calibration dict can be splatted into both
    :func:`malus_power` and :func:`malus_angle`.
    """
    a = np.deg2rad(factor * (np.asarray(angle_deg, dtype=float) - theta0))
    return offset + p_max * np.cos(a) ** 2


def malus_angle(power, p_max, theta0=0.0, offset=0.0, factor=2.0, branch="below"):
    """Inverse of :func:`malus_power`: the stage angle (deg) that transmits ``power``.

    Because ``cos^2`` is symmetric about ``theta0`` (the angle of maximum power), each
    power maps to two angles, ``theta0 +/- delta``. ``branch`` selects which one:

    * ``"below"`` (default) -> ``theta0 - delta`` (power increases with angle), the usual
      rising side most calibrations sit on;
    * ``"above"`` -> ``theta0 + delta`` (power decreases with angle).

    The requested power is clamped to the achievable ``[offset, offset + p_max]`` range.
    """
    power = np.asarray(power, dtype=float)
    frac = np.clip((power - offset) / p_max, 0.0, 1.0)
    delta = np.rad2deg(np.arccos(np.sqrt(frac))) / factor   # >= 0
    return theta0 - delta if branch == "below" else theta0 + delta


class PowerScanAnalyzer:
    """Analyse a power scan: one :class:`~src.core.HBTMeasurement` per pump power,
    same filter / polarisation / sample.

    Parameters
    ----------
    runs : list[HBTMeasurement]
        The full runs of the scan (any order; runs without a known power are dropped).
    harmonics : tuple[int]
        Harmonic orders to analyse (default ``(3, 4, 5)``).
    tau_in_ns : float
        Integration window for the delay g^(2) (the recommended method).
    g2_method : {'delay', 'direct', 'heralded'}
        How g^(2)(0) is computed at each power.
    g2_source : {'auto', 'physical', 'virtual'}
        Histogram source for the delay method.
    intensity : {'countrate', 'counts'}
        Observable used as the harmonic intensity <I_n> (default count rate, counts/s).
    intensity_runs : list[HBTMeasurement], optional
        A DENSE set of intensity-only runs (e.g. from
        :class:`~src.measurement.CountrateRecorder`) used solely to compute a smooth
        ``K(n)``. Their power is taken from ``power_mw`` if set, otherwise from the
        rotation-stage angle via ``malus``.
    malus : dict, optional
        ``{'p_max': float, 'theta0': float, 'offset': float}`` mapping a dense run's
        stage angle to pump power (see :func:`malus_power`). Required for dense runs
        that only carry an angle.
    k_poly_deg : int
        Degree of the log-log polynomial fitted to the dense intensity curve before
        differentiating to obtain ``K(n)`` (default 3).
    """

    def __init__(self, runs, harmonics=(3, 4, 5), tau_in_ns=4.0,
                 g2_method='delay', g2_source='auto', intensity='countrate',
                 intensity_runs=None, malus=None, k_poly_deg=3):
        self.runs = sorted([r for r in runs if r.power_mw is not None],
                           key=lambda r: r.power_mw)
        if len(self.runs) < 2:
            raise ValueError("A power scan needs at least two runs with a known power.")
        self.harmonics = tuple(harmonics)
        self.tau_in_ns = tau_in_ns
        self.g2_method = g2_method
        self.g2_source = g2_source
        self.intensity = intensity
        self.malus = malus
        self.k_poly_deg = int(k_poly_deg)
        self.intensity_runs = [r for r in (intensity_runs or [])
                               if self._power_of(r) is not None]
        self.has_dense = len(self.intensity_runs) >= max(2, k_poly_deg)
        self.hcolor = {n: _AUTO_COLORS[i % len(_AUTO_COLORS)]
                       for i, n in enumerate(self.harmonics)}

        self._build()

    # ---------------- power / dense helpers ----------------

    def _power_of(self, run):
        """Pump power of a run: its ``power_mw`` if known, else Malus' law on the
        rotation-stage angle (when a ``malus`` mapping was provided)."""
        if run.power_mw is not None:
            return float(run.power_mw)
        if self.malus is not None and run.rotation_stage is not None:
            return float(malus_power(run.rotation_stage,
                                     self.malus.get('p_max', 1.0),
                                     self.malus.get('theta0', 0.0),
                                     self.malus.get('offset', 0.0),
                                     self.malus.get('factor', 2.0)))
        return None

    @staticmethod
    def _logfit_deriv(x, y, deg):
        """Derivative coefficients of a degree-``deg`` polynomial fitted to (x, y),
        plus the sorted, finite x used. Returns (None, None) if too few points."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        order = np.argsort(x)
        x, y = x[order], y[order]
        # collapse duplicate x (same power sampled twice) by averaging y
        if len(x) > 1:
            ux, inv = np.unique(x, return_inverse=True)
            if len(ux) != len(x):
                uy = np.array([y[inv == i].mean() for i in range(len(ux))])
                x, y = ux, uy
        d = min(int(deg), len(x) - 1)
        if d < 1:
            return None, x
        coef = np.polyfit(x, y, d)
        return np.polyder(coef), x

    def local_exponent_at(self, n, powers):
        """K(n) = d ln<I_n>/d ln I_0 evaluated at the given ``powers``.

        Uses the dense intensity scan (smooth poly slope) when available, otherwise
        the local gradient of the discrete power-scan points.
        """
        powers = np.atleast_1d(np.asarray(powers, float))
        if self.has_dense:
            dcoef, _ = self._logfit_deriv(np.log(self.P_dense),
                                          np.log(self.In_dense[n]), self.k_poly_deg)
            if dcoef is not None:
                return np.polyval(dcoef, np.log(powers))
        # discrete fallback: only valid at the scan powers themselves
        with np.errstate(divide='ignore', invalid='ignore'):
            k = np.gradient(np.log(self.In[n]), np.log(self.I0))
        return np.interp(np.log(powers), np.log(self.I0), k)

    def exponent_curve(self, n, num=200):
        """A smooth (P, K(n)) curve over the dense power range (dense scan only)."""
        if not self.has_dense:
            return self.I0, self.K[n]
        lo, hi = np.log(self.P_dense.min()), np.log(self.P_dense.max())
        P = np.exp(np.linspace(lo, hi, num))
        return P, self.local_exponent_at(n, P)

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

        # Dense intensity scan (for K(n)) — assembled first so the exponents can use it.
        if self.has_dense:
            pts = sorted(((self._power_of(r), r) for r in self.intensity_runs),
                         key=lambda t: t[0])
            self.P_dense = np.array([p for p, _ in pts], dtype=float)
            self.In_dense = {n: np.array([r.harmonic_intensity(n, self.intensity)
                                          for _, r in pts], dtype=float)
                             for n in self.harmonics}
            self.In_ch_dense = {}
            for n in self.harmonics:
                for arm in ('T', 'R'):
                    name = f"H{n}{arm}"
                    try:
                        self.In_ch_dense[name] = np.array(
                            [r.channel_intensity(r.get_ch(name), self.intensity)
                             for _, r in pts], dtype=float)
                    except KeyError:
                        pass
        else:
            self.P_dense, self.In_dense, self.In_ch_dense = None, {}, {}

        self.In, self.K, self.g2_auto = {}, {}, {}
        for n in self.harmonics:
            self.In[n] = np.array([r.harmonic_intensity(n, self.intensity)
                                   for r in self.runs], dtype=float)
            self.g2_auto[n] = np.array(
                [self._g2_pair(r, f"H{n}T", f"H{n}R") for r in self.runs], dtype=float)
        for n in self.harmonics:
            self.K[n] = self.local_exponent_at(n, self.I0)

        # Per-detector (physical channel) intensities, so the scaling can be shown
        # arm-by-arm (H3T, H3R, H4T, ...) and not only as the merged harmonic Hn.
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
        if self.has_dense:
            bits.append(rf"$K(n)$: {len(self.P_dense)} dense pts")
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
        (saturation) segment [b:] so that the combined two-line residual is minimal."""
        best_b, best_sse = None, np.inf
        for b in range(min_pts, len(x) - min_pts + 1):
            sse = self._line_sse(x[:b], y[:b]) + self._line_sse(x[b:], y[b:])
            if sse < best_sse:
                best_sse, best_b = sse, b
        return best_b

    def two_segment_fit(self, P, y, n_fit=None, split=None, min_pts=2):
        """Two free-slope power-law fits of y vs P in log-log: a low-power *perturbative*
        branch and a high-power *saturation* branch (each slope K is fitted, not imposed)."""
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
        """Indirect estimate of the pump excess g2_0 - 1 = sigma^2, as the mean (and std)
        over the collapsed curves vs power."""
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
            fig, ax = plt.subplots(figsize=figsize, dpi=_INTERACTIVE_DPI)
            return fig, ax, True
        return ax.figure, ax, False

    def _finish(self, fig, ax, own, title, handles=None, labels=None, legend_kw=None):
        """Title block + legend for a standalone figure (own=True).

        Pure builder: NEVER calls ``plt.show()`` — the orchestrator decides when to
        display/save. The descriptive title sits on top (suptitle); the acquisition
        metadata is the smaller grey sub-title just above the axes.
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
        return fig, ax

    def plot_intensity_scaling(self, ax=None, n_fit=None, split=None, per_channel=True,
                               harmonics=None, channels=None, show_ideal=True,
                               xlim=None, ylim=None):
        """Intensity vs I_0 (log-log) with fitted power laws per curve.

        With a dense intensity scan the dense points and a smooth poly fit are drawn
        (markers = the selected power-scan points), and the curve is annotated with
        ``K`` from the lowest to the highest power. Without it, the two-segment
        (perturbative -> saturation) fit on the scan points is used.
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
            dense_of = lambda name: self.In_ch_dense.get(name)
        else:
            series = [(rf"H{n}", n, self.hcolor[n], self.In[n])
                      for n in self.harmonics if n in keep_h]
            label_of = lambda name, n: rf"$H_{{{n}}}$"
            dense_of = lambda name: self.In_dense.get(int(name[1:]))

        any_sat = False
        for name, n, c, y in series:
            yd = dense_of(name) if self.has_dense else None
            if self.has_dense and yd is not None and np.isfinite(yd).sum() >= 2:
                # dense scatter (light) + selected power-scan points (open rings)
                ax.loglog(self.P_dense, yd, '.', color=c, ms=4, alpha=0.45)
                ax.loglog(self.I0, y, 'o', color=c, ms=7, mfc='none', mew=1.8)
                # smooth log-log poly fit and its slope at the range ends
                lp, ly = np.log(self.P_dense), np.log(yd)
                deg = max(1, min(self.k_poly_deg, len(self.P_dense) - 1))
                pc = np.polyfit(lp, ly, deg)
                xs = np.linspace(lp.min(), lp.max(), 200)
                ax.loglog(np.exp(xs), np.exp(np.polyval(pc, xs)), '-', color=c, lw=2.0)
                dpc = np.polyder(pc)
                k_lo = float(np.polyval(dpc, lp.min()))
                k_hi = float(np.polyval(dpc, lp.max()))
                lab = label_of(name, n) + rf": $K={k_lo:.2f}\!\to\!{k_hi:.2f}$ (ideal {n})"
                if show_ideal:
                    i_min = int(np.argmin(self.P_dense))
                    xf = np.array([self.P_dense.min(), self.P_dense.max()])
                    ax.loglog(xf, yd[i_min] * (xf / self.P_dense[i_min]) ** n,
                              ':', color=_IDEAL_GREY, lw=1.0)
                ax.plot([], [], '-', color=c, label=lab)
                continue

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
        return self._finish(fig, ax, own,
                            r"Harmonic intensity scaling $\langle I_n\rangle \sim I_0^{K(n)}$",
                            h, lab, legend_kw=dict(loc='lower right', ncol=ncol))

    # ---------------- per-channel intensity scaling (fits) ----------------

    @staticmethod
    def _perturbative_fixed_fit(P, y, n, frac_low=0.6, min_pts=3, r2_min=0.5):
        """Fit ``y = background + A * P^n`` with the exponent FIXED to the harmonic
        order ``n`` over the lowest ``frac_low`` fraction of points (the perturbative
        regime). The background is found by a 1-D search (log-space slope-``n`` fit
        for each trial background); ``A`` is its amplitude. Returns a dict with
        ``ok=False`` when the data does not follow ``P^n`` well enough."""
        P = np.asarray(P, float); y = np.asarray(y, float)
        good = np.isfinite(P) & np.isfinite(y) & (y > 0)
        P, y = P[good], y[good]
        order = np.argsort(P); P, y = P[order], y[order]
        k = min(len(P), max(min_pts, int(np.ceil(len(P) * frac_low))))
        if k < min_pts:
            return dict(ok=False)
        Pl, yl = P[:k], y[:k]
        lx = np.log(Pl)
        bg_grid = np.concatenate(([0.0], np.linspace(0.0, yl.min() * 0.999, 40)))
        best = None
        for bg in bg_grid:
            z = yl - bg
            if np.any(z <= 0):
                continue
            lz = np.log(z)
            b = float(np.mean(lz - n * lx))           # slope fixed to n, intercept free
            resid = lz - (n * lx + b)
            sse = float(np.sum(resid ** 2))
            sst = float(np.sum((lz - lz.mean()) ** 2))
            r2 = 1 - sse / sst if sst > 0 else np.nan
            if best is None or sse < best[0]:
                best = (sse, bg, np.exp(b), r2)
        if best is None:
            return dict(ok=False)
        _, bg, A, r2 = best
        return dict(ok=(np.isfinite(r2) and r2 >= r2_min and A > 0),
                    bg=float(bg), A=float(A), n=n, r2=float(r2), npts=k)

    @staticmethod
    def _free_high_fit(P, y, n_high=5, min_pts=2):
        """Free-exponent log-log fit ``y = A * P^k`` over the last ``n_high`` points
        (the high-power / non-perturbative end)."""
        P = np.asarray(P, float); y = np.asarray(y, float)
        good = np.isfinite(P) & np.isfinite(y) & (y > 0)
        P, y = P[good], y[good]
        order = np.argsort(P); P, y = P[order], y[order]
        m = min(len(P), max(min_pts, int(n_high)))
        if m < min_pts:
            return dict(ok=False)
        Ph, yh = P[-m:], y[-m:]
        lx, ly = np.log(Ph), np.log(yh)
        slope, intercept = np.polyfit(lx, ly, 1)
        resid = ly - (slope * lx + intercept)
        sst = float(np.sum((ly - ly.mean()) ** 2))
        r2 = 1 - float(np.sum(resid ** 2)) / sst if sst > 0 else np.nan
        return dict(ok=np.isfinite(slope), A=float(np.exp(intercept)),
                    k=float(slope), r2=float(r2), npts=m)

    @staticmethod
    def _subsample(P, y, dense_stride=1, max_points=None):
        """Thin a (P, y) curve for display. ``dense_stride`` keeps every Nth point
        (after sorting by P); ``max_points`` further caps the count by an even
        subsample. The first and last points are always kept so the fit range and
        the high-power end stay anchored."""
        P = np.asarray(P, float); y = np.asarray(y, float)
        order = np.argsort(P)
        P, y = P[order], y[order]
        idx = np.arange(len(P))
        if dense_stride and dense_stride > 1:
            idx = idx[::int(dense_stride)]
        if max_points is not None and len(idx) > max_points > 0:
            idx = idx[np.linspace(0, len(idx) - 1, int(max_points)).round().astype(int)]
        if len(P):
            idx = np.unique(np.concatenate(([0], idx, [len(P) - 1])))
        return P[idx], y[idx]

    def plot_intensity_grid(self, channels=None, harmonics=None, frac_low=0.6,
                            n_high=5, r2_min=0.5, dense_stride=1, max_points=None,
                            xlim=None, ylim=None, figsize=(17, 10)):
        """Per-channel harmonic-intensity scaling (one panel per detector).

        Each panel shows the measured points (the dense scan when available, else the
        power-scan points), the **perturbative fixed-order** fit ``y = bg + A\\,P^n``
        (exponent fixed to the harmonic order ``n``, the ideal ``P^n`` law) and a
        **free-order** fit ``y = A\\,P^k`` of the last ``n_high`` (high-power) points.

        ``dense_stride``/``max_points`` control HOW MANY dense points are drawn
        (``dense_stride=1`` -> every point; ``dense_stride=2`` -> every other;
        ``max_points=15`` -> at most 15, evenly spaced). The fits always use the full
        data, so thinning the markers never changes the result. Returns ``(fig, axes)``."""
        keep_h, _ = self._resolve_keep(harmonics, None)
        hs = [n for n in self.harmonics if n in keep_h]
        arms = ('T', 'R')
        keep_c = set(channels) if channels is not None else None
        fig, axes = plt.subplots(len(arms), max(len(hs), 1), figsize=figsize,
                                 dpi=_INTERACTIVE_DPI, squeeze=False)
        for ai, arm in enumerate(arms):
            for ci, n in enumerate(hs):
                ax = axes[ai, ci]
                name = f"H{n}{arm}"
                if (keep_c is not None and name not in keep_c) or name not in self.In_ch:
                    ax.set_visible(False)
                    continue
                if self.has_dense and name in self.In_ch_dense:
                    P, y = self.P_dense, self.In_ch_dense[name]
                else:
                    P, y = self.I0, self.In_ch[name]
                P = np.asarray(P, float); y = np.asarray(y, float)
                Pp, yp = self._subsample(P, y, dense_stride, max_points)
                try:
                    ch = self.runs[0].get_ch(name)
                except KeyError:
                    ch = "?"
                ax.loglog(Pp, yp, 'o', color='#3b76af', ms=4, alpha=0.7,
                          label=f'Measured data ({len(Pp)} pts)')

                good = np.isfinite(P) & np.isfinite(y) & (y > 0)
                xs = (np.logspace(np.log10(P[good].min()), np.log10(P[good].max()), 200)
                      if good.sum() >= 2 else None)
                lo = self._perturbative_fixed_fit(P, y, n, frac_low=frac_low, r2_min=r2_min)
                hi = self._free_high_fit(P, y, n_high=n_high)

                txt = []
                if lo.get('ok') and xs is not None:
                    ax.loglog(xs, lo['bg'] + lo['A'] * xs ** n, '-', color='#ff7f0e', lw=2.0,
                              label=rf"Perturbative: fixed $n={n}$, $R^2$={lo['r2']:.3f}")
                    txt.append(rf"background = {lo['bg']:.3g} Hz")
                    txt.append(rf"low: $A$={lo['A']:.3g} $P^{{{n}}}$")
                else:
                    txt.append("low fit unavailable")
                if hi.get('ok') and xs is not None:
                    ax.loglog(xs, hi['A'] * xs ** hi['k'], '--', color='#2ca02c', lw=2.0,
                              label=rf"High power: free $n={hi['k']:.3f}$, $R^2$={hi['r2']:.3f}")
                    txt.append(rf"high: $A$={hi['A']:.3g} $P^{{{hi['k']:.3f}}}$")

                ax.set_title(rf"{name} – channel {ch}: perturbative $P^{{{n}}}$", fontsize=12)
                ax.grid(True, which='both', alpha=0.25)
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va='top',
                        ha='left', fontsize=8.5,
                        bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
                ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
                if ai == len(arms) - 1:
                    ax.set_xlabel("Pump power (mW)")
                ax.set_ylabel(f"{name} count rate (Hz)")
            for ci in range(len(hs), axes.shape[1]):
                axes[ai, ci].set_visible(False)
        fig.suptitle("Harmonic scaling: perturbative fixed-order and free-order fits",
                     fontsize=16, y=0.995)
        fig.text(0.5, 0.965, self.header, ha='center', fontsize=9.5, color='#555555')
        fig.subplots_adjust(top=0.91, hspace=0.28, wspace=0.24,
                            left=0.06, right=0.98, bottom=0.08)
        return fig, axes

    def plot_local_slope(self, ax=None, xlim=None, ylim=None):
        """Local exponent K(n) = d ln<I_n>/d ln I_0 vs power; departs from n at saturation.

        With a dense intensity scan the smooth curve is drawn and the values sampled at
        the actual power-scan points are marked."""
        fig, ax, own = self._ax(ax, figsize=(9, 6))
        for n in self.harmonics:
            c = self.hcolor[n]
            if self.has_dense:
                Pc, Kc = self.exponent_curve(n)
                ax.semilogx(Pc, Kc, '-', color=c, lw=2.0, label=rf"$K({n})$ (dense)")
                ax.semilogx(self.I0, self.K[n], 'o', color=c, ms=7, mfc='white', mew=1.8)
            else:
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
        keep_h = set(self.harmonics if harmonics is None else harmonics)
        keep_p = {tuple(sorted(p)) for p in pairs} if pairs is not None else None
        return keep_h, keep_p

    @staticmethod
    def _keep_pair(m, n, keep_h, keep_p):
        if keep_p is not None:
            return tuple(sorted((m, n))) in keep_p
        return m in keep_h and n in keep_h

    def plot_g2_vs_power(self, ax=None, include_cross=False, harmonics=None, pairs=None,
                         xlim=None, ylim=None):
        """g^(2)(0) vs I_0 — a family of distinct curves (one per harmonic / pair).

        ``ylim`` defaults to ``None`` (auto-scale), so a scan whose g^(2) spans a wide
        range (e.g. 1.3 to 9 across powers) is shown fully instead of being clipped."""
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

    def R_cross(self, m, n):
        """Cauchy-Schwarz ratio ``R = g2_mn^2 / (g2_mm * g2_nn)`` vs power for the
        cross pair (m, n). ``R <= 1`` is the classical Cauchy-Schwarz bound; ``R > 1``
        flags a non-classical / strongly correlated harmonic pair."""
        gc = np.asarray(self.g2_cross[(m, n)], dtype=float)
        ga, gb = np.asarray(self.g2_auto[m], float), np.asarray(self.g2_auto[n], float)
        with np.errstate(divide='ignore', invalid='ignore'):
            R = gc ** 2 / (ga * gb)
        R[~np.isfinite(R)] = np.nan
        return R

    def plot_R_vs_power(self, ax=None, harmonics=None, pairs=None, xlim=None, ylim=None):
        """Cauchy-Schwarz ``R`` vs pump power, one curve per cross pair (m, n).

        Companion to :meth:`plot_g2_vs_power`: where ``g^(2)`` shows the bunching of
        each harmonic, ``R`` shows how strongly two harmonics are correlated relative
        to the classical bound ``R=1``."""
        fig, ax, own = self._ax(ax)
        keep_h, keep_p = self._resolve_keep(harmonics, pairs)
        for (m, n) in self.g2_cross:
            if not self._keep_pair(m, n, keep_h, keep_p):
                continue
            ax.plot(self.I0, self.R_cross(m, n), 's-', color=self.ccolor[(m, n)],
                    ms=7, lw=1.8, label=rf"$R_{{{m}{n}}}$")
        ax.axhline(1.0, color='#313131', ls='--', lw=1.3, alpha=0.7)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$R = g^{(2)\,2}_{mn} / (g^{(2)}_{mm}\, g^{(2)}_{nn})$")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        h, lab = ax.get_legend_handles_labels()
        h.append(Line2D([0], [0], color='#313131', ls='--', lw=1.3))
        lab.append(r"classical bound ($R=1$)")
        return self._finish(fig, ax, own, r"Cauchy-Schwarz $R$ vs pump power",
                            h, lab, legend_kw=dict(ncol=2, loc='best'))

    # ---------------- per-detector-pair grids vs power ----------------

    def _pair_g2_vs_power(self, a_name, b_name):
        """g^(2) vs power for an explicit detector pair (e.g. 'H3T','H4R').

        Returns ``None`` if either detector is absent from the runs. Cached so the
        grids and R-grid can share the same per-pair computation.
        """
        key = (a_name, b_name)
        cache = getattr(self, "_pair_cache", None)
        if cache is None:
            cache = self._pair_cache = {}
        if key in cache:
            return cache[key]
        vals = []
        for r in self.runs:
            try:
                r.get_ch(a_name); r.get_ch(b_name)
            except KeyError:
                cache[key] = None
                return None
            vals.append(self._g2_pair(r, a_name, b_name))
        arr = np.array(vals, dtype=float)
        cache[key] = arr
        return arr

    def _pair_R_vs_power(self, cross, autoA, autoB):
        """Cauchy-Schwarz R vs power for an explicit (cross, autoA, autoB) triplet."""
        gc = self._pair_g2_vs_power(*cross)
        ga = self._pair_g2_vs_power(*autoA)
        gb = self._pair_g2_vs_power(*autoB)
        if gc is None or ga is None or gb is None:
            return None
        with np.errstate(divide='ignore', invalid='ignore'):
            R = gc ** 2 / (ga * gb)
        R[~np.isfinite(R)] = np.nan
        return R

    @staticmethod
    def _grid_ax_decor(ax, ylim, ref=1.0):
        ax.axhline(ref, color='#313131', ls='--', lw=1.2, alpha=0.6)
        ax.grid(True, alpha=0.25)
        if ylim is not None:
            ax.set_ylim(ylim)

    def plot_g2_grid_vs_power(self, ylim=None, figsize=(16, 18)):
        """Grid of g^(2)(0) **vs pump power**, one panel per detector pair.

        Row 0 holds the auto-correlations ``H_{nn}`` (R&T); the next four rows are
        the cross pairs in each arm combination (TT, TR, RT, RR). Companion to the
        integration-window grid, but here the x-axis is the pump power. Returns
        ``(fig, axes)``."""
        hs = list(self.harmonics)
        pairs = [(hs[i], hs[j]) for i in range(len(hs)) for j in range(i + 1, len(hs))]
        ncol = max(len(hs), len(pairs), 1)
        fig, axes = plt.subplots(1 + len(_CROSS_ROWS), ncol,
                                 figsize=figsize, dpi=_INTERACTIVE_DPI, squeeze=False)
        for j in range(ncol):
            # header row: autocorrelations
            ax = axes[0, j]
            if j < len(hs):
                n = hs[j]
                y = self._pair_g2_vs_power(f"H{n}R", f"H{n}T")
                if y is not None:
                    ax.plot(self.I0, y, 'o-', color=self.hcolor[n], ms=6)
                    self._grid_ax_decor(ax, ylim)
                    ax.set_title(rf"auto $g^{{(2)}}_{{{n}{n}}}$ (RT)", fontsize=12)
                else:
                    ax.set_visible(False)
            else:
                ax.set_visible(False)
            # cross rows
            for ri, row in enumerate(_CROSS_ROWS, start=1):
                ax = axes[ri, j]
                if j < len(pairs):
                    m, n = pairs[j]
                    y = self._pair_g2_vs_power(f"H{m}{row[0]}", f"H{n}{row[1]}")
                    if y is not None:
                        ax.plot(self.I0, y, 's-', color=self.ccolor.get((m, n), '#1f77b4'),
                                ms=6, lw=1.6)
                        self._grid_ax_decor(ax, ylim)
                        ax.set_title(rf"cross $g^{{(2)}}_{{{m}{n}}}$ ({row})", fontsize=12)
                    else:
                        ax.set_visible(False)
                else:
                    ax.set_visible(False)
        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel(r"Pump power $P$ (mW)")
        for i in range(axes.shape[0]):
            if axes[i, 0].get_visible():
                axes[i, 0].set_ylabel(r"$g^{(2)}(0)$")
        fig.suptitle(r"$g^{(2)}(0)$ vs pump power — every detector pair", fontsize=18, y=0.998)
        fig.text(0.5, 0.965, self.header, ha='center', fontsize=10, color='#555555')
        fig.subplots_adjust(top=0.93, hspace=0.34, wspace=0.22,
                            left=0.06, right=0.98, bottom=0.05)
        return fig, axes

    def plot_R_grid_vs_power(self, ylim=None, figsize=(16, 15)):
        """Grid of Cauchy-Schwarz ``R`` **vs pump power**, one panel per cross pair
        and arm combination (TT, TR, RT, RR). Returns ``(fig, axes)``."""
        hs = list(self.harmonics)
        pairs = [(hs[i], hs[j]) for i in range(len(hs)) for j in range(i + 1, len(hs))]
        ncol = max(len(pairs), 1)
        fig, axes = plt.subplots(len(_CROSS_ROWS), ncol,
                                 figsize=figsize, dpi=_INTERACTIVE_DPI, squeeze=False)
        for ri, row in enumerate(_CROSS_ROWS):
            for j in range(ncol):
                ax = axes[ri, j]
                if j >= len(pairs):
                    ax.set_visible(False)
                    continue
                m, n = pairs[j]
                R = self._pair_R_vs_power(
                    (f"H{m}{row[0]}", f"H{n}{row[1]}"),
                    (f"H{m}R", f"H{m}T"), (f"H{n}R", f"H{n}T"))
                if R is None:
                    ax.set_visible(False)
                    continue
                ax.plot(self.I0, R, 's-', color=self.ccolor.get((m, n), '#1f77b4'),
                        ms=6, lw=1.6)
                self._grid_ax_decor(ax, ylim)
                ax.set_title(rf"$R_{{{m}{n}}}$ ({row})", fontsize=12)
                if ri == len(_CROSS_ROWS) - 1:
                    ax.set_xlabel(r"Pump power $P$ (mW)")
                if j == 0:
                    ax.set_ylabel(r"$R$")
        fig.suptitle(r"Cauchy-Schwarz $R$ vs pump power — every detector pair",
                     fontsize=18, y=0.998)
        fig.text(0.5, 0.96, self.header, ha='center', fontsize=10, color='#555555')
        fig.subplots_adjust(top=0.92, hspace=0.34, wspace=0.22,
                            left=0.06, right=0.98, bottom=0.06)
        return fig, axes

    def plot_g2_collapse(self, ax=None, slope='local', include_cross=True, show_mean=True,
                         harmonics=None, pairs=None, xlim=None, ylim=(0, 0.015)):
        """(g^(2)_n - 1)/K^2(n) vs I_0 — all harmonics should COLLAPSE onto g2_0 - 1."""
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
                      g2_ylim=None, collapse_ylim=(0, 0.015)):
        """2x2 dashboard mirroring the model: g^(2) vs power, the rescaled collapse, the
        intensity scaling (log-log) and the local exponent K(n). Returns (fig, axes)."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=_INTERACTIVE_DPI)
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
        return fig, axes


# =============================================================================
# Comparison of several power scans (e.g. P1 vs no-P1, filter A vs filter B)
# =============================================================================

class PowerScanComparison:
    """Overlay the model curves of several power scans to read off the effect of a
    configuration change (polariser, filter, ...).

    Parameters
    ----------
    scans : dict[str, PowerScanAnalyzer]
        ``{label: analyzer}``; the label names the curve in every legend.
    harmonics : tuple[int], optional
        Restrict to these orders (default: the intersection across scans).
    """

    def __init__(self, scans, harmonics=None):
        if len(scans) < 2:
            raise ValueError("Need at least two power scans to compare.")
        self.scans = dict(scans)
        self.labels = list(self.scans)
        common = set.intersection(*[set(a.harmonics) for a in self.scans.values()])
        self.harmonics = tuple(sorted(harmonics if harmonics is not None else common))
        self.color = {lab: _SCAN_COLORS[i % len(_SCAN_COLORS)]
                      for i, lab in enumerate(self.labels)}

    @staticmethod
    def _ax(ax, figsize=(9, 6.5)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=_INTERACTIVE_DPI)
            return fig, ax, True
        return ax.figure, ax, False

    @staticmethod
    def _finish(fig, ax, own, title, subtitle=None, legend_kw=None):
        lk = dict(framealpha=0.92, edgecolor='0.7', fontsize=10)
        if legend_kw:
            lk.update(legend_kw)
        ax.legend(**lk)
        if own:
            fig.suptitle(title, fontsize=16, y=0.95)
            if subtitle:
                ax.set_title(subtitle, fontsize=9.5, color='#555555')
            fig.tight_layout()
        return fig, ax

    def _hmark(self, i):
        return ['o', 's', '^', 'D', 'v', 'P'][i % 6]

    def plot_g2_vs_power(self, ax=None, harmonics=None, xlim=None, ylim=(0.9, 1.6)):
        """Overlay g^(2)_nn(P) for every scan (colour = scan, marker = harmonic)."""
        fig, ax, own = self._ax(ax)
        hs = harmonics or self.harmonics
        for lab, a in self.scans.items():
            for i, n in enumerate(hs):
                if n not in a.harmonics:
                    continue
                ax.plot(a.I0, a.g2_auto[n], marker=self._hmark(i), ls='-',
                        color=self.color[lab], ms=6,
                        label=rf"{lab} — $H_{{{n}}}$")
        ax.axhline(1.0, color='#313131', ls='--', lw=1.2, alpha=0.7)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$g^{(2)}(0)$")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return self._finish(fig, ax, own, r"$g^{(2)}(0)$ vs pump power — comparison",
                            legend_kw=dict(ncol=len(self.scans), loc='best'))

    def plot_local_slope(self, ax=None, harmonics=None, xlim=None, ylim=None):
        """Overlay K(n)(P) for every scan."""
        fig, ax, own = self._ax(ax, figsize=(9, 6))
        hs = harmonics or self.harmonics
        for lab, a in self.scans.items():
            for i, n in enumerate(hs):
                if n not in a.harmonics:
                    continue
                Pc, Kc = a.exponent_curve(n)
                ax.semilogx(Pc, Kc, ls='-', color=self.color[lab],
                            marker=(None if a.has_dense else self._hmark(i)),
                            ms=6, label=rf"{lab} — $K({n})$")
        for i, n in enumerate(hs):
            ax.axhline(n, color='0.6', ls=':', lw=1.1, alpha=0.6)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$K(n) = \mathrm{d}\ln\langle I_n\rangle/\mathrm{d}\ln I_0$")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return self._finish(fig, ax, own, r"Effective nonlinearity $K(n)$ — comparison",
                            legend_kw=dict(ncol=len(self.scans), loc='best'))

    def plot_intensity_scaling(self, ax=None, harmonics=None, xlim=None, ylim=None):
        """Overlay the merged-harmonic intensity scaling <I_n>(P) for every scan."""
        fig, ax, own = self._ax(ax)
        hs = harmonics or self.harmonics
        for lab, a in self.scans.items():
            for i, n in enumerate(hs):
                if n not in a.harmonics:
                    continue
                if a.has_dense and n in a.In_dense:
                    ax.loglog(a.P_dense, a.In_dense[n], '.', color=self.color[lab],
                              ms=4, alpha=0.4)
                ax.loglog(a.I0, a.In[n], marker=self._hmark(i), ls='-',
                          color=self.color[lab], ms=6, label=rf"{lab} — $H_{{{n}}}$")
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"Intensity (counts/s)")
        ax.grid(True, which='both', alpha=0.25)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return self._finish(fig, ax, own,
                            r"Harmonic intensity scaling $\langle I_n\rangle \sim I_0^{K(n)}$ — comparison",
                            legend_kw=dict(ncol=len(self.scans), loc='lower right'))

    def plot_inferred_sigma2(self, ax=None, slope='local', harmonics=None,
                             include_cross=True, xlim=None, ylim=None):
        """Overlay the inferred pump excess sigma^2 = g2_0 - 1 (mean over harmonics)
        for every scan, with its 1-sigma band — the headline comparison."""
        fig, ax, own = self._ax(ax)
        for lab, a in self.scans.items():
            mean, std = a.inferred_g2_0(slope, harmonics=harmonics,
                                        pairs=(None if include_cross else []))
            c = self.color[lab]
            ax.plot(a.I0, mean, 'o-', color=c, ms=6, label=rf"{lab}")
            ax.fill_between(a.I0, mean - std, mean + std, color=c, alpha=0.13)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$\sigma^2 = g^{(2)}_0 - 1$")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return self._finish(fig, ax, own,
                            r"Inferred pump excess $\sigma^2$ — comparison",
                            legend_kw=dict(loc='best'))

    def plot_overview(self, slope='local', harmonics=None,
                      g2_ylim=(0.9, 1.6), sigma2_ylim=None):
        """2x2 comparison dashboard: g^(2) vs power, inferred sigma^2, intensity
        scaling and K(n), each overlaying every scan. Returns (fig, axes)."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=_INTERACTIVE_DPI)
        self.plot_g2_vs_power(ax=axes[0, 0], harmonics=harmonics, ylim=g2_ylim)
        axes[0, 0].set_title(r"(1) $g^{(2)}(0)$ vs pump power", fontsize=15)
        self.plot_inferred_sigma2(ax=axes[0, 1], slope=slope, harmonics=harmonics, ylim=sigma2_ylim)
        axes[0, 1].set_title(r"(2) Inferred $\sigma^2 = g^{(2)}_0 - 1$", fontsize=15)
        self.plot_intensity_scaling(ax=axes[1, 0], harmonics=harmonics)
        axes[1, 0].set_title(r"(3) Intensity scaling $\langle I_n\rangle \sim I_0^{K(n)}$", fontsize=15)
        self.plot_local_slope(ax=axes[1, 1], harmonics=harmonics)
        axes[1, 1].set_title(r"(4) Local exponent $K(n)$", fontsize=15)
        fig.suptitle("Power-scan comparison", fontsize=19, y=0.96)
        fig.subplots_adjust(top=0.92, hspace=0.20, wspace=0.20,
                            left=0.07, right=0.97, bottom=0.06)
        return fig, axes

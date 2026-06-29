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
# Neutral dark used for the fixed "ideal P^n" reference line so it never clashes with
# the data-point colour (the two used to share a hue and were hard to tell apart).
_REF_DARK = '#222222'

_INTERACTIVE_DPI = 96
# On-screen width budget (px) for a figure shown inline. The interactive ipympl canvas is
# rendered at its NATIVE pixel size (width_in x dpi); if that is wider than the notebook
# output area it gets cropped (it cannot be shrunk after the fact in VS Code / Cursor). So
# we BUILD every interactive figure at a dpi that keeps its native width within this budget
# -> the whole figure fits and stays fully interactive. Saved PNGs are unaffected (savefig
# uses its own high dpi), so on-disk detail is preserved.
_FIT_WIDTH_PX = 820


def _fit_dpi(width_in, default=_INTERACTIVE_DPI):
    """A render dpi so a ``width_in``-inch figure is at most ``_FIT_WIDTH_PX`` px wide
    on screen (never upscales past ``default``)."""
    return max(18.0, min(float(default), _FIT_WIDTH_PX / max(float(width_in), 0.1)))


def _darken(color, factor=0.55):
    """Return a darker shade of ``color`` (factor in (0, 1], smaller = darker).

    Used to draw a fit/reference line in the same family as its data points while
    staying clearly distinguishable from them."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)

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
        # Remembers the y-limits a standalone plot was last drawn with, so the 2x2
        # overview can reuse them instead of falling back to its own defaults.
        self._ylim = {}

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
            # The dense scan only helps if it actually covers the power-scan range. If it
            # sits only near full power (a too-narrow angle sweep), K(n) is measured in the
            # saturation band and extrapolated down — the perturbative P^n scaling then
            # looks wrong. Flag that loudly instead of silently returning bad exponents.
            if self.P_dense.min() > self.I0.min() * 1.2:
                print(f"[PowerScanAnalyzer] WARNING: the dense K(n) scan covers only "
                      f"{self.P_dense.min():g}-{self.P_dense.max():g} mW, but the power "
                      f"scan spans {self.I0.min():g}-{self.I0.max():g} mW. The dense scan "
                      f"misses the low-power region, so K(n) is fitted in the saturation "
                      f"band and extrapolated — the intensity scaling will look wrong. "
                      f"Re-take the dense scan over the FULL scan angle range (down to the "
                      f"lowest-power angle), or omit intensity_runs to use the discrete "
                      f"power-point K(n) instead.")
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
        self._header_bits = bits
        return "  |  ".join(bits)

    def _header_text(self, max_chars=90):
        """The acquisition sub-title, greedily wrapped onto as many rows as needed so
        no line ever exceeds ``max_chars`` (and so never overflows the figure width).
        Short headers stay on one line."""
        bits = getattr(self, "_header_bits", None) or [self.header]
        sep = "  |  "
        one = sep.join(bits)
        if len(one) <= max_chars or len(bits) < 2:
            return one
        lines, cur = [], ""
        for b in bits:
            cand = b if not cur else cur + sep + b
            if cur and len(cand) > max_chars:
                lines.append(cur)
                cur = b
            else:
                cur = cand
        if cur:
            lines.append(cur)
        return "\n".join(lines)

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
    def _ax(ax, figsize=(6.5, 4.5), dpi=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi or _fit_dpi(figsize[0]))
            return fig, ax, True
        return ax.figure, ax, False

    def _finish(self, fig, ax, own, title, handles=None, labels=None, legend_kw=None):
        """Title block + legend for a standalone figure (own=True).

        Pure builder: NEVER calls ``plt.show()`` — the orchestrator decides when to
        display/save. The descriptive title sits on top (suptitle); the acquisition
        metadata is the smaller grey sub-title just above the axes (wrapped onto two
        rows when it is long).
        """
        lk = dict(framealpha=0.92, edgecolor='0.7', fontsize=10)
        if legend_kw:
            lk.update(legend_kw)
        if handles is not None:
            ax.legend(handles, labels, **lk)
        else:
            ax.legend(**lk)
        if own:
            # Wrap the grey metadata to the actual figure width so it never spills past
            # the edge, and keep a clear gap between the title, the sub-title and the axes.
            w_in = float(fig.get_size_inches()[0]) or 6.5
            max_chars = max(36, int(w_in * 11))
            header = self._header_text(max_chars=max_chars)
            header_lines = header.split("\n")
            y = 0.975
            fig.text(0.5, y, title, ha="center", va="top", fontsize=14,
                     transform=fig.transFigure)
            y -= 0.058
            for line in header_lines:
                fig.text(0.5, y, line, ha="center", va="top", fontsize=8.0,
                         color="#555555", transform=fig.transFigure)
                y -= 0.034
            top = max(y - 0.010, 0.74)
            fig.subplots_adjust(top=top, bottom=0.12, left=0.13, right=0.96)
        return fig, ax

    def plot_intensity_scaling(self, ax=None, n_fit=None, split=None, per_channel=True,
                               harmonics=None, channels=None, show_ideal=True,
                               frac_low=0.6, xlim=None, ylim=None, dpi=None):
        """Intensity vs I_0 (log-log): measured points + ideal ``P^n`` reference lines.

        No fits are drawn here — only a **solid** ``P^n`` line whose slope is FIXED to
        the harmonic order and whose offset is placed (median, robust) through the
        perturbative cloud, so you can eyeball whether the low-power points follow the
        expected order. Saturation fits live on :meth:`plot_intensity_grid` only.
        """
        fig, ax, own = self._ax(ax, dpi=dpi)
        if own:
            self._ylim['intensity'] = (xlim, ylim)
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

        for name, n, c, y in series:
            yd = dense_of(name) if self.has_dense else None
            P_plot = self.P_dense if (self.has_dense and yd is not None) else self.I0
            y_plot = yd if (self.has_dense and yd is not None) else y
            line_c = _darken(c)   # ideal line in a darker shade so it stands out from the points
            if self.has_dense and yd is not None and np.isfinite(yd).sum() >= 2:
                ax.loglog(self.P_dense, yd, '.', color=c, ms=4, alpha=0.35)
                ax.loglog(self.I0, y, 'o', color=c, ms=7, mfc=c, mec='white', mew=1.0)
            else:
                good = np.isfinite(self.I0) & np.isfinite(y) & (y > 0)
                if not good.any():
                    continue
                ax.loglog(self.I0[good], y[good], 'o', color=c, ms=7,
                          mfc=c, mec='white', mew=1.0)

            if show_ideal:
                ref = self._perturbative_ref(P_plot, y_plot, n, frac_low=frac_low)
                if ref.get('ok'):
                    xf = np.array([ref['P0'], ref['P_hi']])
                    ax.loglog(xf, ref['A'] * xf ** n, '-', color=line_c, lw=2.6,
                              solid_capstyle='round', zorder=5)
            lab = label_of(name, n) + rf" (ideal $P^{{{n}}}$)"
            ax.plot([], [], 'o-', color=line_c, mfc=c, mec='white', mew=1.0, label=lab)

        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"Intensity (counts/s)")
        ax.grid(True, which='both', alpha=0.25)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        h, lab = ax.get_legend_handles_labels()
        ncol = 2 if len(series) > 3 else 1
        return self._finish(fig, ax, own,
                            r"Harmonic intensity scaling $\langle I_n\rangle \sim I_0^{K(n)}$",
                            h, lab, legend_kw=dict(loc='lower right', ncol=ncol))

    # ---------------- per-channel intensity scaling (grid) ----------------

    @staticmethod
    def _perturbative_ref(P, y, n, frac_low=0.6, min_pts=3):
        """Placement of an ideal slope-``n`` reference line through the low-power
        (perturbative) data — **no fit of the exponent**.

        The slope is FIXED to the harmonic order ``n``; only the vertical offset is
        set so the line ``y = A * P^n`` runs through the perturbative cloud, letting
        you eyeball whether the points really follow that order. The offset is the
        median of ``ln y - n ln P`` over the band (robust to scatter, unlike
        anchoring on a single noisy lowest point).

        The low-power band is the bottom ``frac_low`` fraction of the **log-power
        range** (not a point count), so a dense scan whose points pile up near full
        power (Malus' law is flat there) still uses only the genuinely low-power
        points. Returns ``A`` plus the band edges ``(P0, P_hi)`` to draw over.
        """
        P = np.asarray(P, float); y = np.asarray(y, float)
        good = np.isfinite(P) & np.isfinite(y) & (y > 0)
        P, y = P[good], y[good]
        if len(P) < min_pts:
            return dict(ok=False)
        order = np.argsort(P); P, y = P[order], y[order]
        lx = np.log(P)
        span = float(lx.max() - lx.min())
        if span > 0:
            mask = lx <= lx.min() + float(frac_low) * span
            if int(mask.sum()) < min_pts:                  # widen to keep enough points
                mask = np.zeros(len(P), dtype=bool)
                mask[:min(len(P), min_pts)] = True
        else:
            mask = np.ones(len(P), dtype=bool)
        Pl, yl = P[mask], y[mask]
        logA = float(np.median(np.log(yl) - n * np.log(Pl)))
        return dict(ok=True, A=float(np.exp(logA)), n=n,
                    P0=float(Pl.min()), P_hi=float(Pl.max()), npts=int(mask.sum()))

    @staticmethod
    def _free_high_fit(P, y, n_high=5, frac_high=0.4, min_pts=2):
        """Free-exponent log-log fit ``y = A * P^k`` over the high-power end.

        The high-power branch is taken as the points sitting in the upper
        ``frac_high`` fraction of the *log-power* range. This matters for a dense
        scan: its points pile up near full power (Malus' law is flat there), so the
        plain "last ``n_high`` points" would span an almost-zero power range and
        return a meaningless (often steeply negative) slope. Spanning a real range
        instead recovers the gentle, still-rising saturation slope. For a sparse
        scan that leaves too few points in that band it falls back to the last
        ``n_high`` points. Returns ``P_lo`` (lowest power used) so the caller can
        draw the line only over the fitted range."""
        P = np.asarray(P, float); y = np.asarray(y, float)
        good = np.isfinite(P) & np.isfinite(y) & (y > 0)
        P, y = P[good], y[good]
        order = np.argsort(P); P, y = P[order], y[order]
        if len(P) < min_pts:
            return dict(ok=False)
        lx_all = np.log(P)
        span = float(lx_all.max() - lx_all.min())
        mask = (lx_all >= lx_all.max() - float(frac_high) * span) if span > 0 \
            else np.zeros(len(P), dtype=bool)
        if int(mask.sum()) >= max(min_pts, 2):
            Ph, yh = P[mask], y[mask]
        else:
            m = min(len(P), max(min_pts, int(n_high)))
            Ph, yh = P[-m:], y[-m:]
        lx, ly = np.log(Ph), np.log(yh)
        if len(np.unique(lx)) < 2:
            return dict(ok=False)
        slope, intercept = np.polyfit(lx, ly, 1)
        resid = ly - (slope * lx + intercept)
        sst = float(np.sum((ly - ly.mean()) ** 2))
        r2 = 1 - float(np.sum(resid ** 2)) / sst if sst > 0 else np.nan
        return dict(ok=np.isfinite(slope), A=float(np.exp(intercept)),
                    k=float(slope), r2=float(r2), npts=len(Ph),
                    P_lo=float(Ph.min()), P_hi=float(Ph.max()))

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
                            n_high=5, frac_high=0.4, dense_stride=1,
                            max_points=None, xlim=None, ylim=None, figsize=(14, 8),
                            dpi=None):
        """Per-channel harmonic-intensity scaling (one panel per detector).

        Each panel shows measured points, a **solid** ideal ``P^n`` reference line
        (slope fixed to the harmonic order, placed through the perturbative cloud —
        no exponent fit) and a **dashed** free-exponent fit of the high-power
        (saturation) band only — the only fit in the whole intensity suite.
        """
        keep_h, _ = self._resolve_keep(harmonics, None)
        hs = [n for n in self.harmonics if n in keep_h]
        arms = ('T', 'R')
        keep_c = set(channels) if channels is not None else None
        fig, axes = plt.subplots(len(arms), max(len(hs), 1), figsize=figsize,
                                 dpi=dpi or _fit_dpi(figsize[0]), squeeze=False)
        for ai, arm in enumerate(arms):
            for ci, n in enumerate(hs):
                ax = axes[ai, ci]
                name = f"H{n}{arm}"
                if (keep_c is not None and name not in keep_c) or name not in self.In_ch:
                    ax.set_visible(False)
                    continue
                c = self.chcolor[name]
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
                ax.loglog(Pp, yp, 'o', color=c, ms=4.5, alpha=0.85, mec='white',
                          mew=0.6, label=f'data ({len(Pp)} pts)')

                good = np.isfinite(P) & np.isfinite(y) & (y > 0)
                ref = self._perturbative_ref(P, y, n, frac_low=frac_low)
                hi = self._free_high_fit(P, y, n_high=n_high, frac_high=frac_high)

                # Three clearly distinct styles so the points, the fixed ideal slope and
                # the saturation fit never blur together (they used to share one colour):
                # data = channel colour points, ideal = neutral dark solid, sat fit = dashed.
                txt = []
                if ref.get('ok'):
                    xf = np.array([ref['P0'], ref['P_hi']])
                    ax.loglog(xf, ref['A'] * xf ** n, '-', color=_REF_DARK, lw=2.2,
                              solid_capstyle='round', zorder=5,
                              label=rf"perturbative $P^{{{n}}}$ (no fit)")
                    txt.append(rf"ideal slope $K={n}$")
                if hi.get('ok') and good.sum() >= 2:
                    xs_hi = np.logspace(np.log10(hi['P_lo']), np.log10(P[good].max()), 120)
                    ax.loglog(xs_hi, hi['A'] * xs_hi ** hi['k'], '--', color=_darken(c),
                              lw=2.4, alpha=0.95, zorder=6,
                              label=rf"saturation fit: $K={hi['k']:.2f}$")
                    txt.append(rf"sat fit: $K={hi['k']:.2f}$, $R^2$={hi['r2']:.3f}")

                ax.set_title(rf"{name} (ch {ch})", fontsize=12)
                ax.grid(True, which='both', alpha=0.25)
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if txt:
                    ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va='top',
                            ha='left', fontsize=8.5,
                            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
                ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
                if ai == len(arms) - 1:
                    ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
                ax.set_ylabel(f"{name} count rate (Hz)")
            for ci in range(len(hs), axes.shape[1]):
                axes[ai, ci].set_visible(False)
        header = self._header_text(max_chars=max(60, int(figsize[0] * 11)))
        nlines = header.count("\n") + 1
        y_title = 0.975
        fig.text(0.5, y_title, "Harmonic intensity scaling — per channel",
                 ha="center", va="top", fontsize=15, transform=fig.transFigure)
        y_hdr = y_title - 0.042
        for line in header.split("\n"):
            fig.text(0.5, y_hdr, line, ha="center", va="top", fontsize=8.0,
                     color="#555555", transform=fig.transFigure)
            y_hdr -= 0.026
        # leave a clear gap below the header so the top-row panel titles never collide
        top = max(y_hdr - 0.040, 0.80)
        fig.subplots_adjust(top=top, hspace=0.32, wspace=0.24,
                            left=0.07, right=0.98, bottom=0.08)
        return fig, axes

    def plot_local_slope(self, ax=None, xlim=None, ylim=None, dpi=None):
        """Local exponent K(n) = d ln<I_n>/d ln I_0 vs power; departs from n at saturation.

        With a dense intensity scan the smooth curve is drawn and the values sampled at
        the actual power-scan points are marked."""
        fig, ax, own = self._ax(ax, figsize=(9, 6), dpi=dpi)
        if own:
            self._ylim['local_slope'] = ylim
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
                         xlim=None, ylim=None, dpi=None):
        """g^(2)(0) vs I_0 — a family of distinct curves (one per harmonic / pair).

        ``ylim`` defaults to ``None`` (auto-scale), so a scan whose g^(2) spans a wide
        range (e.g. 1.3 to 9 across powers) is shown fully instead of being clipped."""
        fig, ax, own = self._ax(ax, dpi=dpi)
        if own:
            self._ylim['g2'] = ylim
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

    def plot_R_vs_power(self, ax=None, harmonics=None, pairs=None, xlim=None, ylim=None,
                        dpi=None):
        """Cauchy-Schwarz ``R`` vs pump power, one curve per cross pair (m, n).

        Companion to :meth:`plot_g2_vs_power`: where ``g^(2)`` shows the bunching of
        each harmonic, ``R`` shows how strongly two harmonics are correlated relative
        to the classical bound ``R=1``."""
        fig, ax, own = self._ax(ax, dpi=dpi)
        if own:
            self._ylim['R'] = ylim
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

    def plot_g2_grid_vs_power(self, ylim=None, figsize=(16, 18), dpi=None):
        """Grid of g^(2)(0) **vs pump power**, one panel per detector pair.

        Row 0 holds the auto-correlations ``H_{nn}`` (R&T); the next four rows are
        the cross pairs in each arm combination (TT, TR, RT, RR). Companion to the
        integration-window grid, but here the x-axis is the pump power. Returns
        ``(fig, axes)``."""
        hs = list(self.harmonics)
        pairs = [(hs[i], hs[j]) for i in range(len(hs)) for j in range(i + 1, len(hs))]
        ncol = max(len(hs), len(pairs), 1)
        fig, axes = plt.subplots(1 + len(_CROSS_ROWS), ncol,
                                 figsize=figsize, dpi=dpi or _fit_dpi(figsize[0]), squeeze=False)
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
        header = self._header_text(max_chars=max(80, int(figsize[0] * 11)))
        nlines = header.count("\n") + 1
        fig.suptitle(r"$g^{(2)}(0)$ vs pump power — every detector pair",
                     fontsize=18, y=0.998, va='top')
        fig.text(0.5, 0.973 if nlines >= 2 else 0.965, header, ha='center', va='top',
                 fontsize=10, color='#555555')
        fig.subplots_adjust(top=0.925 if nlines >= 2 else 0.93, hspace=0.34, wspace=0.22,
                            left=0.06, right=0.98, bottom=0.05)
        return fig, axes

    def plot_R_grid_vs_power(self, ylim=None, figsize=(16, 15), dpi=None):
        """Grid of Cauchy-Schwarz ``R`` **vs pump power**, one panel per cross pair
        and arm combination (TT, TR, RT, RR). Returns ``(fig, axes)``."""
        hs = list(self.harmonics)
        pairs = [(hs[i], hs[j]) for i in range(len(hs)) for j in range(i + 1, len(hs))]
        ncol = max(len(pairs), 1)
        fig, axes = plt.subplots(len(_CROSS_ROWS), ncol,
                                 figsize=figsize, dpi=dpi or _fit_dpi(figsize[0]), squeeze=False)
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
        header = self._header_text(max_chars=max(80, int(figsize[0] * 11)))
        nlines = header.count("\n") + 1
        fig.suptitle(r"Cauchy-Schwarz $R$ vs pump power — every detector pair",
                     fontsize=18, y=0.998, va='top')
        fig.text(0.5, 0.967 if nlines >= 2 else 0.96, header, ha='center', va='top',
                 fontsize=10, color='#555555')
        fig.subplots_adjust(top=0.915 if nlines >= 2 else 0.92, hspace=0.34, wspace=0.22,
                            left=0.06, right=0.98, bottom=0.06)
        return fig, axes

    def plot_g2_collapse(self, ax=None, slope='local', include_cross=True, show_mean=True,
                         harmonics=None, pairs=None, xlim=None, ylim=(0, 0.015), dpi=None):
        """(g^(2)_n - 1)/K^2(n) vs I_0 — all harmonics should COLLAPSE onto g2_0 - 1."""
        fig, ax, own = self._ax(ax, dpi=dpi)
        if own:
            self._ylim['collapse'] = ylim
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
                      g2_ylim=None, collapse_ylim=None, intensity_ylim=None,
                      local_slope_ylim=None, dpi=None):
        """2x2 dashboard mirroring the model: g^(2) vs power, the rescaled collapse, the
        intensity scaling (log-log) and the local exponent K(n). Returns (fig, axes).

        Any panel ``*_ylim`` left as ``None`` reuses the y-limits that the matching
        standalone plot was last drawn with (e.g. a ``plot_g2_collapse(ylim=(0, 0.5))``
        run just above), so the dashboard mirrors the single plots instead of falling
        back to a default that may clip the data."""
        g2_ylim = g2_ylim if g2_ylim is not None else self._ylim.get('g2')
        if collapse_ylim is None:
            collapse_ylim = self._ylim.get('collapse', (0, 0.015))
        if intensity_ylim is None:
            intensity_ylim = (self._ylim.get('intensity') or (None, None))[1]
        if local_slope_ylim is None:
            local_slope_ylim = self._ylim.get('local_slope')
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=dpi or _fit_dpi(18))
        self.plot_g2_vs_power(ax=axes[0, 0], include_cross=True, harmonics=harmonics,
                              pairs=pairs, ylim=g2_ylim)
        axes[0, 0].set_title(r"(1) $g^{(2)}(0)$ vs pump power", fontsize=15)
        self.plot_g2_collapse(ax=axes[0, 1], slope=slope, harmonics=harmonics, pairs=pairs,
                              ylim=collapse_ylim)
        axes[0, 1].set_title(r"(2) Rescaled collapse $\to g^{(2)}_0 - 1 = \sigma^2$", fontsize=15)
        self.plot_intensity_scaling(ax=axes[1, 0], n_fit=n_fit, split=split,
                                    per_channel=per_channel, harmonics=harmonics,
                                    channels=channels, ylim=intensity_ylim)
        axes[1, 0].set_title(r"(3) Intensity scaling $\langle I_n\rangle \sim I_0^{K(n)}$", fontsize=15)
        self.plot_local_slope(ax=axes[1, 1], ylim=local_slope_ylim)
        axes[1, 1].set_title(r"(4) Local exponent $K(n)$  (dotted = ideal $n$)", fontsize=15)
        header = self._header_text(max_chars=180)
        nlines = header.count("\n") + 1
        fig.suptitle("Power scan — harmonic fluctuation model", fontsize=19, y=0.985,
                     va='top')
        fig.text(0.5, 0.955 if nlines >= 2 else 0.95, header, ha='center', va='top',
                 fontsize=11, color='#555555')
        fig.subplots_adjust(top=0.875 if nlines >= 2 else 0.90, hspace=0.20, wspace=0.20,
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
    def _ax(ax, figsize=(6.5, 4.5), dpi=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi or _fit_dpi(figsize[0]))
            return fig, ax, True
        return ax.figure, ax, False

    def _cross_pairs(self, harmonics=None):
        hs = list(harmonics or self.harmonics)
        return [(hs[i], hs[j]) for i in range(len(hs)) for j in range(i + 1, len(hs))]

    def _legend_side(self, fig, handles, fontsize=9):
        """Scan-colour legend to the right of the figure (one entry per scan)."""
        max_len = max((len(h.get_label()) for h in handles), default=8)
        legend_width = min(0.30, 0.07 + 0.011 * max_len)
        axes_right = 1.0 - legend_width - 0.02
        fig.legend(handles=handles, loc='center left',
                   bbox_to_anchor=(axes_right + 0.008, 0.5), ncol=1,
                   frameon=True, fontsize=fontsize, framealpha=0.92, edgecolor='0.7')
        return axes_right

    def _finish(self, fig, ax, own, title, subtitle=None, legend='scans',
                harmonics=None, pair_legend=None, marker='o'):
        """Finish a standalone comparison axes.

        ``legend='scans'``: one entry per scan (colour only) on the right; marker /
        linestyle inside the axes encodes harmonic or R pair. ``legend='none'``: for
        dashboard sub-panels."""
        if own:
            fig.suptitle(title, fontsize=15, y=0.96)
            if subtitle:
                ax.set_title(subtitle, fontsize=9.5, color='#555555', pad=8)
        if legend == 'none':
            if own:
                fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.12)
            return fig, ax
        axes_right = self._legend_side(fig, self._scan_legend_handles(marker))
        if pair_legend is not None:
            inset = self._pair_marker_handles(pair_legend)
            inset_title = 'Pair'
        elif harmonics is not None:
            inset = self._harmonic_marker_handles(harmonics)
            inset_title = 'Harmonic'
        else:
            inset = None
        if inset:
            ax.legend(handles=inset, loc='upper left', fontsize=8, framealpha=0.92,
                      edgecolor='0.7', title=inset_title, title_fontsize=8)
        if own:
            fig.subplots_adjust(left=0.12, right=axes_right, top=0.88, bottom=0.12)
        return fig, ax

    def _finish_grid(self, fig, title, marker='o', top=0.94, **adjust_kw):
        """Shared scan legend + margins for multi-panel comparison grids."""
        axes_right = self._legend_side(fig, self._scan_legend_handles(marker), fontsize=10)
        fig.suptitle(title, fontsize=16, y=0.98, va='top')
        kw = dict(top=top, hspace=0.36, wspace=0.22,
                  left=0.06, right=axes_right, bottom=0.06)
        kw.update(adjust_kw)
        fig.subplots_adjust(**kw)
        return fig

    def _hmark(self, i):
        return ['o', 's', '^', 'D', 'v', 'P'][i % 6]

    def _hls(self, i):
        """Line style per harmonic (colour already encodes the scan), so harmonics
        stay distinguishable within one scan."""
        return ['-', '--', ':', '-.'][i % 4]

    def _scan_legend_handles(self, marker='o'):
        """One legend entry per scan (colour = scan)."""
        return [Line2D([0], [0], color=self.color[lab], marker=marker, ls='-', ms=6,
                       label=lab) for lab in self.labels]

    def _harmonic_marker_handles(self, harmonics):
        """Marker / linestyle key for harmonics (colour already encodes the scan)."""
        hs = list(harmonics or self.harmonics)
        return [Line2D([0], [0], color='0.35', marker=self._hmark(i), ls=self._hls(i),
                         ms=6, lw=1.8, label=rf"$H_{{{n}}}$")
                for i, n in enumerate(hs)]

    def _pair_marker_handles(self, pairs):
        """Marker / linestyle key for Cauchy-Schwarz cross pairs."""
        return [Line2D([0], [0], color='0.35', marker=self._hmark(i), ls=self._hls(i),
                         ms=6, lw=1.8, label=rf"$R_{{{m}{n}}}$")
                for i, (m, n) in enumerate(pairs)]

    def plot_g2_vs_power(self, ax=None, harmonics=None, xlim=None, ylim=(0.9, 1.6),
                         dpi=None, legend='scans'):
        """Overlay g^(2)_nn(P) for every scan (colour = scan, marker = harmonic)."""
        fig, ax, own = self._ax(ax, dpi=dpi)
        hs = harmonics or self.harmonics
        for lab, a in self.scans.items():
            for i, n in enumerate(hs):
                if n not in a.harmonics:
                    continue
                ax.plot(a.I0, a.g2_auto[n], marker=self._hmark(i), ls='-',
                        color=self.color[lab], ms=6)
        ax.axhline(1.0, color='#313131', ls='--', lw=1.2, alpha=0.7)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$g^{(2)}(0)$")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return self._finish(fig, ax, own, r"$g^{(2)}(0)$ vs pump power — comparison",
                            harmonics=hs if legend == 'scans' else None, legend=legend)

    def plot_R_vs_power(self, ax=None, harmonics=None, pairs=None, xlim=None, ylim=None,
                        dpi=None, legend='scans'):
        """Overlay Cauchy-Schwarz ``R`` vs pump power for every scan (colour = scan,
        marker / linestyle = cross pair). Companion to :meth:`plot_g2_vs_power`."""
        fig, ax, own = self._ax(ax, dpi=dpi)
        hs = list(harmonics or self.harmonics)
        cross = self._cross_pairs(hs)
        if pairs is not None:
            keep = set(pairs) | {(b, a) for a, b in pairs}
            cross = [p for p in cross if p in keep]
        drew = False
        for lab, a in self.scans.items():
            if not a.g2_cross:
                continue
            for pi, (m, n) in enumerate(cross):
                if (m, n) not in a.g2_cross:
                    continue
                ax.plot(a.I0, a.R_cross(m, n), marker=self._hmark(pi), ls=self._hls(pi),
                        color=self.color[lab], ms=6, lw=1.8)
                drew = True
        ax.axhline(1.0, color='#313131', ls='--', lw=1.2, alpha=0.7)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$R = g^{(2)\,2}_{mn} / (g^{(2)}_{mm}\, g^{(2)}_{nn})$")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if not drew and own:
            ax.text(0.5, 0.5, "No cross-pair $R$ data", transform=ax.transAxes,
                    ha='center', va='center', color='0.45')
        return self._finish(fig, ax, own, r"Cauchy-Schwarz $R$ vs pump power — comparison",
                            pair_legend=cross if legend == 'scans' and cross else None,
                            legend=legend, marker='s')

    def plot_local_slope(self, ax=None, harmonics=None, xlim=None, ylim=None, dpi=None,
                         legend='scans'):
        """Overlay K(n)(P) for every scan."""
        fig, ax, own = self._ax(ax, figsize=(9, 6), dpi=dpi)
        hs = harmonics or self.harmonics
        for lab, a in self.scans.items():
            for i, n in enumerate(hs):
                if n not in a.harmonics:
                    continue
                Pc, Kc = a.exponent_curve(n)
                ax.semilogx(Pc, Kc, ls=self._hls(i), color=self.color[lab],
                            marker=(None if a.has_dense else self._hmark(i)),
                            ms=6, lw=2.0)
        for n in hs:
            ax.axhline(n, color='0.6', ls=':', lw=1.1, alpha=0.6)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$K(n) = \mathrm{d}\ln\langle I_n\rangle/\mathrm{d}\ln I_0$")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return self._finish(fig, ax, own, r"Effective nonlinearity $K(n)$ — comparison",
                            harmonics=hs if legend == 'scans' else None, legend=legend)

    # ---------------- per-detector-pair grids vs power (overlay every scan) ----------------

    def plot_g2_grid_vs_power(self, harmonics=None, ylim=None, figsize=(16, 18),
                              dpi=None):
        """Grid of g^(2)(0) vs pump power, one panel per detector pair, OVERLAYING every
        scan (colour = scan). Row 0 holds the auto-correlations ``H_{nn}`` (R&T); the
        next four rows are the cross pairs in each arm combination (TT, TR, RT, RR).
        Returns ``(fig, axes)``."""
        hs = list(harmonics or self.harmonics)
        pairs = [(hs[i], hs[j]) for i in range(len(hs)) for j in range(i + 1, len(hs))]
        ncol = max(len(hs), len(pairs), 1)
        fig, axes = plt.subplots(1 + len(_CROSS_ROWS), ncol, figsize=figsize,
                                 dpi=dpi or _fit_dpi(figsize[0]), squeeze=False)
        for j in range(ncol):
            ax = axes[0, j]
            if j < len(hs):
                n = hs[j]
                drew = False
                for lab, a in self.scans.items():
                    if n not in a.harmonics:
                        continue
                    y = a._pair_g2_vs_power(f"H{n}R", f"H{n}T")
                    if y is None:
                        continue
                    ax.plot(a.I0, y, 'o-', color=self.color[lab], ms=5, lw=1.6)
                    drew = True
                if drew:
                    ax.axhline(1.0, color='#313131', ls='--', lw=1.0, alpha=0.6)
                    ax.grid(True, alpha=0.25)
                    if ylim is not None:
                        ax.set_ylim(ylim)
                    ax.set_title(rf"auto $g^{{(2)}}_{{{n}{n}}}$ (RT)", fontsize=12)
                else:
                    ax.set_visible(False)
            else:
                ax.set_visible(False)
            for ri, row in enumerate(_CROSS_ROWS, start=1):
                ax = axes[ri, j]
                if j < len(pairs):
                    m, n = pairs[j]
                    drew = False
                    for lab, a in self.scans.items():
                        y = a._pair_g2_vs_power(f"H{m}{row[0]}", f"H{n}{row[1]}")
                        if y is None:
                            continue
                        ax.plot(a.I0, y, 's-', color=self.color[lab], ms=5, lw=1.6)
                        drew = True
                    if drew:
                        ax.axhline(1.0, color='#313131', ls='--', lw=1.0, alpha=0.6)
                        ax.grid(True, alpha=0.25)
                        if ylim is not None:
                            ax.set_ylim(ylim)
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
        self._finish_grid(fig, r"$g^{(2)}(0)$ vs pump power — every detector pair (comparison)",
                          marker='o', top=0.96)
        return fig, axes

    def plot_R_grid_vs_power(self, harmonics=None, ylim=None, figsize=(16, 15),
                             dpi=None):
        """Grid of Cauchy-Schwarz ``R`` vs pump power, one panel per cross pair and arm
        combination (TT, TR, RT, RR), OVERLAYING every scan (colour = scan). Returns
        ``(fig, axes)``."""
        hs = list(harmonics or self.harmonics)
        pairs = [(hs[i], hs[j]) for i in range(len(hs)) for j in range(i + 1, len(hs))]
        ncol = max(len(pairs), 1)
        fig, axes = plt.subplots(len(_CROSS_ROWS), ncol, figsize=figsize,
                                 dpi=dpi or _fit_dpi(figsize[0]), squeeze=False)
        for ri, row in enumerate(_CROSS_ROWS):
            for j in range(ncol):
                ax = axes[ri, j]
                if j >= len(pairs):
                    ax.set_visible(False)
                    continue
                m, n = pairs[j]
                drew = False
                for lab, a in self.scans.items():
                    R = a._pair_R_vs_power(
                        (f"H{m}{row[0]}", f"H{n}{row[1]}"),
                        (f"H{m}R", f"H{m}T"), (f"H{n}R", f"H{n}T"))
                    if R is None:
                        continue
                    ax.plot(a.I0, R, 's-', color=self.color[lab], ms=5, lw=1.6)
                    drew = True
                if not drew:
                    ax.set_visible(False)
                    continue
                ax.axhline(1.0, color='#313131', ls='--', lw=1.0, alpha=0.6)
                ax.grid(True, alpha=0.25)
                if ylim is not None:
                    ax.set_ylim(ylim)
                ax.set_title(rf"$R_{{{m}{n}}}$ ({row})", fontsize=12)
                if ri == len(_CROSS_ROWS) - 1:
                    ax.set_xlabel(r"Pump power $P$ (mW)")
                if j == 0:
                    ax.set_ylabel(r"$R$")
        self._finish_grid(fig, r"Cauchy-Schwarz $R$ vs pump power — every detector pair (comparison)",
                          marker='s', top=0.96)
        return fig, axes

    def plot_intensity_scaling(self, ax=None, harmonics=None, xlim=None, ylim=None,
                               dpi=None, legend='scans'):
        """Overlay the merged-harmonic intensity scaling <I_n>(P) for every scan."""
        fig, ax, own = self._ax(ax, dpi=dpi)
        hs = harmonics or self.harmonics
        for lab, a in self.scans.items():
            for i, n in enumerate(hs):
                if n not in a.harmonics:
                    continue
                if a.has_dense and n in a.In_dense:
                    ax.loglog(a.P_dense, a.In_dense[n], '.', color=self.color[lab],
                              ms=2.5, alpha=0.2)
                ax.loglog(a.I0, a.In[n], marker=self._hmark(i), ls=self._hls(i),
                          color=self.color[lab], ms=6, mec='white', mew=0.6)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"Intensity (counts/s)")
        ax.grid(True, which='both', alpha=0.25)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return self._finish(fig, ax, own,
                            r"Harmonic intensity scaling $\langle I_n\rangle \sim I_0^{K(n)}$ — comparison",
                            harmonics=hs if legend == 'scans' else None, legend=legend)

    def plot_inferred_sigma2(self, ax=None, slope='local', harmonics=None,
                             include_cross=True, xlim=None, ylim=None, dpi=None,
                             legend='scans'):
        """Overlay the inferred pump excess sigma^2 = g2_0 - 1 (mean over harmonics)
        for every scan, with its 1-sigma band — the headline comparison."""
        fig, ax, own = self._ax(ax, dpi=dpi)
        for lab, a in self.scans.items():
            mean, std = a.inferred_g2_0(slope, harmonics=harmonics,
                                        pairs=(None if include_cross else []))
            c = self.color[lab]
            ax.plot(a.I0, mean, 'o-', color=c, ms=6)
            ax.fill_between(a.I0, mean - std, mean + std, color=c, alpha=0.13)
        ax.set_xlabel(r"Pump power $P \propto I_0$ (mW)")
        ax.set_ylabel(r"$\sigma^2 = g^{(2)}_0 - 1$")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return self._finish(fig, ax, own,
                            r"Inferred pump excess $\sigma^2$ — comparison",
                            legend=legend)

    def plot_overview(self, slope='local', harmonics=None,
                      g2_ylim=(0.9, 1.6), sigma2_ylim=None, dpi=None):
        """2x2 comparison dashboard: g^(2) vs power, inferred sigma^2, intensity
        scaling and K(n), each overlaying every scan. Returns (fig, axes)."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=dpi or _fit_dpi(18))
        hs = harmonics or self.harmonics
        self.plot_g2_vs_power(ax=axes[0, 0], harmonics=hs, ylim=g2_ylim, legend='none')
        axes[0, 0].set_title(r"(1) $g^{(2)}(0)$ vs pump power", fontsize=15)
        self.plot_inferred_sigma2(ax=axes[0, 1], slope=slope, harmonics=hs,
                                  ylim=sigma2_ylim, legend='none')
        axes[0, 1].set_title(r"(2) Inferred $\sigma^2 = g^{(2)}_0 - 1$", fontsize=15)
        self.plot_intensity_scaling(ax=axes[1, 0], harmonics=hs, legend='none')
        axes[1, 0].set_title(r"(3) Intensity scaling $\langle I_n\rangle \sim I_0^{K(n)}$",
                             fontsize=15)
        axes[1, 0].legend(handles=self._harmonic_marker_handles(hs), loc='upper left',
                          fontsize=7, framealpha=0.92, edgecolor='0.7',
                          title='Harmonic', title_fontsize=7)
        self.plot_local_slope(ax=axes[1, 1], harmonics=hs, legend='none')
        axes[1, 1].set_title(r"(4) Local exponent $K(n)$", fontsize=15)
        axes[1, 1].legend(handles=self._harmonic_marker_handles(hs), loc='upper left',
                          fontsize=7, framealpha=0.92, edgecolor='0.7',
                          title='Harmonic', title_fontsize=7)
        fig.suptitle("Power-scan comparison", fontsize=19, y=0.98)
        axes_right = self._legend_side(fig, self._scan_legend_handles('o'), fontsize=10)
        fig.subplots_adjust(top=0.92, hspace=0.28, wspace=0.28,
                            left=0.07, right=axes_right, bottom=0.08)
        return fig, axes

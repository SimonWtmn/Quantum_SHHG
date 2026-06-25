"""
=============================================================================
HBT Analysis Report / Orchestration
=============================================================================

The glue between the engine (:mod:`src.core`) and the figure builders
(:mod:`src.visu`) that the day-to-day notebooks drive. It:

* discovers and loads runs from the ``data/`` tree (preferring the MERGED pickle),
* runs the full plot suite for each observable (coherence / g^(2) / R) as the
  comparison GRID *and* every detector pair INDIVIDUALLY,
* saves every figure to a tidy, title-driven folder
  ``results/<sample>/<title>/<date>/<observable>/`` so every run of the same
  configuration stays grouped and easy to find without manual bookkeeping,
* displays figures with the interactive ipympl backend (drag-to-zoom on any plot),
  while still writing the static PNGs to disk.

Typical use (single test)::

    from src.report import AnalysisReport
    rep = AnalysisReport("data/Jun18/RMS/337-335", title="RMS_337-335",
                         results_root="results")
    rep.run_all(coherence=dict(xlim=150, integration_window_ns=10),
                g2=dict(methods=["direct", "delay"]),
                R=dict(methods=["direct"]))

Typical use (end-of-day comparison)::

    from src.report import discover_runs, AnalysisReport
    runs = discover_runs("data/Jun18/RMS")
    rep = AnalysisReport(runs, title="RMS_day_compare",
                         comparison_variable="Rotation angle")
    rep.run_all()

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquee (LOA), Ecole Polytechnique
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from .core import HBTMeasurement
from .visu import GridVisualizer, CROSS_ROWS, CROSS_COLS, AUTO_COLS
from .powerscan import PowerScanAnalyzer, PowerScanComparison

__all__ = [
    "AnalysisReport",
    "PowerScanReport",
    "PowerScanComparisonReport",
    "load_run",
    "discover_runs",
    "discover_power_scan",
    "find_run_pkl",
]


# =============================================================================
# Folder-name helpers (results are grouped <sample>/<title>/<date>/)
# =============================================================================

def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _safe(title):
    """Turn an arbitrary label into a safe folder name."""
    return re.sub(r"[^\w\-.+]+", "_", str(title)).strip("_")


def _sample_of(run):
    """Folder-safe sample name used to group results (e.g. 'CdTe110')."""
    name = getattr(run, "sample", None) or getattr(run, "material", None) or "Unknown"
    return _safe(name) or "Unknown"


def _results_dir(results_root, run, title, date):
    """Common results layout: ``<results_root>/<sample>/<title>/<date>/``.

    Grouping by sample then by the descriptive title (and only then by date) keeps
    every run of the same configuration together and easy to find, instead of
    burying it under the acquisition date.
    """
    return Path(results_root) / _sample_of(run) / _safe(title) / date


# =============================================================================
# Discovery / loading
# =============================================================================

def find_run_pkl(path):
    """Resolve a run directory (or a file) to the best data pickle.

    Preference order: ``MERGED/*_MERGED.pkl`` > ``*_chunk0.pkl`` > any ``*.pkl``
    (ignoring the lightweight ``*.meta.pkl`` written by the raw recorder).
    A path already pointing at a ``.pkl`` is returned unchanged.
    """
    p = Path(path)
    if p.is_file() and p.suffix == ".pkl":
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"No run found at {p}")

    merged = sorted(p.glob("MERGED/*_MERGED.pkl"))
    if merged:
        return merged[0]

    def _data_pkls(folder):
        return sorted(f for f in folder.glob("*.pkl") if not f.name.endswith(".meta.pkl"))

    chunk0 = [f for f in _data_pkls(p) if "chunk0" in f.name]
    if chunk0:
        return chunk0[0]
    any_pkl = _data_pkls(p)
    if any_pkl:
        return any_pkl[0]
    raise FileNotFoundError(f"No analysable .pkl under {p}")


def load_run(path):
    """Load a single run as an :class:`HBTMeasurement` from a dir or a .pkl file."""
    return HBTMeasurement(find_run_pkl(path))


def _loose_point_pkls(folder):
    """Independent per-point pickles sitting loose in one folder (e.g. a
    count-rate intensity scan: ``ang_44.pkl``, ``ang_46.pkl``, ...).

    Returns them only when the folder is NOT a single chunked run, i.e. it has no
    ``MERGED/`` and no ``*chunk*`` pickles. Such a folder holds many runs (one per
    file), not one run made of chunks — so each file is its own measurement.
    """
    folder = Path(folder)
    if list(folder.glob("MERGED/*_MERGED.pkl")):
        return []
    loose = [f for f in folder.glob("*.pkl") if not f.name.endswith(".meta.pkl")]
    if any("chunk" in f.name for f in loose):
        return []
    return sorted(loose) if len(loose) >= 2 else []


def discover_runs(root, prefer_merged=True, sort_key="time"):
    """Find every acquisition run under ``root`` and load it.

    A "run" is any directory that yields a data pickle via :func:`find_run_pkl`
    (a ``MERGED/`` or chunk pickles). Folders holding many loose per-point pickles
    (e.g. a count-rate intensity scan, one ``ang_*.pkl`` per angle) contribute one
    run PER file. Returns a list of :class:`HBTMeasurement`, sorted by ``sort_key``
    in {'time', 'angle', 'power', 'config', 'name'}.
    """
    root = Path(root)
    seen, runs = set(), []
    candidates = [root] + [d for d in root.rglob("*") if d.is_dir()]
    for d in candidates:
        if d.name.upper() == "MERGED":
            continue
        loose = _loose_point_pkls(d)
        if loose:
            for pkl in loose:
                key = pkl.resolve()
                if key in seen:
                    continue
                seen.add(key)
                try:
                    runs.append(HBTMeasurement(pkl))
                except Exception as exc:  # noqa: BLE001 - keep going
                    print(f"[discover_runs] skipped {pkl}: {exc}")
            continue
        try:
            pkl = find_run_pkl(d)
        except FileNotFoundError:
            continue
        key = pkl.resolve()
        if key in seen:
            continue
        seen.add(key)
        try:
            runs.append(HBTMeasurement(pkl))
        except Exception as exc:  # noqa: BLE001 - skip unreadable runs, keep going
            print(f"[discover_runs] skipped {pkl}: {exc}")

    def _key(r):
        if sort_key == "angle":
            return (r.rotation_stage if r.rotation_stage is not None else 1e9)
        if sort_key == "power":
            return (r.power_mw if r.power_mw is not None else 1e9)
        if sort_key == "config":
            return r.legend_tag()
        if sort_key == "name":
            return r.run_dir.name
        return getattr(r, "_t_key", r.run_dir.name)

    # Stable time key from the filename timestamp when present.
    for r in runs:
        m = re.search(r"(\d{2}-\d{2}-\d{2})", r.pkl_path.stem)
        r._t_key = m.group(1) if m else r.run_dir.name
    runs.sort(key=_key)
    return runs


def discover_power_scan(scan_dir):
    """Load every pump-power point of a single power-scan acquisition.

    A power scan is one acquisition directory whose ``MERGED/`` folder holds one
    ``*_MERGED.pkl`` per power (falling back to per-power ``*_chunk0.pkl``). Returns
    a list of :class:`HBTMeasurement` sorted by power (points without a known power
    are dropped).
    """
    scan_dir = Path(scan_dir)
    files = sorted(scan_dir.glob("MERGED/*_MERGED.pkl"))
    if not files:
        files = sorted(f for f in scan_dir.glob("*chunk0.pkl")
                       if not f.name.endswith(".meta.pkl"))
    if not files:
        raise FileNotFoundError(f"No power-scan points found under {scan_dir}")
    runs = [HBTMeasurement(f) for f in files]
    runs = [r for r in runs if r.power_mw is not None]
    runs.sort(key=lambda r: r.power_mw)
    return runs


# =============================================================================
# Single-pair enumerations (mirror the grids)
# =============================================================================

def _coherence_pairs():
    """(name, detA, detB) for the 5x3 coherence / g2 grid, as detector names."""
    pairs = [(f"H{n}{n}_RT", f"H{n}R", f"H{n}T") for n in AUTO_COLS]
    for row in CROSS_ROWS:
        for hA, hB in CROSS_COLS:
            pairs.append((f"H{hA}{hB}_{row}", f"H{hA}{row[0]}", f"H{hB}{row[1]}"))
    return pairs


def _R_triplets():
    """(name, cross, autoA, autoB) for the 4x3 R grid, as detector-name tuples."""
    out = []
    for row in CROSS_ROWS:
        for hA, hB in CROSS_COLS:
            out.append((
                f"R_{hA}{hB}_{row}",
                (f"H{hA}{row[0]}", f"H{hB}{row[1]}"),
                (f"H{hA}R", f"H{hA}T"),
                (f"H{hB}R", f"H{hB}T"),
            ))
    return out


# =============================================================================
# Shared figure sink (save to disk + interactive ipympl display)
# =============================================================================

class _FigureSink:
    """Mixin giving a report the ``_emit`` / `_display` machinery: save a figure as
    a PNG and/or show it with the interactive ipympl backend (drag-to-zoom), capping
    the on-screen size so large grids/dashboards stay readable in the notebook."""

    dpi = 200
    # Target on-screen width (pixels) for interactive figures; an ipympl canvas is
    # sized by inches x dpi, so we cap the *display* dpi to keep large figures on
    # screen. Saved PNGs are unaffected (savefig uses self.dpi).
    DISPLAY_MAX_WIDTH_PX = 980

    def _emit(self, fig, save_path, show):
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        if show:
            self._display(fig)
        else:
            plt.close(fig)

    def _display(self, fig):
        try:
            from IPython.display import display
            canvas = getattr(fig, "canvas", None)
            mod = canvas.__class__.__module__ if canvas is not None else ""
            if "ipympl" in mod or "backend_nbagg" in mod:
                w_in = fig.get_size_inches()[0]
                if w_in > 0:
                    fig.set_dpi(min(fig.get_dpi(), self.DISPLAY_MAX_WIDTH_PX / w_in))
                try:
                    canvas.header_visible = False
                    canvas.footer_visible = False
                    canvas.resizable = False
                    canvas.layout.width = "100%"
                    canvas.layout.height = "auto"
                except Exception:          # noqa: BLE001
                    pass
                display(canvas)            # interactive (drag-to-zoom)
            else:
                display(fig)               # inline / static fallback
        except Exception:                  # noqa: BLE001
            plt.show()


# =============================================================================
# Report
# =============================================================================

class AnalysisReport(_FigureSink):
    """Drive the full plot suite for one run or a set of runs and save everything.

    Parameters
    ----------
    runs : str | Path | HBTMeasurement | list of those
        A single run (directory or .pkl), an :class:`HBTMeasurement`, or a list to
        overlay (comparison). Strings/paths are loaded with :func:`load_run`.
    title : str
        Drives both the results subfolder and stays out of the plot title clutter.
    results_root : str | Path
        Base results directory (default ``"results"``).
    date : str, optional
        Date folder (default: the first run's acquisition date, else today).
    comparison_variable : str, optional
        What varies across overlaid runs (shown in grid titles).
    show_details : bool
        Add the extra acquisition metadata line to every title.
    dpi : int
        Save resolution for the PNGs.
    """

    OBSERVABLES = ("coherence", "g2", "R")

    def __init__(self, runs, title, results_root="results", date=None,
                 comparison_variable=None, show_details=False, dpi=200):
        self.runs = self._load(runs)
        self.title = title
        self.dpi = dpi
        r0 = self.runs[0]
        self.date = date or (r0.date if r0.date and r0.date != "Unknown"
                             else _today())
        self.out_dir = _results_dir(results_root, r0, title, self.date)
        self.visu = GridVisualizer(self.runs, comparison_variable=comparison_variable,
                                   show_details=show_details)
        # Keep figure creation from auto-displaying; we control display explicitly so
        # the to-be-saved-only singles do not flood the notebook.
        plt.ioff()
        print(f"AnalysisReport '{title}': {len(self.runs)} run(s) "
              f"({'comparison' if self.visu.is_comparison else 'single'}), "
              f"source={self.visu.histogram_source}")
        print(f"  -> results: {self.out_dir}")

    @staticmethod
    def _load(runs):
        if not isinstance(runs, (list, tuple)):
            runs = [runs]
        out = []
        for r in runs:
            out.append(r if isinstance(r, HBTMeasurement) else load_run(r))
        if not out:
            raise ValueError("No runs to analyse.")
        return out

    # -- IO / display ---------------------------------------------------------

    def _dir(self, observable):
        d = self.out_dir / observable
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _has_dets(self, names):
        return all(self.runs[0].has_detector(n) for n in names)

    def _ch(self, name):
        return self.runs[0].get_ch(name)

    # -- suites ---------------------------------------------------------------

    def coherence(self, grid=True, singles=True, show_grid=True, show_singles=False,
                  **kw):
        """Coherence spectra: grid + every pair, all saved under coherence/."""
        d = self._dir("coherence")
        if grid:
            fig, _ = self.visu.plot_coherence(**kw)
            self._emit(fig, d / "grid.png", show_grid)
        if singles:
            for name, a, b in _coherence_pairs():
                if not self._has_dets((a, b)):
                    continue
                fig, _ = self.visu.plot_coherence(self._ch(a), self._ch(b), **kw)
                self._emit(fig, d / f"{name}.png", show_singles)
        print(f"  coherence -> {d}")

    def g2(self, grid=True, singles=True, show_grid=True, show_singles=False, **kw):
        """g^(2)(0) sweeps: grid + every pair, all saved under g2/."""
        d = self._dir("g2")
        if grid:
            fig, _ = self.visu.plot_g2(**kw)
            self._emit(fig, d / "grid.png", show_grid)
        if singles:
            for name, a, b in _coherence_pairs():
                if not self._has_dets((a, b)):
                    continue
                fig, _ = self.visu.plot_g2(self._ch(a), self._ch(b), **kw)
                self._emit(fig, d / f"{name}.png", show_singles)
        print(f"  g2        -> {d}")

    def R(self, grid=True, singles=True, show_grid=True, show_singles=False, **kw):
        """Cauchy-Schwarz R sweeps: grid + every cross triplet, saved under R/."""
        d = self._dir("R")
        if grid:
            fig, _ = self.visu.plot_R(**kw)
            self._emit(fig, d / "grid.png", show_grid)
        if singles:
            for name, cross, aA, aB in _R_triplets():
                if not self._has_dets(cross + aA + aB):
                    continue
                fig, _ = self.visu.plot_R(
                    cross_pair=(self._ch(cross[0]), self._ch(cross[1])),
                    auto_pair_1=(self._ch(aA[0]), self._ch(aA[1])),
                    auto_pair_2=(self._ch(aB[0]), self._ch(aB[1])), **kw)
                self._emit(fig, d / f"{name}.png", show_singles)
        print(f"  R         -> {d}")

    def run_all(self, coherence=None, g2=None, R=None, grid=True, singles=True,
                show_grid=True, show_singles=False):
        """Run all three suites. Pass per-observable kwargs as dicts (or None to
        skip that observable)."""
        if coherence is not None:
            self.coherence(grid=grid, singles=singles, show_grid=show_grid,
                           show_singles=show_singles, **coherence)
        if g2 is not None:
            self.g2(grid=grid, singles=singles, show_grid=show_grid,
                    show_singles=show_singles, **g2)
        if R is not None:
            self.R(grid=grid, singles=singles, show_grid=show_grid,
                   show_singles=show_singles, **R)
        print(f"Done. All figures under {self.out_dir}")


# =============================================================================
# Power-scan reports
# =============================================================================

class PowerScanReport(_FigureSink):
    """Apply the full intensity-fluctuation model to ONE power scan and save/show
    every figure (the four model plots + a 2x2 dashboard), plus optionally the
    coherence / g^(2) / R comparison grids across powers.

    Parameters
    ----------
    runs : str | Path | HBTMeasurement | list
        The full power-scan points. A single directory is auto-expanded with
        :func:`discover_power_scan`; a list of dirs/pkls/measurements is loaded as-is.
    title : str
        Drives ``results/<sample>/<title>/<date>/`` (model plots under ``model/``).
    intensity_runs : str | Path | list, optional
        A DENSE intensity-only scan (e.g. a :class:`~src.measurement.CountrateRecorder`
        angle sweep) used only to compute a smoother ``K(n)``. A directory is expanded
        with :func:`discover_runs`.
    malus : dict, optional
        ``{'p_max', 'theta0', 'offset'}`` to convert a dense run's stage angle to pump
        power (see :func:`src.powerscan.malus_power`).
    harmonics, tau_in_ns, g2_method, g2_source, intensity, k_poly_deg
        Forwarded to :class:`~src.powerscan.PowerScanAnalyzer`.
    """

    def __init__(self, runs, title, results_root="results", date=None,
                 intensity_runs=None, malus=None, harmonics=(3, 4, 5),
                 tau_in_ns=4.0, g2_method="delay", g2_source="auto",
                 intensity="countrate", k_poly_deg=3, dpi=200,
                 comparison_variable="Pump power"):
        self.runs = self._expand(runs)
        self.intensity_runs = self._expand(intensity_runs, dense=True) if intensity_runs else []
        self.title = title
        self.dpi = dpi
        self.analyzer = PowerScanAnalyzer(
            self.runs, harmonics=harmonics, tau_in_ns=tau_in_ns, g2_method=g2_method,
            g2_source=g2_source, intensity=intensity, intensity_runs=self.intensity_runs,
            malus=malus, k_poly_deg=k_poly_deg)
        r0 = self.runs[0]
        self.date = date or (r0.date if r0.date and r0.date != "Unknown" else _today())
        self.out_dir = _results_dir(results_root, r0, title, self.date)
        # Re-use the comparison machinery for the coherence/g2/R grids vs power.
        self.grids = AnalysisReport(self.runs, title=title, results_root=results_root,
                                    date=self.date,
                                    comparison_variable=comparison_variable, dpi=dpi)
        plt.ioff()
        dense = (f", dense K(n) on {len(self.intensity_runs)} pts"
                 if self.analyzer.has_dense else "")
        print(f"PowerScanReport '{title}': {len(self.runs)} powers "
              f"({self.analyzer.I0.min():g}-{self.analyzer.I0.max():g} mW){dense}")
        print(f"  -> results: {self.out_dir}")

    @staticmethod
    def _expand(runs, dense=False):
        """Load runs, auto-expanding a single scan directory into its power points."""
        if isinstance(runs, (list, tuple)):
            return AnalysisReport._load(list(runs))
        if isinstance(runs, HBTMeasurement):
            return [runs]
        p = Path(runs)
        if p.is_dir():
            return discover_runs(p) if dense else discover_power_scan(p)
        return [load_run(p)]

    def summary(self):
        """Print the per-power table of I_0, <I_n>, K(n), g^(2) and the collapse."""
        self.analyzer.summary_table()

    def _mdir(self):
        d = self.out_dir / "model"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, fig, name, show=True):
        """Save a single model figure under ``model/<name>.png`` and (optionally)
        display it. Lets a notebook drive ONE plot per line with full control over
        that plot's arguments, e.g.::

            fig, _ = rep.analyzer.plot_g2_vs_power(harmonics=(3, 5), ylim=(0.9, 3))
            rep.save(fig, "g2_vs_power")
        """
        self._emit(fig, self._mdir() / f"{name}.png", show)
        return fig

    def model(self, slope="local", n_fit=None, split=None, per_channel=True,
              harmonics=None, channels=None, pairs=None, include_cross=True,
              g2_ylim=None, r_ylim=None, collapse_ylim=(0, 0.015),
              grid_g2_ylim=None, grid_r_ylim=None, grids=True, intensity_grid=True,
              overview=True, show=True):
        """Build, save and (optionally) display the model plots + dashboard.

        ``g2_ylim``/``r_ylim`` default to ``None`` (auto-scale to the data) so a scan
        whose g^(2) spans e.g. 1.3 to 9 is shown fully rather than clipped.
        """
        d = self._mdir()
        a = self.analyzer
        fig, _ = a.plot_g2_vs_power(include_cross=include_cross, harmonics=harmonics,
                                    pairs=pairs, ylim=g2_ylim)
        self._emit(fig, d / "g2_vs_power.png", show)
        if a.g2_cross:   # R is only defined for cross pairs (>= 2 harmonics)
            fig, _ = a.plot_R_vs_power(harmonics=harmonics, pairs=pairs, ylim=r_ylim)
            self._emit(fig, d / "R_vs_power.png", show)
        if grids:        # per-detector-pair grids vs power (TT/TR/RT/RR)
            fig, _ = a.plot_g2_grid_vs_power(ylim=grid_g2_ylim)
            self._emit(fig, d / "g2_grid_vs_power.png", show)
            if a.g2_cross:
                fig, _ = a.plot_R_grid_vs_power(ylim=grid_r_ylim)
                self._emit(fig, d / "R_grid_vs_power.png", show)
        fig, _ = a.plot_g2_collapse(slope=slope, include_cross=include_cross,
                                    harmonics=harmonics, pairs=pairs, ylim=collapse_ylim)
        self._emit(fig, d / "collapse.png", show)
        fig, _ = a.plot_intensity_scaling(n_fit=n_fit, split=split, per_channel=per_channel,
                                          harmonics=harmonics, channels=channels)
        self._emit(fig, d / "intensity_scaling.png", show)
        if intensity_grid:
            fig, _ = a.plot_intensity_grid(harmonics=harmonics, channels=channels)
            self._emit(fig, d / "intensity_grid.png", show)
        fig, _ = a.plot_local_slope()
        self._emit(fig, d / "local_slope.png", show)
        if overview:
            fig, _ = a.plot_overview(slope=slope, n_fit=n_fit, split=split,
                                     per_channel=per_channel, harmonics=harmonics,
                                     channels=channels, pairs=pairs,
                                     g2_ylim=g2_ylim, collapse_ylim=collapse_ylim)
            self._emit(fig, d / "overview.png", show)
        print(f"  model -> {d}")

    def correlations(self, coherence=None, g2=None, R=None, singles=False,
                     show_grid=True):
        """Save the coherence / g^(2) / R grids across powers (pass per-observable
        kwargs as dicts; None skips that observable)."""
        if coherence is not None:
            self.grids.coherence(singles=singles, show_grid=show_grid,
                                 show_singles=False, **coherence)
        if g2 is not None:
            self.grids.g2(singles=singles, show_grid=show_grid,
                          show_singles=False, **g2)
        if R is not None:
            self.grids.R(singles=singles, show_grid=show_grid,
                         show_singles=False, **R)

    def run_all(self, model=None, coherence=None, g2=None, R=None, show=True):
        """Convenience: model plots + the requested correlation grids."""
        self.model(show=show, **(model or {}))
        self.correlations(coherence=coherence, g2=g2, R=R, show_grid=show)
        print(f"Done. All figures under {self.out_dir}")


class PowerScanComparisonReport(_FigureSink):
    """Overlay the model curves of several power scans (e.g. P1 vs no-P1) and save
    every comparison figure.

    Parameters
    ----------
    scans : dict[str, ...]
        ``{label: scan}`` where ``scan`` is a :class:`~src.powerscan.PowerScanAnalyzer`,
        a :class:`PowerScanReport`, a scan directory, or a list of runs.
    title : str
        Drives ``results/<sample>/<title>/<date>/compare/``.
    harmonics : tuple[int], optional
        Restrict the comparison to these orders.
    **scan_kw
        Forwarded to :class:`~src.powerscan.PowerScanAnalyzer` when a value needs
        building from runs (e.g. ``tau_in_ns``, ``g2_method``).
    """

    def __init__(self, scans, title, results_root="results", date=None,
                 harmonics=None, dpi=200, **scan_kw):
        self.title = title
        self.dpi = dpi
        analyzers = {lab: self._as_analyzer(v, **scan_kw) for lab, v in scans.items()}
        self.comp = PowerScanComparison(analyzers, harmonics=harmonics)
        r0 = next(iter(analyzers.values())).runs[0]
        self.date = date or (r0.date if r0.date and r0.date != "Unknown" else _today())
        self.out_dir = _results_dir(results_root, r0, title, self.date)
        plt.ioff()
        print(f"PowerScanComparisonReport '{title}': {len(analyzers)} scans "
              f"({', '.join(analyzers)})")
        print(f"  -> results: {self.out_dir}")

    @staticmethod
    def _as_analyzer(value, **scan_kw):
        if isinstance(value, PowerScanAnalyzer):
            return value
        if isinstance(value, PowerScanReport):
            return value.analyzer
        runs = PowerScanReport._expand(value)
        return PowerScanAnalyzer(runs, **scan_kw)

    def _cdir(self):
        d = self.out_dir / "compare"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def plots(self, harmonics=None, slope="local", g2_ylim=(0.9, 1.6),
              sigma2_ylim=None, overview=True, show=True):
        """Build, save and (optionally) display all comparison figures."""
        d = self._cdir()
        c = self.comp
        fig, _ = c.plot_g2_vs_power(harmonics=harmonics, ylim=g2_ylim)
        self._emit(fig, d / "g2_vs_power.png", show)
        fig, _ = c.plot_inferred_sigma2(slope=slope, harmonics=harmonics, ylim=sigma2_ylim)
        self._emit(fig, d / "inferred_sigma2.png", show)
        fig, _ = c.plot_intensity_scaling(harmonics=harmonics)
        self._emit(fig, d / "intensity_scaling.png", show)
        fig, _ = c.plot_local_slope(harmonics=harmonics)
        self._emit(fig, d / "local_slope.png", show)
        if overview:
            fig, _ = c.plot_overview(slope=slope, harmonics=harmonics,
                                     g2_ylim=g2_ylim, sigma2_ylim=sigma2_ylim)
            self._emit(fig, d / "overview.png", show)
        print(f"  compare -> {d}")

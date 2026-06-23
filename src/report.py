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
  ``results/<date>/<title>/<observable>/`` so the same notebook can be re-run
  for each test without manual bookkeeping,
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

__all__ = ["AnalysisReport", "load_run", "discover_runs", "find_run_pkl"]


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


def discover_runs(root, prefer_merged=True, sort_key="time"):
    """Find every acquisition run under ``root`` and load it.

    A "run" is any directory that yields a data pickle via :func:`find_run_pkl`
    (i.e. it has a ``MERGED/`` or chunk pickles). Returns a list of
    :class:`HBTMeasurement`, sorted by ``sort_key`` in {'time', 'angle', 'power',
    'config', 'name'}.
    """
    root = Path(root)
    seen, runs = set(), []
    candidates = [root] + [d for d in root.rglob("*") if d.is_dir()]
    for d in candidates:
        if d.name.upper() == "MERGED":
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
# Report
# =============================================================================

class AnalysisReport:
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
                             else datetime.now().strftime("%Y-%m-%d"))
        safe_title = re.sub(r"[^\w\-.+]+", "_", title).strip("_")
        self.out_dir = Path(results_root) / self.date / safe_title
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

    def _emit(self, fig, save_path, show):
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight",
                        facecolor="white")
        if show:
            self._display(fig)
        else:
            plt.close(fig)

    # Target on-screen width (pixels) for interactive figures; an ipympl canvas is
    # sized by inches x dpi, so we cap the *display* dpi to keep large grids on
    # screen. Saved PNGs are unaffected (savefig uses self.dpi).
    DISPLAY_MAX_WIDTH_PX = 980

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

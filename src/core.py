"""
=============================================================================
HBT Measurement Core Engine
=============================================================================

Loads one acquisition (`.pkl`) and exposes its data plus the quantum-optical
metrics g^(2)(0) and the Cauchy-Schwarz parameter R.

Data model (important)
----------------------
For the `g2_heralded_virtual` acquisition format the pickle stores:
* PHYSICAL scalars : `counts_physical` / `countrates_physical` (per detector 1..6)
  and `coincidences_twofold_physical` (15 detector pairs).
* PHYSICAL histograms : `correlations_physical` (15 pairs, {time_bins, counts}).
* VIRTUAL data (per merged harmonic Hn = T + R) : scalars and
  `correlations_virtual` histograms.

Consequently:
* the physical g^(2)(0) is computed from the scalar physical coincidences
  (`compute_g2_direct`) and is always available and artefact-free;
* histogram methods (`compute_g2_delay`, coherence traces) pick their data with
  `source`: 'auto' uses the physical histogram when present and falls back to the
  virtual one otherwise (never silently relabelled).

Improvements over the previous `hbt_core`:
* the repetition rate is read from the file metadata (`laser.rep_rate_hz`) with a
  fallback, instead of being hard-coded;
* sample / filter / polariser / rotation-angle metadata is parsed from the rich
  `material` string (and the folder name as a fallback), so titles and legends are
  correct;
* title helpers are simple and explicit (`title`, `subtitle`, `legend_tag`).

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquee (LOA), Ecole Polytechnique
"""

from __future__ import annotations

import re
import pickle
from pathlib import Path

import numpy as np

DEFAULT_REP_RATE_HZ = 18.66e6


class HBTMeasurement:
    """A single loaded acquisition with its derived g^(2) / R observables."""

    def __init__(self, pkl_filepath):
        self.pkl_path = Path(pkl_filepath)
        self.dir_path = self.pkl_path.parent
        self.run_dir = (self.dir_path.parent
                        if self.dir_path.name.upper() == "MERGED" else self.dir_path)

        with open(self.pkl_path, "rb") as f:
            self.raw_dump = pickle.load(f)

        params_raw = self.raw_dump.get("Parameters", {})
        self.params = params_raw.get("general", params_raw)
        self.experimental = params_raw.get("experimental", {})

        tt = self.params.get("timetagging", {})
        c_raw = tt.get("channels", [1, 2, 3, 4, 5, 6])
        m_raw = tt.get("mode_on_channel", ["H3T", "H3R", "H4T", "H4R", "H5T", "H5R"])
        channels = list(c_raw.values()) if isinstance(c_raw, dict) else c_raw
        modes = list(m_raw.values()) if isinstance(m_raw, dict) else m_raw
        self.channel_map = dict(zip([int(c) for c in channels], modes))

        self.tau_res_ps = float(tt.get("binwidth_ps", 100))
        laser = self.params.get("laser", {})
        self.rep_rate_hz = float(laser.get("rep_rate_hz") or DEFAULT_REP_RATE_HZ)
        self.rep_period_ns = (1.0 / self.rep_rate_hz) * 1e9

        try:
            self.duration = float(self.experimental.get("duration", 60 * 1e12))
        except (KeyError, TypeError):
            self.duration = 60.0 * 1e12

        self.data = self.raw_dump.get("data", self.raw_dump)
        self.has_physical_histograms = bool(self.data.get("correlations_physical"))
        self.has_virtual_histograms = bool(self.data.get("correlations_virtual"))
        self._t0_cache = {}
        self._parse_metadata()

    # =====================================================================
    # Metadata parsing (sample / filter / polariser / angle / power / date)
    # =====================================================================

    def _parse_metadata(self):
        stem = self.pkl_path.stem
        laser = self.params.get("laser", {})
        tt = self.params.get("timetagging", {})

        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
        self.date = (date_match.group(1) if date_match
                     else self.params.get("date", "Unknown"))

        self.power_mw = self.experimental.get("laser_power")
        if self.power_mw is None:
            m = re.search(r"(\d+(?:\.\d+)?)mW", stem)
            self.power_mw = float(m.group(1)) if m else None
        self.power = f"{self.power_mw:g} mW" if self.power_mw is not None else None

        self.material = self.params.get("material", "Unknown")
        self.wavelength_nm = laser.get("wavelength_nm")
        cw = tt.get("coincidence_window", 0)
        self.coincidence_window_ns = (cw / 1e3) if cw else None
        self.binwidth_ps = float(tt.get("binwidth_ps", self.tau_res_ps))
        self.duration_s = self.duration * 1e-12
        self.rotation_stage = self.experimental.get("rotation_stage")

        # Preferred source of the optical configuration: the typed `configuration`
        # block written at acquisition time. Fall back to parsing the `material`
        # string / run-folder name for older data that encoded it there.
        self.config = self.params.get("configuration", {}) or {}
        self.sample, fil_fallback, pol_fallback = self._parse_conditions(
            self.material, self.run_dir.name)
        self.filters_by_harmonic = {}
        self.filter_layout = None

        if self.config:
            self.polarizers, self.polarization = self._polarizers_from_config(self.config)
            (self.filters, self.filter_label, self.filters_by_harmonic,
             self.filter_layout) = self._filters_from_config(self.config)
        else:
            self.polarizers = pol_fallback
            self.polarization = "+".join(pol_fallback) if pol_fallback else None
            self.filters = fil_fallback
            self.filter_label = " + ".join(fil_fallback) if fil_fallback else None

    @staticmethod
    def _parse_conditions(material, dirname):
        """Best-effort extraction of (sample, filters, polarizers) from the rich
        `material` string, falling back to the run-folder name.

        * sample      : leading alpha-numeric token, e.g. 'GaAs100'.
        * filters     : bandpass tokens 'CCC-BB' with bandwidth BB <= 99 (real optical
                        filters), e.g. '700-40', '425-25' -> '700/40 nm', '425/25 nm'.
                        Rotation angles like '337-335' (BB >= 100) are NOT treated as
                        filters.
        * polarizers  : 'P1', 'P3', ... or 'No pol.' when an explicit 'noP'/'no P'.
        """
        raw = material if material and material != "Unknown" else (dirname or "")
        # Underscores/dashes-as-separators defeat \b boundaries, so normalise the
        # field separators to spaces first (keeping the 'CCC-BB' filter dash).
        text = re.sub(r"[_]+", " ", raw)

        sample = None
        ms = re.match(r"\s*([A-Za-z][A-Za-z0-9]*)", text)
        if ms:
            sample = ms.group(1)

        filters = []
        for c, b in re.findall(r"\b(\d{3,4})-(\d{1,3})\b", text):
            if int(b) <= 99:  # bandwidth; angles like 337-335 (>=100) are not filters
                lab = f"{c}/{b} nm"
                if lab not in filters:
                    filters.append(lab)

        polarizers = []
        if re.search(r"\bno\s*P(ol|\d)", text, re.IGNORECASE):
            pass  # explicit "no pol" -> empty list (rendered as 'No pol.')
        else:
            for n in re.findall(r"\bP(\d+)\b", text):
                p = f"P{n}"
                if p not in polarizers:
                    polarizers.append(p)
        return sample, filters, polarizers

    # ---- structured `configuration` block (preferred) -----------------------

    @staticmethod
    def _polarizers_from_config(config):
        """(polarizers, label) from the boolean P# flags in ``configuration``.

        e.g. {'P1': True, 'P3': False} -> (['P1'], 'P1'); both True -> 'P1+P3';
        all present and False -> ([], 'No pol.').
        """
        flags = {k: v for k, v in config.items() if re.fullmatch(r"P\d+", str(k))}
        if not flags:
            return [], None
        active = [k for k in sorted(flags, key=lambda s: int(s[1:])) if flags[k]]
        return active, ("+".join(active) if active else "No pol.")

    @staticmethod
    def _norm_filter(value):
        """Normalise a filter token: '700-40' -> '700/40 nm'; pass through anything
        that already looks formatted."""
        s = str(value).strip()
        m = re.fullmatch(r"(\d{3,4})-(\d{1,3})", s)
        return f"{m.group(1)}/{m.group(2)} nm" if m else s

    @classmethod
    def _filters_from_config(cls, config):
        """(filters_list, label, by_harmonic, layout) from ``configuration``.

        ``filters`` may be a per-harmonic dict {'H3': '700-40', 'H5': '425-25'},
        a list, or a single string. ``filter_layout`` ('per-channel' | 'single')
        is appended to the label when present.
        """
        raw = config.get("filters")
        layout = config.get("filter_layout")
        by_harmonic = {}
        if isinstance(raw, dict):
            by_harmonic = {h: cls._norm_filter(v) for h, v in raw.items()}
            uniq = list(dict.fromkeys(by_harmonic.values()))
            if len(uniq) == 1:
                label = uniq[0]
            else:
                label = ", ".join(f"{h} {v}" for h, v in by_harmonic.items())
            filters = list(by_harmonic.values())
        elif isinstance(raw, (list, tuple)):
            filters = [cls._norm_filter(v) for v in raw]
            label = " + ".join(filters)
        elif raw:
            label = cls._norm_filter(raw)
            filters = [label]
        else:
            return [], None, {}, layout
        if layout:
            label = f"{label} ({layout})"
        return filters, label, by_harmonic, layout

    # =====================================================================
    # Title / legend helpers  (one consistent place; configurable verbosity)
    # =====================================================================

    def _cond_bits(self):
        """The few physical-configuration fields worth showing: filter, polariser,
        power (only those that are known)."""
        bits = []
        if self.filter_label:
            bits.append(rf"Filter: {self.filter_label}")
        if self.polarization:
            bits.append(self.polarization)
        if self.power_mw is not None:
            bits.append(rf"$P = {self.power_mw:g}$ mW")
        return bits

    def title(self, base):
        """Main figure title: `base` only. Kept deliberately short."""
        return base

    def subtitle(self, details=False):
        """Metadata line under the title. By default: sample, filter, polariser,
        power, date. With `details=True` add laser wavelength, rep-rate, binning,
        coincidence window and integration time."""
        bits = []
        if self.sample:
            bits.append(rf"Sample: {self.material}" if details
                        else rf"Sample: {self.sample}")
        bits += self._cond_bits()
        if self.date and self.date != "Unknown":
            bits.append(self.date)
        if details:
            extra = []
            if self.wavelength_nm:
                extra.append(rf"$\lambda_L = {self.wavelength_nm:g}$ nm")
            if self.rep_rate_hz:
                extra.append(rf"$f_{{rep}} = {self.rep_rate_hz / 1e6:.2f}$ MHz")
            extra.append(rf"bin $= {self.binwidth_ps:g}$ ps")
            if self.coincidence_window_ns:
                extra.append(rf"coinc. $= {self.coincidence_window_ns:g}$ ns")
            extra.append(rf"$T_{{acq}} = {self.duration_s:g}$ s")
            if self.rotation_stage is not None:
                extra.append(rf"stage $= {self.rotation_stage:g}^\circ$")
            return "  |  ".join(bits) + "\n" + "  |  ".join(extra)
        return "  |  ".join(bits)

    def legend_tag(self):
        """Compact descriptor for multi-run legends, one token per condition
        (filter | polariser | angle | power). Falls back to the run-folder name
        when nothing distinguishing is known."""
        bits = []
        if self.filter_label:
            bits.append(self.filter_label)
        if self.polarization:
            bits.append(self.polarization)
        if self.rotation_stage is not None:
            bits.append(rf"{self.rotation_stage:g}$^\circ$")
        if self.power_mw is not None:
            bits.append(f"{self.power_mw:g} mW")
        return " | ".join(bits) if bits else self.run_dir.name

    # Backwards-compatible aliases used by older code/notebooks.
    def short_tag(self):
        return self.legend_tag()

    def acquisition_essentials(self):
        return self.subtitle(details=False)

    def acquisition_summary(self):
        return self.subtitle(details=True).split("\n")[0]

    def acquisition_details(self):
        full = self.subtitle(details=True)
        return full.split("\n")[1] if "\n" in full else ""

    # =====================================================================
    # Structure inspection
    # =====================================================================

    def describe(self, max_depth=6, max_list_preview=4):
        """Pretty-print the structure of the loaded pickle."""
        print(f"PKL FILE : {self.pkl_path.name}")
        print(f"FOLDER   : {self.dir_path.name}")
        print(f"top-level keys: {list(self.raw_dump.keys())}")
        print("=" * 78)
        self._describe_node(self.raw_dump, "", 0, max_depth, max_list_preview)

    @staticmethod
    def _describe_node(node, prefix, depth, max_depth, max_list_preview):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}/  (dict, {len(v)} keys)")
                if depth + 1 < max_depth:
                    HBTMeasurement._describe_node(
                        v, prefix + "    ", depth + 1, max_depth, max_list_preview)
                else:
                    print(f"{prefix}    ... ({list(v.keys())})")
            elif isinstance(v, (list, tuple, np.ndarray)):
                seq = list(v)
                preview = ", ".join(f"{x}" for x in seq[:max_list_preview])
                more = ", ..." if len(seq) > max_list_preview else ""
                print(f"{prefix}{k}: {type(v).__name__}[{len(seq)}]  ({preview}{more})")
            else:
                print(f"{prefix}{k}: {type(v).__name__} = {v!r}")

    # =====================================================================
    # Channel helpers
    # =====================================================================

    def get_ch(self, physical_name):
        inv_map = {v: k for k, v in self.channel_map.items()}
        if physical_name not in inv_map:
            raise KeyError(f"Detector '{physical_name}' not found.")
        return inv_map[physical_name]

    def has_detector(self, physical_name):
        return physical_name in self.channel_map.values()

    def _get_physical_name(self, c1, c2):
        return rf"{self.channel_map.get(c1, f'Ch{c1}')} \& {self.channel_map.get(c2, f'Ch{c2}')}"

    def _get_virtual_name(self, ch):
        name = self.channel_map.get(ch, "")
        if name.endswith("T") or name.endswith("R"):
            return name[:-1]
        return name

    # =====================================================================
    # Extraction (explicit physical vs virtual, no silent fallback)
    # =====================================================================

    def _get_channel_key(self, target_dict, c1, c2, is_virtual):
        if not target_dict:
            return None
        v1 = self._get_virtual_name(c1) if is_virtual else str(c1)
        v2 = self._get_virtual_name(c2) if is_virtual else str(c2)
        keys = [f"({v1},{v2})", f"({v2},{v1})", f"({v1}, {v2})", f"({v2}, {v1})"]
        return next((k for k in keys if k in target_dict), None)

    def resolve_source(self, source="auto"):
        """'auto' -> 'physical' if per-detector histograms exist, else 'virtual'."""
        if source == "auto":
            return "physical" if self.has_physical_histograms else "virtual"
        return source

    def get_correlation_trace(self, c1, c2, source="auto"):
        """Correlation HISTOGRAM as (tau_ns, counts) for the detector pair (c1, c2).

        Time bins are converted ps -> ns. For a VIRTUAL auto-correlation (Hn x Hn)
        the single non-physical self-coincidence bin at tau=0 is suppressed.
        Returns (None, None) when the requested histogram is unavailable.
        """
        used = self.resolve_source(source)
        is_virtual = (used == "virtual")

        store = "correlations_virtual" if is_virtual else "correlations_physical"
        target_dict = self.data.get(store, {})

        key = self._get_channel_key(target_dict, c1, c2, is_virtual)
        if key is None:
            return None, None

        trace = target_dict[key]
        if isinstance(trace, dict):
            x_k = next((k for k in trace if "time" in k.lower() or "bin" in k.lower()), None)
            y_k = next((k for k in trace if "count" in k.lower() or "coinc" in k.lower()), "counts")
            x = np.array(trace[x_k], dtype=float) * 1e-3
            y = np.array(trace[y_k], dtype=float)
        elif isinstance(trace, (list, tuple)):
            x = np.array(trace[0], dtype=float) * 1e-3
            y = np.array(trace[1], dtype=float)
        else:
            return None, None

        if is_virtual and self._get_virtual_name(c1) == self._get_virtual_name(c2):
            y = self._suppress_self_correlation(x, y)
        return x, y

    @staticmethod
    def _suppress_self_correlation(x, y):
        """Replace the zero-lag self-coincidence bin of an auto-correlation by the
        mean of its neighbours."""
        if x is None or len(x) < 3:
            return y
        i0 = int(np.argmin(np.abs(x)))
        if 0 < i0 < len(y) - 1:
            y = y.copy()
            y[i0] = 0.5 * (y[i0 - 1] + y[i0 + 1])
        return y

    def _get_counts(self, c, is_virtual=False):
        store = "counts_virtual" if is_virtual else "counts_physical"
        target_dict = self.data.get(
            store, self.data.get("Heralded_Countrate" if is_virtual else "Countrate", {}))
        key = self._get_virtual_name(c) if is_virtual else str(c)
        val = target_dict.get(key)
        if val is None:
            return 0.0
        return float(val[1]) if isinstance(val, (list, tuple)) else float(val)

    def harmonic_intensity(self, n, kind="countrate"):
        """Mean intensity of harmonic n = H{n} (merged T + R)."""
        vname = f"H{n}"
        store = "countrates_virtual" if kind == "countrate" else "counts_virtual"
        d = self.data.get(store, {})
        if vname in d:
            v = d[vname]
            return float(v[1]) if isinstance(v, (list, tuple)) else float(v)

        pstore = "countrates_physical" if kind == "countrate" else "counts_physical"
        pd_ = self.data.get(pstore, {})
        total, found = 0.0, False
        for ch in self.channel_map:
            if self._get_virtual_name(ch) == vname:
                val = pd_.get(str(ch))
                if val is not None:
                    total += float(val[1]) if isinstance(val, (list, tuple)) else float(val)
                    found = True
        return total if found else np.nan

    def channel_intensity(self, ch, kind="countrate"):
        store = "countrates_physical" if kind == "countrate" else "counts_physical"
        d = self.data.get(store, {})
        val = d.get(str(ch))
        if val is None:
            return np.nan
        return float(val[1]) if isinstance(val, (list, tuple)) else float(val)

    def pump_intensity(self):
        return self.power_mw

    def _get_twofold_coincidence(self, c1, c2, is_virtual=False):
        store = ("coincidences_twofold_virtual" if is_virtual
                 else "coincidences_twofold_physical")
        target_dict = self.data.get(store, {})
        if not target_dict:
            return None
        a = self._get_virtual_name(c1) if is_virtual else str(c1)
        b = self._get_virtual_name(c2) if is_virtual else str(c2)
        for k in (f"({a},{b})", f"({b},{a})", f"({a}, {b})", f"({b}, {a})"):
            if k in target_dict:
                val = target_dict[k]
                return float(val[1]) if isinstance(val, (list, tuple)) else float(val)
        return None

    # =====================================================================
    # T0 tracking
    # =====================================================================

    def calculate_t0_shift(self, c1, c2, source="auto"):
        """Centroid (ns) of the central correlation peak; absorbs residual delay."""
        key = f"t0_{c1}-{c2}_{self.resolve_source(source)}"
        if key in self._t0_cache:
            return self._t0_cache[key]

        x, y = self.get_correlation_trace(c1, c2, source=source)
        if x is None or len(y) == 0:
            return 0.0

        approx_t0 = x[np.argmax(y)]
        tight_mask = (x >= approx_t0 - 1.0) & (x <= approx_t0 + 1.0)
        x_tight, y_tight = x[tight_mask], y[tight_mask]
        res = approx_t0 if np.sum(y_tight) == 0 else np.sum(x_tight * y_tight) / np.sum(y_tight)
        self._t0_cache[key] = res
        return res

    # =====================================================================
    # Quantum calculations
    # =====================================================================

    def compute_g2_delay(self, c1, c2, tau_in_ns, num_side_peaks=3, source="auto"):
        """Pulsed g^(2)(0) from the histogram peak-area ratio (central / mean side)."""
        x, y = self.get_correlation_trace(c1, c2, source=source)
        if x is None:
            return np.nan

        rep_period_ns = self.rep_period_ns
        t0_shift = self.calculate_t0_shift(c1, c2, source=source)

        central_counts = np.sum(y[(x >= t0_shift - tau_in_ns / 2) & (x <= t0_shift + tau_in_ns / 2)])
        total_side, valid = 0, 0
        for i in range(1, num_side_peaks + 1):
            for sign in (-1, 1):
                center = t0_shift + sign * (i * rep_period_ns)
                total_side += np.sum(y[(x >= center - tau_in_ns / 2) & (x <= center + tau_in_ns / 2)])
                valid += 1

        if valid == 0 or total_side == 0:
            return np.nan
        return central_counts / (total_side / valid)

    def compute_g2_direct(self, c1, c2, tau_in_ns=None):
        """PHYSICAL g^(2)(0) from scalar two-fold coincidences and singles:
        g2 = N_pulse * N_12 / (N_1 * N_2). Artefact-free, primary observable."""
        n_12 = self._get_twofold_coincidence(c1, c2, is_virtual=False)
        if n_12 is None:
            return np.nan
        n_1, n_2 = self._get_counts(c1, is_virtual=False), self._get_counts(c2, is_virtual=False)
        if n_1 * n_2 == 0:
            return np.nan
        n_pulse = (self.duration * 1e-12) * self.rep_rate_hz
        return (n_pulse * n_12) / (n_1 * n_2)

    def g2(self, c1, c2, tau_in_ns=4.0, method="delay", source="auto"):
        """Unified g^(2)(0): method in {'delay', 'direct', 'heralded'}."""
        if method == "delay":
            return self.compute_g2_delay(c1, c2, tau_in_ns, source=source)
        if method == "heralded":
            return self.compute_g2_heralded(c1, c2, tau_in_ns)
        return self.compute_g2_direct(c1, c2, tau_in_ns)

    def compute_g2_heralded(self, c1, c2, tau_in_ns):
        """Heralded g^(2) on the merged-harmonic (virtual) channels."""
        x, y = self.get_correlation_trace(c1, c2, source="virtual")
        if x is None:
            return np.nan
        t0_shift = self.calculate_t0_shift(c1, c2, source="virtual")
        n_12 = float(np.sum(y[(x >= t0_shift - tau_in_ns / 2) & (x <= t0_shift + tau_in_ns / 2)]))
        n_1, n_2 = self._get_counts(c1, is_virtual=True), self._get_counts(c2, is_virtual=True)
        if n_1 * n_2 == 0:
            return np.nan
        n_pulse = (self.duration * 1e-12) * self.rep_rate_hz
        return (n_pulse * n_12) / (n_1 * n_2)

    def compute_R_parameter(self, g2_cross, g2_auto_1, g2_auto_2):
        if np.isnan(g2_auto_1) or np.isnan(g2_auto_2) or g2_auto_1 <= 0 or g2_auto_2 <= 0:
            return np.nan
        return (g2_cross ** 2) / (g2_auto_1 * g2_auto_2)

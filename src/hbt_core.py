"""
=============================================================================
HBT Measurement Core Engine
=============================================================================

Data extraction and processing pipeline designed for analyzing second-order photon correlations, 
tailored for High-Harmonic Generation (HHG) and non-linear optics experiments.

This module handles both standard (physical) and heralded (virtual) datasets extracted from time-tagging hardware, 
ensuring accurate computation of quantum optical metrics such as g^(2)(tau) and the Cauchy-Schwarz violation R parameter.

Data model (important)
----------------------
For the `g2_heralded_virtual` acquisition format, the pickle stores:
* PHYSICAL data as scalars only: `counts_physical` / `countrates_physical` (per detector 1..6) and
  `coincidences_twofold_physical` (integrated two-fold coincidences for all 15 detector pairs).
* VIRTUAL data (per merged harmonic Hn = T+R) as scalars AND as full correlation histograms
  (`correlations_virtual`). No physical per-detector histogram is recorded.

Consequently:
* The physical g^(2)(0) is computed from the scalar physical coincidences (`compute_g2_direct`).
  This is the artifact-free, physically meaningful observable and is the default.
* Histogram-based methods (`compute_g2_delay`, coherence traces) can only read the VIRTUAL
  histograms, so they are explicitly virtual. There is NO silent physical<->virtual fallback.

Key Features
------------
* Automated T0 Tracking: Dynamically calculates global T0 shifts to automatically compensate for large, arbitrary hardware delays.
* Physical-first processing: g^(2) and R default to the physical scalar coincidences; virtual histograms are used only where no physical equivalent exists, and are never relabelled as physical.
* Self-correlation artifact removal: zero-lag self-coincidence spikes in virtual auto-correlation histograms (Hn x Hn) are suppressed.

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquée (LOA), École Polytechnique
Date: 15/06/2026
"""

import re
import pickle
import json
from pathlib import Path
import numpy as np

class HBTMeasurement:
    def __init__(self, pkl_filepath):
        self.pkl_path = Path(pkl_filepath)
        self.dir_path = self.pkl_path.parent
        
        with open(self.pkl_path, 'rb') as f:
            self.raw_dump = pickle.load(f)
            
        json_path = self.dir_path / "general_parameters.json"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                self.params = json.load(f)
        else:
            print(f"Warning: Metadata JSON missing for {self.pkl_path.stem}. Using pickle defaults.")
            self.params = self.raw_dump.get('Parameters', {})

        tt = self.params.get('timetagging', {})
        c_raw = tt.get('channels', [1, 2, 3, 4, 5, 6]) 
        m_raw = tt.get('mode_on_channel', ['H3T', 'H3R', 'H4T', 'H4R', 'H5T', 'H5R'])
        channels = list(c_raw.values()) if isinstance(c_raw, dict) else c_raw
        modes = list(m_raw.values()) if isinstance(m_raw, dict) else m_raw
        self.channel_map = dict(zip([int(c) for c in channels], modes))
        
        self.tau_res_ps = float(tt.get('binwidth_ps', 100))
        self.rep_rate_hz = float(self.params.get('laser', {}).get('rep_rate_hz', 21e6))
        # The laser repetition period is known exactly from the metadata, so there is no need
        # to estimate it from the histogram peak spacing.
        self.rep_period_ns = (1.0 / self.rep_rate_hz) * 1e9
        
        try:
            self.duration = float(self.raw_dump.get('Parameters', {}).get('experimental', {}).get('duration', 60*1e12))
        except KeyError:
            self.duration = 60.0 * 1e12 
            
        self.data = self.raw_dump.get('data', self.raw_dump)

        self._t0_cache = {}
        self._parse_filename_metadata()


    def _parse_filename_metadata(self):
        stem = self.pkl_path.stem
        exp = self.raw_dump.get('Parameters', {}).get('experimental', {})
        laser = self.params.get('laser', {})
        tt = self.params.get('timetagging', {})

        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', stem)
        self.date = date_match.group(1) if date_match else self.params.get('date', "Unknown Date")

        # Power: prefer the numeric value stored in the pickle, fall back to the filename token.
        self.power_mw = exp.get('laser_power')
        if self.power_mw is None:
            m = re.search(r'(\d+(?:\.\d+)?)mW', stem)
            self.power_mw = float(m.group(1)) if m else None
        self.power = f"{self.power_mw:g} mW" if self.power_mw is not None else "Unknown Power"

        self.material = self.params.get('material', 'Unknown')
        self.wavelength_nm = laser.get('wavelength_nm')
        self.coincidence_window_ns = tt.get('coincidence_window', 0) / 1e3 or None
        self.binwidth_ps = float(tt.get('binwidth_ps', self.tau_res_ps))
        self.duration_s = self.duration * 1e-12
        self.rotation_stage = exp.get('rotation_stage')

        self.filter_label, self.polarization = self._parse_run_conditions(self.dir_path.name)

    @staticmethod
    def _parse_run_conditions(dirname):
        """Extract the bandpass filter and polarisation configuration from the run folder name,
        e.g. '700-10 no P1 2026-06-12_...' -> ('700/10 nm', 'No pol.').
        Returns (None, None) for the corresponding field when it cannot be identified."""
        filter_label = None
        m = re.match(r'\s*(\d{3,4})-(\d{1,3})\b', dirname)
        if m:
            filter_label = f"{m.group(1)}/{m.group(2)} nm"

        polarization = None
        p = re.search(r'\b(no\s*)?P(\d+)\b', dirname, re.IGNORECASE)
        if p:
            polarization = "No pol." if p.group(1) else f"Pol. P{p.group(2)}"
        return filter_label, polarization

    # ---------------- Human-readable metadata for plot titles / legends ----------------

    def acquisition_essentials(self):
        """The few fields worth showing by default: sample, power, filter, polarisation."""
        parts = []
        if self.material and self.material != 'Unknown':
            parts.append(rf"Sample: {self.material}")
        if self.power_mw is not None:
            parts.append(rf"$P = {self.power_mw:g}$ mW")
        if self.filter_label:
            parts.append(rf"Filter: {self.filter_label}")
        if self.polarization:
            parts.append(self.polarization)
        return "  |  ".join(parts)

    def acquisition_summary(self):
        """One-line physical configuration shared by a run (sample, laser, filter, polarisation)."""
        parts = []
        if self.material and self.material != 'Unknown':
            parts.append(rf"Sample: {self.material}")
        if self.wavelength_nm:
            parts.append(rf"$\lambda_L = {self.wavelength_nm:g}$ nm")
        if self.power_mw is not None:
            parts.append(rf"$P = {self.power_mw:g}$ mW")
        if self.rep_rate_hz:
            parts.append(rf"$f_{{rep}} = {self.rep_rate_hz / 1e6:.2f}$ MHz")
        if self.filter_label:
            parts.append(rf"Filter: {self.filter_label}")
        if self.polarization:
            parts.append(self.polarization)
        return "  |  ".join(parts)

    def acquisition_details(self):
        """One-line acquisition settings (binning, coincidence window, integration time)."""
        parts = []
        parts.append(rf"bin $= {self.binwidth_ps:g}$ ps")
        if self.coincidence_window_ns:
            parts.append(rf"coinc. window $= {self.coincidence_window_ns:g}$ ns")
        parts.append(rf"$T_{{acq}} = {self.duration_s:g}$ s")
        if self.rotation_stage is not None:
            parts.append(rf"stage $= {self.rotation_stage:g}$")
        parts.append(rf"{self.date}")
        return "  |  ".join(parts)

    def short_tag(self):
        """Compact dataset descriptor for multi-run legends (filter / polarisation / power)."""
        bits = [b for b in (self.filter_label, self.polarization,
                            f"{self.power_mw:g} mW" if self.power_mw is not None else None) if b]
        return " | ".join(bits) if bits else self.pkl_path.stem

    def describe(self, max_depth=6, max_list_preview=4):
        """Pretty-print the structure of the loaded pickle: every key with its type, the length
        of arrays/lists (with a short preview) and scalar values. Handy to discover exactly which
        keys hold the physical vs virtual data before processing."""
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
                    HBTMeasurement._describe_node(v, prefix + "    ", depth + 1, max_depth, max_list_preview)
                else:
                    print(f"{prefix}    ... ({list(v.keys())})")
            elif isinstance(v, (list, tuple, np.ndarray)):
                seq = list(v)
                preview = ", ".join(f"{x}" for x in seq[:max_list_preview])
                more = ", ..." if len(seq) > max_list_preview else ""
                print(f"{prefix}{k}: {type(v).__name__}[{len(seq)}]  ({preview}{more})")
            else:
                print(f"{prefix}{k}: {type(v).__name__} = {v!r}")


    def get_ch(self, physical_name):
        inv_map = {v: k for k, v in self.channel_map.items()}
        if physical_name not in inv_map:
            raise KeyError(f"Detector '{physical_name}' not found.")
        return inv_map[physical_name]


    def _get_physical_name(self, c1, c2):
        return rf"{self.channel_map.get(c1, f'Ch{c1}')} \& {self.channel_map.get(c2, f'Ch{c2}')}"


    def _get_virtual_name(self, ch):
        name = self.channel_map.get(ch, "")
        if name.endswith('T') or name.endswith('R'): return name[:-1]
        return name



    # ---------------- Extraction (explicit physical vs virtual, no silent fallback) ----------------

    def _get_channel_key(self, target_dict, c1, c2, is_virtual):
        if not target_dict: return None
        v1 = self._get_virtual_name(c1) if is_virtual else str(c1)
        v2 = self._get_virtual_name(c2) if is_virtual else str(c2)
        
        keys = [f"({v1},{v2})", f"({v2},{v1})", f"({v1}, {v2})", f"({v2}, {v1})"]
        return next((k for k in keys if k in target_dict), None)


    def get_correlation_trace(self, c1, c2, is_virtual=True):
        """Return a correlation HISTOGRAM as (tau_ns, counts).

        NOTE: in this acquisition format only VIRTUAL (merged-harmonic) histograms are stored,
        so this method reads `correlations_virtual` by default. Physical per-detector histograms
        are not recorded; requesting `is_virtual=False` returns (None, None) rather than silently
        substituting virtual data. For a physical g^(2)(0) use `compute_g2_direct`.

        Auto-correlations (Hn x Hn) of a merged virtual channel contain a non-physical
        self-coincidence spike at exactly tau=0; that single zero-lag bin is suppressed here.
        """
        store = 'correlations_virtual' if is_virtual else 'correlations_physical'
        target_dict = self.data.get(store, {})
        key = self._get_channel_key(target_dict, c1, c2, is_virtual)
        if key is None:
            return None, None

        trace = target_dict[key]
        if isinstance(trace, dict):
            x_k = next((k for k in trace if 'time' in k.lower() or 'bin' in k.lower()), None)
            y_k = next((k for k in trace if 'count' in k.lower() or 'coinc' in k.lower()), 'counts')
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
        """Replace the single zero-lag self-coincidence bin of an auto-correlation by the
        mean of its neighbours (removes the detector self-correlation delta artifact)."""
        if x is None or len(x) < 3:
            return y
        i0 = int(np.argmin(np.abs(x)))
        if 0 < i0 < len(y) - 1:
            y = y.copy()
            y[i0] = 0.5 * (y[i0 - 1] + y[i0 + 1])
        return y


    def _get_counts(self, c, is_virtual=False):
        """Total singles for one channel. Physical -> per-detector; virtual -> merged harmonic.
        No physical<->virtual fallback (the two are not interchangeable)."""
        store = 'counts_virtual' if is_virtual else 'counts_physical'
        target_dict = self.data.get(store, self.data.get('Heralded_Countrate' if is_virtual else 'Countrate', {}))
        key = self._get_virtual_name(c) if is_virtual else str(c)
        val = target_dict.get(key)
        if val is None:
            return 0.0
        return float(val[1]) if isinstance(val, (list, tuple)) else float(val)


    def _get_twofold_coincidence(self, c1, c2, is_virtual=False):
        """Integrated two-fold coincidence count for a channel pair (scalar).
        Physical -> `coincidences_twofold_physical` (genuine cross-detector coincidences);
        virtual  -> `coincidences_twofold_virtual` (merged harmonics)."""
        store = 'coincidences_twofold_virtual' if is_virtual else 'coincidences_twofold_physical'
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




    # ---------------- T0 Tracking ----------------

    def calculate_t0_shift(self, c1, c2, rep_period_ns=None):
        """Locate the central peak of the (virtual) histogram and return its centroid in ns.

        This absorbs the residual electronic/optical delay so the central and side peaks can be
        windowed correctly. The repetition period itself is known from the metadata
        (`self.rep_period_ns`) and is no longer estimated from the data.
        """
        key = f"t0_{c1}-{c2}"
        if key in self._t0_cache: return self._t0_cache[key]
            
        x, y = self.get_correlation_trace(c1, c2)
        if x is None or len(y) == 0: return 0.0
        
        approx_t0 = x[np.argmax(y)]
        
        tight_mask = (x >= approx_t0 - 1.0) & (x <= approx_t0 + 1.0)
        x_tight, y_tight = x[tight_mask], y[tight_mask]
        
        res = approx_t0 if np.sum(y_tight) == 0 else np.sum(x_tight * y_tight) / np.sum(y_tight)
        self._t0_cache[key] = res
        return res




    # ---------------- Quantum Calculations ----------------

    def compute_g2_delay(self, c1, c2, tau_in_ns, num_side_peaks=3):
        """Pulsed g^(2)(0) from the histogram peak-area ratio (central / mean side peak).

        Histograms only exist for virtual (merged-harmonic) channels in this format, so this
        is intrinsically a VIRTUAL estimate. The zero-lag self-correlation artifact of auto
        pairs is already suppressed in `get_correlation_trace`.
        """
        x, y = self.get_correlation_trace(c1, c2, is_virtual=True)
        if x is None: return np.nan

        rep_period_ns = self.rep_period_ns
        t0_shift = self.calculate_t0_shift(c1, c2)
        
        central_counts = np.sum(y[(x >= t0_shift - tau_in_ns/2) & (x <= t0_shift + tau_in_ns/2)])
        total_side, valid = 0, 0
        
        for i in range(1, num_side_peaks + 1):
            for sign in [-1, 1]:
                center = t0_shift + sign * (i * rep_period_ns)
                total_side += np.sum(y[(x >= center - tau_in_ns/2) & (x <= center + tau_in_ns/2)])
                valid += 1

        if valid == 0 or total_side == 0: return np.nan
        return central_counts / (total_side / valid)


    def compute_g2_direct(self, c1, c2, tau_in_ns=None):
        """PHYSICAL g^(2)(0) from the scalar two-fold coincidences and singles.

            g2 = N_pulse * N_12 / (N_1 * N_2)

        N_12 is the integrated physical coincidence count for the genuine detector pair
        (`coincidences_twofold_physical`), N_1/N_2 the per-detector singles, and N_pulse the
        number of laser pulses over the acquisition. This is artifact-free and is the primary
        physical observable. `tau_in_ns` is accepted for API symmetry but unused: the coincidence
        window is fixed at acquisition time, so the value is independent of the integration sweep.
        """
        n_12 = self._get_twofold_coincidence(c1, c2, is_virtual=False)
        if n_12 is None:
            return np.nan
        n_1, n_2 = self._get_counts(c1, is_virtual=False), self._get_counts(c2, is_virtual=False)
        if n_1 * n_2 == 0:
            return np.nan
        n_pulse = (self.duration * 1e-12) * self.rep_rate_hz
        return (n_pulse * n_12) / (n_1 * n_2)


    def compute_g2_heralded(self, c1, c2, tau_in_ns):
        x, y = self.get_correlation_trace(c1, c2, is_virtual=True)
        if x is None: return np.nan
            
        t0_shift = self.calculate_t0_shift(c1, c2)
        
        n_12 = float(np.sum(y[(x >= t0_shift - tau_in_ns/2) & (x <= t0_shift + tau_in_ns/2)]))
        n_1, n_2 = self._get_counts(c1, is_virtual=True), self._get_counts(c2, is_virtual=True)
        
        if n_1 * n_2 == 0: return np.nan
        n_pulse = (self.duration * 1e-12) * self.rep_rate_hz 
        
        return (n_pulse * n_12) / (n_1 * n_2)


    def compute_R_parameter(self, g2_cross, g2_auto_1, g2_auto_2):
        if np.isnan(g2_auto_1) or np.isnan(g2_auto_2) or g2_auto_1 <= 0 or g2_auto_2 <= 0: return np.nan
        return (g2_cross ** 2) / (g2_auto_1 * g2_auto_2)
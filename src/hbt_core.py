"""
=============================================================================
HBT Measurement Core Engine
=============================================================================

Data extraction and processing pipeline designed for analyzing second-order photon correlations, 
tailored for High-Harmonic Generation (HHG) and non-linear optics experiments.

This module handles both standard (physical) and heralded (virtual) datasets extracted from time-tagging hardware, 
ensuring accurate computation of quantum optical metrics such as g^(2)(tau) and the Cauchy-Schwarz violation R parameter.

Key Features
------------
* Automated T0 Tracking: Dynamically calculates global T0 shifts to automatically compensate for large, arbitrary hardware delays.
* Adaptive Data Fallbacks: Implements silent, dynamic switching to heralded (virtual) correlation arrays if physical datasets are missing.
* Robust Normalization: Applies count-rate normalization to prevent S-curve divergence and non-physical mathematical artifacts near zero.

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquée (LOA), École Polytechnique
Date: 15/06/2026
"""

import re
import pickle
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

class HBTMeasurement:
    def __init__(self, pkl_filepath):
        self.pkl_path = Path(pkl_filepath)
        self.dir_path = self.pkl_path.parent
        
        with open(self.pkl_path, 'rb') as f:
            self.raw_dump = pickle.load(f)
            
        json_path = self.dir_path / "general_parameters.json"
        
        # --- PATCH: JSON is optionnal for older datasets ---
        if json_path.exists():
            with open(json_path, 'r') as f:
                self.params = json.load(f)
        else:
            print(f"Warning: Metadata JSON missing for {self.pkl_path.stem}. Using pickle defaults.")
            self.params = self.raw_dump.get('Parameters', {}) # Fallback sur le pickle

        tt = self.params.get('timetagging', {})
        c_raw = tt.get('channels', [1, 2, 3, 4, 5, 6, 7, 8]) # Valeurs par défaut au besoin
        m_raw = tt.get('mode_on_channel', ['Sync', 'Ref', 'H3R', 'H3T', 'H4R', 'H4T', 'H5R', 'H5T'])
        channels = list(c_raw.values()) if isinstance(c_raw, dict) else c_raw
        modes = list(m_raw.values()) if isinstance(m_raw, dict) else m_raw
        self.channel_map = dict(zip([int(c) for c in channels], modes))
        
        self.tau_res_ps = float(tt.get('binwidth_ps', 100))
        self.rep_rate_hz = float(self.params.get('laser', {}).get('rep_rate_hz', 21e6))
        
        try:
            self.duration = float(self.raw_dump.get('Parameters', {}).get('experimental', {}).get('duration', 60*1e12))
        except KeyError:
            self.duration = 60.0 * 1e12 
            
        self.data = self.raw_dump.get('data', self.raw_dump)

        self._rep_cache = {}
        self._t0_cache = {}
        self._parse_filename_metadata()


    def _parse_filename_metadata(self):
        stem = self.pkl_path.stem
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', stem)
        self.date = date_match.group(1) if date_match else "Unknown Date"
        power_match = re.search(r'(\d+(?:\.\d+)?mW)', stem)
        self.power = power_match.group(1) if power_match else "Unknown Power"
        self.material = self.params.get('material', 'Unknown')


    def get_ch(self, physical_name):
        inv_map = {v: k for k, v in self.channel_map.items()}
        if physical_name not in inv_map:
            raise KeyError(f"Detector '{physical_name}' not found.")
        return inv_map[physical_name]


    def _get_physical_name(self, c1, c2):
        return f"{self.channel_map.get(c1, f'Ch{c1}')} \& {self.channel_map.get(c2, f'Ch{c2}')}"


    def _get_virtual_name(self, ch):
        name = self.channel_map.get(ch, "")
        if name.endswith('T') or name.endswith('R'): return name[:-1]
        return name



    # ---------------- Extraction Adaptative & Fallbacks ----------------

    def _get_channel_key(self, target_dict, c1, c2, is_virtual):
        if not target_dict: return None
        v1 = self._get_virtual_name(c1) if is_virtual else str(c1)
        v2 = self._get_virtual_name(c2) if is_virtual else str(c2)
        
        keys = [f"({v1},{v2})", f"({v2},{v1})", f"({v1}, {v2})", f"({v2}, {v1})"]
        return next((k for k in keys if k in target_dict), None)


    def get_correlation_trace(self, c1, c2, is_virtual=False, allow_fallback=True):
        """Get trace. If physical missing, shift to heralded automatically."""
        target_dict = self.data.get('correlations_virtual' if is_virtual else 'correlations_physical', 
                                    self.data.get('Heralded_Correlation' if is_virtual else 'Correlation', {}))
        
        key = self._get_channel_key(target_dict, c1, c2, is_virtual)
        
        if key:
            trace = target_dict[key]
        elif allow_fallback:
            return self.get_correlation_trace(c1, c2, not is_virtual, allow_fallback=False)
        else:
            return None, None
            
        if isinstance(trace, dict):
            x_k = next((k for k in trace if 'time' in k.lower() or 'bin' in k.lower()), None)
            y_k = next((k for k in trace if 'count' in k.lower() or 'coinc' in k.lower()), 'counts')
            return np.array(trace[x_k]) * 1e-3, np.array(trace[y_k])
        elif isinstance(trace, (list, tuple)):
            return np.array(trace[0]) * 1e-3, np.array(trace[1])
            
        return None, None


    def _get_counts(self, c, is_virtual=False, allow_fallback=True):
        """Get total counts with same fallback error method"""
        target_dict = self.data.get('counts_virtual' if is_virtual else 'counts_physical', 
                                    self.data.get('Heralded_Countrate' if is_virtual else 'Countrate', {}))
        
        key = self._get_virtual_name(c) if is_virtual else str(c)
        val = target_dict.get(key)
        
        if val is None and allow_fallback:
            return self._get_counts(c, not is_virtual, allow_fallback=False)
            
        if val is not None:
            return float(val[1]) if isinstance(val, (list, tuple)) else float(val)
        return 0.0




    # ---------------- Jitter & Tracking Global ----------------

    def estimate_rep_period(self, c1, c2, prominence_threshold=50):
        key = f"{c1}-{c2}"
        if key in self._rep_cache: return self._rep_cache[key]
            
        x, y = self.get_correlation_trace(c1, c2)
        if x is None or len(y) == 0: return None
        
        min_dist = int(40 / (self.tau_res_ps * 1e-3))
        peaks, _ = find_peaks(y, prominence=prominence_threshold, distance=min_dist)
        if len(peaks) < 2: return None
        
        res = np.median(np.diff(x[peaks]))
        self._rep_cache[key] = res
        return res


    def calculate_t0_shift(self, c1, c2, rep_period_ns):
        """Find higest peak and renormalize x-axis to cover electronical delays"""
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
        x, y = self.get_correlation_trace(c1, c2, is_virtual=False)
        if x is None: return np.nan

        empirical_rep = self.estimate_rep_period(c1, c2)
        rep_period_ns = empirical_rep if empirical_rep is not None else (1 / self.rep_rate_hz * 1e9)
        t0_shift = self.calculate_t0_shift(c1, c2, rep_period_ns)
        
        central_counts = np.sum(y[(x >= t0_shift - tau_in_ns/2) & (x <= t0_shift + tau_in_ns/2)])
        total_side, valid = 0, 0
        
        for i in range(1, num_side_peaks + 1):
            for sign in [-1, 1]:
                center = t0_shift + sign * (i * rep_period_ns)
                total_side += np.sum(y[(x >= center - tau_in_ns/2) & (x <= center + tau_in_ns/2)])
                valid += 1

        if valid == 0 or total_side == 0: return np.nan
        return central_counts / (total_side / valid)


    def compute_g2_direct(self, c1, c2, tau_in_ns):
        x, y = self.get_correlation_trace(c1, c2, is_virtual=False)
        if x is None: return np.nan
        
        empirical_rep = self.estimate_rep_period(c1, c2)
        rep_period_ns = empirical_rep if empirical_rep is not None else (1 / self.rep_rate_hz * 1e9)
        t0_shift = self.calculate_t0_shift(c1, c2, rep_period_ns)
        
        n_12 = float(np.sum(y[(x >= t0_shift - tau_in_ns/2) & (x <= t0_shift + tau_in_ns/2)]))
        n_1, n_2 = self._get_counts(c1, is_virtual=False), self._get_counts(c2, is_virtual=False)
        
        if n_1 * n_2 == 0: return np.nan
        n_pulse = (self.duration * 1e-12) * self.rep_rate_hz
        return (n_pulse * n_12) / (n_1 * n_2)


    def compute_g2_heralded(self, c1, c2, tau_in_ns):
        x, y = self.get_correlation_trace(c1, c2, is_virtual=True)
        if x is None: return np.nan
            
        empirical_rep = self.estimate_rep_period(c1, c2)
        rep_period_ns = empirical_rep if empirical_rep is not None else (1 / self.rep_rate_hz * 1e9)
        t0_shift = self.calculate_t0_shift(c1, c2, rep_period_ns)
        
        n_12 = float(np.sum(y[(x >= t0_shift - tau_in_ns/2) & (x <= t0_shift + tau_in_ns/2)]))
        n_1, n_2 = self._get_counts(c1, is_virtual=True), self._get_counts(c2, is_virtual=True)
        
        if n_1 * n_2 == 0: return np.nan
        n_pulse = (self.duration * 1e-12) * self.rep_rate_hz 
        
        return (n_pulse * n_12) / (n_1 * n_2)


    def compute_R_parameter(self, g2_cross, g2_auto_1, g2_auto_2):
        if np.isnan(g2_auto_1) or np.isnan(g2_auto_2) or g2_auto_1 <= 0 or g2_auto_2 <= 0: return np.nan
        return (g2_cross ** 2) / (g2_auto_1 * g2_auto_2)
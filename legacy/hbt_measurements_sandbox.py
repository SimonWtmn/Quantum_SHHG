"""
Hanbury Brown and Twiss (HBT) Measurement Core Engine

This module provides the core data ingestion and parameter estimation engine for processing photon coincidence data from Hanbury Brown and Twiss interferometry setups. 
It cleanly decouples file format variations (supporting both JSON and Pickle serialization) and isolates experimental hardware mappings from downstream physical analysis.

Key Capabilities:
    * File format agnosticism (.pkl and .json).
    * Automated string token parsing to extract crystal, power, date, and run parameters.
    * Hardware channel translation to human-readable physical optical paths.
    * Statistical parameter estimation for laser repetition rates and zero-delay electronic jitter.

Author: Simon WITTMANN
Target Environment: Python 3.8+
Dependencies: numpy, scipy, matplotlib
"""

import re
import pickle
import json
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


class HBTMeasurement:
    def __init__(self, filepath):
        """
        Initializes an HBT measurement workspace from a serialized data file.
        
        Parameters:
            filepath (str): Path to the target data file (.pkl or .json)
        """
        self.filepath = filepath
        
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                self.data = pickle.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format for {filepath}. Use .pkl or .json")

        self.tau_res_ps = self.data['Parameters']['binwidth']
        self.bins = self.data['Parameters']['bins']
        self.duration = self.data['Parameters']['duration'] 

        channels = self.data['Parameters']['channels']
        modes = self.data['Parameters']['mode_on_channel']
        self.channel_map = dict(zip(channels, modes))
        self._rep_cache = {}
        self._t0_cache = {}
        
        self._parse_filename_metadata()

    def _parse_filename_metadata(self):
        """
        Parses experimental parameters directly from the filename string token using regex.
        Falls back gracefully to dictionary metadata or defaults if token matching fails.
        """
        stem = Path(self.filepath).stem
        
        # Extract Date
        date_match = re.search(r'(?:^|_)(\d{8})(?:_|$)', stem)
        if date_match:
            d = date_match.group(1)
            self.date = f"{d[:2]}/{d[2:4]}/{d[4:]}"
        else:
            self.date = self.data.get('Parameters', {}).get('general', {}).get('date', "Unknown Date")
            
        # Extract Power
        power_match = re.search(r'(\d+mW)', stem)
        self.power = power_match.group(1) if power_match else "Unknown Power"
        
        # Extract Repetition/Run Number
        num_match = re.search(r'num(\d+)', stem)
        self.run_number = f"Run {num_match.group(1)}" if num_match else "Run 1"
        
        # Extract Crystal/Material
        self.material = self.data.get('Parameters', {}).get('material')

    def _get_channel_key(self, c1, c2):
        """Resolves structural mismatch variations in saved dictionary keys."""
        c_min = min(c1, c2)
        c_max = max(c1, c2)
        keys_to_try = [
            f"({c_min}, {c_max})",   
            f"({c_min},{c_max})",    
            (c_min, c_max),          
            f"[{c_min}, {c_max}]",   
            f"[{c_min},{c_max}]"     
        ]
        for k in keys_to_try:
            if k in self.data['Correlation']:
                return k
        available_keys = list(self.data['Correlation'].keys())[:5]
        raise KeyError(f"Could not find correlation data for sorted pair ({c_min}, {c_max}). File sample keys: {available_keys}...")
    
    def _get_physical_name(self, c1, c2):
        """Translates raw hardware channel indices to physical optical path IDs."""
        name1 = self.channel_map.get(c1, f"Ch{c1}")
        name2 = self.channel_map.get(c2, f"Ch{c2}")
        return f"{name1} \& {name2}"
    
    def get_ch(self, physical_name):
        """Performs a reverse lookup to convert an optical path name back to its channel integer."""
        inv_map = {v: k for k, v in self.channel_map.items()}
        if physical_name not in inv_map:
            raise KeyError(f"Detector '{physical_name}' not found. Available options: {list(inv_map.keys())}")
        return inv_map[physical_name]



    # ---------------- Parameter Estimation Methods ----------------

    def estimate_rep_period(self, c1, c2, prominence_threshold=50):
        """Estimates the laser repetition period by evaluating histogram peak intervals."""
        key = self._get_channel_key(c1, c2)
        
        # Check cache first
        if key in self._rep_cache:
            return self._rep_cache[key]
            
        delta_t_ns = np.array(self.data['Correlation'][key][0]) * 1e-3 
        counts_y = np.array(self.data['Correlation'][key][1])
        
        min_distance_bins = int(40 / (self.tau_res_ps * 1e-3))
        peaks, _ = find_peaks(counts_y, prominence=prominence_threshold, distance=min_distance_bins)
        
        if len(peaks) < 2:
            return None
            
        peak_times = delta_t_ns[peaks]
        result = np.median(np.diff(peak_times))
        
        # Save to cache and return
        self._rep_cache[key] = result
        return result

    def calculate_t0_shift(self, c1, c2, rep_period_ns):
        """Calculates precise electronic tracking zero-delay offset using a center of mass refinement."""
        key = self._get_channel_key(c1, c2)
        
        # Check cache first
        if key in self._t0_cache:
            return self._t0_cache[key]
            
        delta_t_ns = np.array(self.data['Correlation'][key][0]) * 1e-3
        counts_y = np.array(self.data['Correlation'][key][1])

        window_mask = (delta_t_ns >= -rep_period_ns/2) & (delta_t_ns <= rep_period_ns/2)
        x_window = delta_t_ns[window_mask]
        y_window = counts_y[window_mask]

        if len(y_window) == 0 or np.sum(y_window) == 0:
            return 0
        
        max_idx = np.argmax(y_window)
        approx_t0 = x_window[max_idx]

        tight_mask = (x_window >= approx_t0 - 1.0) & (x_window <= approx_t0 + 1.0)
        x_tight = x_window[tight_mask]
        y_tight = y_window[tight_mask]
        
        if np.sum(y_tight) == 0:
            result = approx_t0
        else:
            result = np.sum(x_tight * y_tight) / np.sum(y_tight)
            
        # Save to cache and return
        self._t0_cache[key] = result
        return result



    # ---------------- Calculation Methods ----------------

    def compute_g2_delay(self, c1, c2, tau_in_ns, laser_rep_rate_hz=None, num_side_peaks=3):
        """Calculates g^(2) via sideband normalization method."""
        key = self._get_channel_key(c1, c2)
        delta_t_ns = np.array(self.data['Correlation'][key][0]) * 1e-3
        counts_y = np.array(self.data['Correlation'][key][1])

        if laser_rep_rate_hz is not None:
            rep_period_ns = (1 / laser_rep_rate_hz) * 1e9
        else:
            rep_period_ns = self.estimate_rep_period(c1, c2)
            if rep_period_ns is None: return 0

        t0_shift = self.calculate_t0_shift(c1, c2, rep_period_ns)

        central_mask = (delta_t_ns >= t0_shift - tau_in_ns/2) & (delta_t_ns <= t0_shift + tau_in_ns/2)
        central_counts = np.sum(counts_y[central_mask])

        total_side_counts = 0
        valid_peaks = 0
        
        for i in range(1, num_side_peaks + 1):
            left_center = t0_shift - (i * rep_period_ns)
            left_mask = (delta_t_ns >= left_center - tau_in_ns/2) & (delta_t_ns <= left_center + tau_in_ns/2)
            total_side_counts += np.sum(counts_y[left_mask])
            valid_peaks += 1
            
            right_center = t0_shift + (i * rep_period_ns)
            right_mask = (delta_t_ns >= right_center - tau_in_ns/2) & (delta_t_ns <= right_center + tau_in_ns/2)
            total_side_counts += np.sum(counts_y[right_mask])
            valid_peaks += 1

        if valid_peaks == 0: return 0
        return central_counts / (total_side_counts / valid_peaks)

    def compute_g2_direct(self, c1, c2, tau_in_ns, laser_rep_rate_hz=None):
        """Calculates g^(2) via raw countrate mathematical normalization."""
        key = self._get_channel_key(c1, c2)
        delta_t_ns = np.array(self.data['Correlation'][key][0]) * 1e-3
        counts_y = np.array(self.data['Correlation'][key][1])
        
        if laser_rep_rate_hz is None:
            rep_period_ns = self.estimate_rep_period(c1, c2)
            if rep_period_ns is None: return 0
            laser_rep_rate_hz = 1 / (rep_period_ns * 1e-9)
        else:
            rep_period_ns = (1 / laser_rep_rate_hz) * 1e9

        t0_shift = self.calculate_t0_shift(c1, c2, rep_period_ns)

        central_mask = (delta_t_ns >= t0_shift - tau_in_ns/2) & (delta_t_ns <= t0_shift + tau_in_ns/2)
        n_12 = float(np.sum(counts_y[central_mask]))

        n_1 = float(self.data['Countrate'][str(c1)][1])
        n_2 = float(self.data['Countrate'][str(c2)][1])
        duration_seconds = self.duration * 1e-12 
        n_pulse = duration_seconds * laser_rep_rate_hz
        
        if n_1 * n_2 == 0: return 0
        return (n_pulse * n_12) / (n_1 * n_2)

    def compute_R_parameter(self, g2_cross, g2_auto_1, g2_auto_2):
        """Calculates the Cauchy-Schwarz R parameter to verify classical inequality violations."""
        if g2_auto_1 <= 0 or g2_auto_2 <= 0:
            return np.nan
        return (g2_cross ** 2) / (g2_auto_1 * g2_auto_2)



    # ---------------- Visualization Method ----------------

    def plot_correlation(self, c1=None, c2=None, xlim=None, show_shift=True, laser_rep_rate_hz=None, tau_in_ns=None):
        """
        Polymorphic visualization interface for coincidence statistics.
        
        If c1 and c2 are provided: Plots a high-resolution single channel-pair trace.
        If c1 and c2 are None: Automatically generates the complete 5x3 master grid.
        """
        inv_map = {v: k for k, v in self.channel_map.items()}
        
        # ------------------ BRANCH 1: MASTER GRID PLOT ------------------
        if c1 is None or c2 is None:
            cross_rows = ['TT', 'TR', 'RT', 'RR']
            cross_cols = [('3', '4'), ('3', '5'), ('4', '5')]
            auto_cols = ['3', '4', '5']
            
            # Default xlim for grid visibility if not specified
            grid_xlim = xlim
            
            fig, axes = plt.subplots(5, 3, figsize=(18, 20), dpi=300)
            
            for i in range(5):
                for j in range(3):
                    ax = axes[i, j]
                    
                    try:
                        if i == 0:
                            h = auto_cols[j]
                            ch1_idx = inv_map[f"H{h}R"]
                            ch2_idx = inv_map[f"H{h}T"]
                            title_str = f"Auto-Correlation $H_{{{h}{h}}}$ (RT)"
                        else:
                            row_type = cross_rows[i-1]
                            hA, hB = cross_cols[j]
                            ch1_idx = inv_map[f"H{hA}{row_type[0]}"]
                            ch2_idx = inv_map[f"H{hB}{row_type[1]}"]
                            title_str = f"Cross-Correlation $H_{{{hA}{hB}}}$ ({row_type})"
                    except KeyError as e:
                        print(f"Skipping subplot at layout position ({i},{j}): Missing mapping metadata key {e}.")
                        ax.set_visible(False)
                        continue
                    
                    key = self._get_channel_key(ch1_idx, ch2_idx)
                    delta_t_ns = np.array(self.data['Correlation'][key][0]) * 1e-3
                    counts_y = np.array(self.data['Correlation'][key][1]) * 1e-3
                    
                    if laser_rep_rate_hz is not None:
                        rep_period_ns = (1 / laser_rep_rate_hz) * 1e9
                    else:
                        rep_period_ns = self.estimate_rep_period(ch1_idx, ch2_idx)
                    
                    t0_shift = 0
                    if rep_period_ns is not None:
                        t0_shift = self.calculate_t0_shift(ch1_idx, ch2_idx, rep_period_ns)
                    
                    if show_shift:
                        delta_t_ns -= t0_shift
                        peak_center = 0.0
                    else:
                        peak_center = t0_shift
                    
                    ax.step(delta_t_ns, counts_y, where='mid', color='#2c3e50', lw=1.0)
                    ax.fill_between(delta_t_ns, counts_y, step='mid', color='#3498db', alpha=0.2)
                    
                    if tau_in_ns is not None:
                        window_mask = (delta_t_ns >= peak_center - tau_in_ns/2) & (delta_t_ns <= peak_center + tau_in_ns/2)
                        lbl = r"Integration Window ($\tau_{in}$)" if i == 0 and j == 0 else ""
                        ax.fill_between(delta_t_ns, counts_y, step='mid', where=window_mask, color='#e74c3c', alpha=0.5, label=lbl)
                    
                    ax.set_title(title_str, fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(grid_xlim)
                    
                    if i == 4: ax.set_xlabel(r"$\Delta t$ (ns)")
                    if j == 0: ax.set_ylabel(r"Counts $N \times 10^3$")
                    if i == 0 and j == 0 and tau_in_ns is not None:
                        ax.legend(loc='upper right')

            plt.suptitle(f"Master Coincidence Spectrum Grid\n"
                         f"Sample: {self.material} $\mid$ {self.power} $\mid$ {self.run_number} ({self.date})", fontsize=18, y=1)
            plt.tight_layout()
            plt.show()
            return fig, axes

        # ------------------ BRANCH 2: SINGLE PAIR PLOT ------------------
        else:
            key = self._get_channel_key(c1, c2)
            physical_name = self._get_physical_name(c1, c2)
            delta_t_ns = np.array(self.data['Correlation'][key][0]) * 1e-3
            counts_y = np.array(self.data['Correlation'][key][1]) * 1e-3

            if laser_rep_rate_hz is not None:
                rep_period_ns = (1 / laser_rep_rate_hz) * 1e9
            else:
                rep_period_ns = self.estimate_rep_period(c1, c2)
                
            t0_shift = 0
            if rep_period_ns is not None:
                t0_shift = self.calculate_t0_shift(c1, c2, rep_period_ns)

            if show_shift:
                delta_t_ns -= t0_shift
                peak_center = 0.0
            else:
                peak_center = t0_shift

            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            ax.step(delta_t_ns, counts_y, where='mid', color='#2c3e50', lw=1.2, label='Coincidences')
            ax.fill_between(delta_t_ns, counts_y, step='mid', color='#3498db', alpha=0.3)

            if tau_in_ns is not None:
                window_mask = (delta_t_ns >= peak_center - tau_in_ns/2) & (delta_t_ns <= peak_center + tau_in_ns/2)
                ax.fill_between(delta_t_ns, np.max(counts_y), step='mid', where=window_mask, color='#e74c3c', alpha=0.6, label=r'Integrated Counts ($\tau_{in}$)')

            ax.set_title(f"Coincidence Spectrum: {physical_name}\n"
                         f"Sample: {self.material} $\mid$ {self.power} $\mid$ {self.run_number} ({self.date})", fontsize=11)
            ax.set_xlabel(r"$\Delta t$ (ns)") 
            ax.set_ylabel(r"Counts $N \times 10^3$")

            if xlim:
                ax.set_xlim(xlim)
                
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            plt.tight_layout()
            plt.show()
            return fig, ax
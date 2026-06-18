# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:29:57 2025

@author: Karuseichyck
"""

# -*- coding: utf-8 -*-

"""
g2 Window Dependence Analysis with R Parameters

Analyzes how g2(0) and R parameters depend on integration window (WINDOW_BAND)
Compares sideband normalization vs countrate normalization methods

This version:
- Keeps plots 01–03 as before
- Replaces plot 04 with a 4×3 grid of R_ij:
    • Columns: harmonic pairs 34, 35, 45
    • Rows: detector combinations TT, TR, RT, RR

Author: Window analysis script with R parameters
Date: December 2025
"""

# import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Input file
JSON_FILE = 'merged\CdTe110_g2_heralded_v2_5000bins_timing100ps_res_66.59mW_merged.pkl'
JSON_FILE = 'CdTe110_g2_heralded_v2_5000bins_timing100ps_res_59.31mW_merged.pkl'
JSON_FILE = r'24052024_CdTe110_timing300ps_2009bins\24052024_CdTe110_g2_2009bins_timing300ps_res_85mW_num3.json'
# JSON_FILE = r'g2\g2_h3_h4_h5_pol_bef_sample\24052024_CdTe110_timing50ps_12000bins\24052024_CdTe110_g2_12000bins_timing50ps_res_75mW_num4.json'
JSON_FILE = r'singles\x_no_pol.json'
JSON_FILE = r'singles\x_ORT_pol.json'
JSON_FILE = r'singles\x_ORT_pol.json'
JSON_FILE = r'35mm_82mW.pkl'
JSON_FILE = r'35mm_25mW.pkl'

JSON_FILE = r'40mW_K.json'
# JSON_FILE = r'40mW_L.json'
# JSON_FILE = r'100mW_GaAs.json' #violation! - not reproduced

# JSON_FILE = r'Mar31_CdTe_38mW.json' 

# JSON_FILE = r'70mW_GaAs.json' 

# JSON_FILE = r'78mW_CdTe2.json' 

# JSON_FILE = r'18Dec_50mW.pkl' 

# JSON_FILE = r'100_50mW_low_H4.json' 
# JSON_FILE = r'GaAs_small_40mW.json' 
# JSON_FILE = '50mW_GaAs_170.json'
# JSON_FILE = '95mW_GaAs_170.json'
# JSON_FILE = 'CdTe_52mW_55deg.json'
# JSON_FILE = r'singles/k_par_53mW.json'
# JSON_FILE = r'singles/k_nopol_46mW.json'
# JSON_FILE = r'singles/par_38mW.json'
# JSON_FILE = r'GaAs_small_85mW.json' 
# JSON_FILE = r'GaAs_50mW.json' 
# JSON_FILE = r'g2\g2_h3_h4_h5_pol_bef_sample\24052024_CdTe110_timing50ps_12000bins\24052024_CdTe110_g2_12000bins_timing50ps_res_75mW_num4.json'
# JSON_FILE = r'24052024_CdTe110_timing300ps_2009bins\24052024_CdTe110_g2_2009bins_timing300ps_res_35mW_num3.json'
# JSON_FILE = r'singles/L_par_filters.json'
# JSON_FILE = r'singles/L_95mW.json'
# JSON_FILE = r'singles/GaAs/80mW.json'
# JSON_FILE = r'singles/GaAs/96mW.json'

# JSON_FILE = r'singles/100/77.json'
# JSON_FILE = r'singles/100/78_245.json'
# JSON_FILE = r'4Feb25mW.pkl'
# JSON_FILE = r'4Feb18mW.pkl'
# JSON_FILE = r'4Feb40mW.pkl'


# JSON_FILE = r'g2\g2_h3_h4_h5_pol_bef_sample\24052024_CdTe110_timing50ps_12000bins\24052024_CdTe110_g2_12000bins_timing50ps_res_75mW_num2.json'
# JSON_FILE = r'24052024_CdTe110_timing300ps_2009bins\24052024_CdTe110_g2_2009bins_timing300ps_res_35mW_num3.json'
# JSON_FILE = '24052024_CdTe110_g2_2009bins_timing300ps_res_95mW_num3.json'
# JSON_FILE = r'Mar31_CdTe_38mW.json' 
JSON_FILE = r'1Apr_13Mw.pkl' 
# JSON_FILE = r'1Apr_10Mw.pkl' 
# JSON_FILE = r'GaAs_small_85mW.json' 

# Laser parameters
REP_RATE = 18.66e6  # Hz - repetition rate

# Delay calibration parameters
CALIBRATION_WINDOW_NS = 3.0   # ns - window for COM calculation (±2 ns)
DELAY_SEARCH_RANGE_NS = (-10, 50)  # ns - range to search for peaks

# Analysis parameters
WINDOW_BAND_VALUES = np.arange(0.01, 6.1, 0.1)  # ns - integration window range

# Plot parameters
TIME_PLOT_RANGE = (-250, 250)  # ns - range for correlation plots
FIGSIZE_LARGE = (20, 12)
FIGSIZE_MEDIUM = (18, 6)

# Channel names
CHANNEL_NAMES = {
    1: 'H3T',
    2: 'H3R',
    3: 'H4T',
    4: 'H4R',
    5: 'H5T',
    6: 'H5R',
}

# Channel pairs to analyze for g2(1,n)
CHANNEL_PAIRS = [(1, n) for n in range(2, 7)]
CHANNEL_PAIRS = [(1,2), (3,4),(5,6), (1,3), (3,5), (1,6)]

# Autocorrelations for harmonics (needed for R parameter denominators)
# H3: (1,2), H4: (3,4), H5: (5,6)
HARMONIC_PAIRS = [
    (1, 2),  # H3: H3T-H3R
    (3, 4),  # H4: H4T-H4R
    (5, 6),  # H5: H5T-H5R
]

# Mapping harmonic index to its auto pair
HARMONIC_AUTOS = {
    3: (1, 2),  # H3
    4: (3, 4),  # H4
    5: (5, 6),  # H5
}

# Cross-harmonic detector combinations for R_ij
# 3 harmonics (3,4,5) → harmonic pairs 34,35,45
# 4 detector combinations: TT, TR, RT, RR
CROSS_COMBOS = {
    'TT': {
        '34': (1, 3),  # H3T–H4T
        '35': (1, 5),  # H3T–H5T
        '45': (3, 5),  # H4T–H5T
    },
    'TR': {
        '34': (1, 4),  # H3T–H4R
        '35': (1, 6),  # H3T–H5R
        '45': (3, 6),  # H4T–H5R
    },
    'RT': {
        '34': (2, 3),  # H3R–H4T
        '35': (2, 5),  # H3R–H5T
        '45': (4, 5),  # H4R–H5T
    },
    'RR': {
        '34': (2, 4),  # H3R–H4R
        '35': (2, 6),  # H3R–H5R
        '45': (4, 6),  # H4R–H5R
    },
}

# Flatten all cross-harmonic pairs used for R
CROSS_HARMONIC_PAIRS = sorted(
    {pair for combo_dict in CROSS_COMBOS.values() for pair in combo_dict.values()}
)

# All pairs needed for complete analysis
ALL_PAIRS = sorted(set(CHANNEL_PAIRS + HARMONIC_PAIRS + CROSS_HARMONIC_PAIRS))

# ============================================================================
# Utility Functions
# ============================================================================

def load_json_data(filename):
    """
    Load measurement data from .json or pickle-based files (.pkl / .npk),
    then extract and return the fields used by the g2/R analysis.

    Returns dict with:
        duration_s, n_pulses, countrates, total_counts, correlations
    """
    from pathlib import Path
    import pickle
    import json
    import numpy as np

    filename = Path(filename)
    print(f"Loading data from: {filename}")

    # ----------------------------
    # 1) Load raw container
    # ----------------------------
    suffix = filename.suffix.lower()

    if suffix == ".json":
        with open(filename, "r") as f:
            data = json.load(f)

    elif suffix in [".pkl", ".npk"]:
        # Your merging script writes with pickle.dump(..., protocol=HIGHEST_PROTOCOL)
        with open(filename, "rb") as f:
            data = pickle.load(f)

    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Expected .json, .pkl, or .npk"
        )

    # ----------------------------
    # 2) Duration handling
    # ----------------------------
    duration_ps = data["Parameters"]["duration"]
    duration_s = duration_ps / 1e12

    # ----------------------------
    # 3) Countrates + totals
    # ----------------------------
    countrates = {}
    total_counts = {}

    # In merge script, Countrate entries are [countrate, total_counts]
    # In json, they looked like [countrate] (but your earlier code assumed [0]).
    # This makes it robust for both.
    for ch in range(1, 7):
        entry = data["Countrate"][str(ch)]

        # entry could be [rate] or [rate, totals]
        rate = entry[0]
        countrates[ch] = rate

        if len(entry) >= 2:
            # Prefer stored totals if present (important for merged/pickled files)
            total_counts[ch] = entry[1]
        else:
            total_counts[ch] = rate * duration_s

    # ----------------------------
    # 4) Correlations (pairs only)
    # ----------------------------
    correlations = {}
    corr_block = data.get("Correlation", {})

    for key in corr_block.keys():
        # Key is usually "(1, 2)" as string, but can also be "(1,2,3)" etc.
        try:
            parsed = eval(key)
        except Exception as e:
            print(f"  Warning: could not parse correlation key '{key}': {e}")
            continue

        if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
            # Skip higher-order correlations
            continue

        ch1, ch2 = parsed

        # Expected format: [time_bins, counts]
        time_bins = np.array(corr_block[key][0])
        counts = np.array(corr_block[key][1])
        correlations[(ch1, ch2)] = (time_bins, counts)

    # ----------------------------
    # 5) Pulses
    # ----------------------------
    n_pulses = int(REP_RATE * duration_s)

    print(f"  Duration: {duration_s:.0f} s")
    print(f"  Total pulses: {n_pulses:,}")
    print(f"  Countrates: {[f'{countrates[i]/1e3:.1f}k' for i in range(1, 7)]}")
    print(f"  Loaded {len(correlations)} pair correlations")

    return {
        "duration_s": duration_s,
        "n_pulses": n_pulses,
        "countrates": countrates,
        "total_counts": total_counts,
        "correlations": correlations,
    }


def calculate_center_of_mass(time_ns, counts, peak_idx, window_ns):
    """
    Calculate center of mass of peak within a window.
    """
    peak_time = time_ns[peak_idx]
    mask = (time_ns >= peak_time - window_ns) & (time_ns <= peak_time + window_ns)

    time_window = time_ns[mask]
    counts_window = counts[mask]

    if np.sum(counts_window) == 0:
        return peak_time

    com_time = np.sum(time_window * counts_window) / np.sum(counts_window)
    return com_time


def calibrate_channel_delays_com(correlations, search_range_ns, com_window_ns=2.0):
    """
    Calibrate time delays between channel 1 and other channels
    using center of mass (COM) of the main peak.
    """
    print("\nCalibrating channel delays using center-of-mass method...")
    print(f"  COM window: ±{com_window_ns} ns")
    print(f"  Search range: {search_range_ns} ns")

    delays = {}

    # First: delays for correlations with channel 1
    for ch_ref, ch_target in CHANNEL_PAIRS:
        if (ch_ref, ch_target) not in correlations:
            print(f"  Warning: No correlation data for channels {ch_ref}-{ch_target}")
            delays[(ch_ref, ch_target)] = 0.0
            continue

        time_bins, counts = correlations[(ch_ref, ch_target)]
        time_ns = time_bins / 1000.0

        # Find peaks in search range
        mask = (time_ns >= search_range_ns[0]) & (time_ns <= search_range_ns[1])
        time_search = time_ns[mask]
        counts_search = counts[mask]

        if len(counts_search) == 0:
            delays[(ch_ref, ch_target)] = 0.0
            continue

        max_idx_local = np.argmax(counts_search)
        max_count = counts_search[max_idx_local]
        max_time = time_search[max_idx_local]

        max_idx_global = np.where(time_ns == max_time)[0][0]

        # COM around peak
        com_delay = calculate_center_of_mass(time_ns, counts, max_idx_global, com_window_ns)
        delays[(ch_ref, ch_target)] = com_delay

        print(
            f"  Ch{ch_ref}-Ch{ch_target}: Peak at {max_time:+.2f} ns "
            f"({max_count:.0f} counts), COM at {com_delay:+.2f} ns"
        )

    # For cross-harmonic pairs, reuse delay if first channel is 1, else 0
    for ch1, ch2 in CROSS_HARMONIC_PAIRS:
        if ch1 == 1:
            # Use existing calibration Ch1–ChX if available
            delays[(ch1, ch2)] = delays.get((ch1, ch2), 0.0)
        else:
            # Assume zero delay for other cross pairs
            delays[(ch1, ch2)] = 0.0

    # For harmonic autocorrelations, assume zero delay
    for ch1, ch2 in HARMONIC_PAIRS:
        if (ch1, ch2) not in delays:
            delays[(ch1, ch2)] = 0.0

    print("  ✓ Delay calibration complete!")
    return delays


def apply_time_shift_counts(time_ns, counts, shift_ns):
    """
    Shift counts array to correct for time delay (simple interpolation).
    Positive shift_ns moves peak to lower time (left).
    """
    from scipy.ndimage import shift as shift_array

    binwidth_ns = np.mean(np.diff(time_ns))
    shift_bins = shift_ns / binwidth_ns

    counts_shifted = shift_array(counts, -shift_bins, order=1, mode='constant', cval=0)
    return counts_shifted

# ============================================================================
# g2 Calculation Functions
# ============================================================================

def calculate_g2_sideband(time_ns, counts, window_ns, rep_rate=REP_RATE):
    """
    Calculate g2(0) using sideband normalization:

        g2_SB(0) = N_central / <N_sideband>
    """
    period_ns = 1e9 / rep_rate

    # Find all peaks
    binwidth_ns = np.mean(np.diff(time_ns))
    min_distance_bins = int(period_ns / binwidth_ns * 0.8)

    peaks, _ = find_peaks(counts, distance=min_distance_bins, height=np.mean(counts))

    if len(peaks) < 3:
        return np.nan, np.nan, np.nan, np.nan

    # Identify central peak (closest to zero)
    peaks_time = time_ns[peaks]
    central_idx = np.argmin(np.abs(peaks_time))
    central_peak_idx = peaks[central_idx]

    # Integrate peaks
    window_bins = int(window_ns / binwidth_ns)

    start = max(0, central_peak_idx - window_bins)
    end = min(len(counts), central_peak_idx + window_bins + 1)
    central_counts = np.sum(counts[start:end])

    # Integrate sideband peaks
    sideband_counts_list = []
    for peak_idx in peaks:
        if abs(peak_idx - central_peak_idx) > 10:  # skip central peak
            start = max(0, peak_idx - window_bins)
            end = min(len(counts), peak_idx + window_bins + 1)
            sideband_counts_list.append(np.sum(counts[start:end]))

    if len(sideband_counts_list) < 1 or central_counts == 0:
        return np.nan, np.nan, np.nan, np.nan

    sideband_counts = np.array(sideband_counts_list)
    sideband_mean = np.mean(sideband_counts)
    sideband_std = np.std(sideband_counts)

    g2_value = central_counts / sideband_mean

    return g2_value, central_counts, sideband_mean, sideband_std


def calculate_g2_countrate(time_ns, counts, window_ns, n_k, n_n, n_pulses, rep_rate=REP_RATE):
    """
    Calculate g2(0) using countrate normalization:

        g2_CR(k,n)(0) = N_kn * N_pulses / (N_k * N_n)

    Implemented as:

        g2 = (N_kn / N_k) * (N_pulses / N_n)

    to avoid overflow in intermediate products.
    """
    period_ns = 1e9 / rep_rate

    # Find central peak
    binwidth_ns = np.mean(np.diff(time_ns))
    min_distance_bins = int(period_ns / binwidth_ns * 0.8)

    peaks, _ = find_peaks(counts, distance=min_distance_bins, height=np.mean(counts))

    if len(peaks) < 1:
        return np.nan, np.nan

    peaks_time = time_ns[peaks]
    central_idx = np.argmin(np.abs(peaks_time))
    central_peak_idx = peaks[central_idx]

    # Integrate central peak
    window_bins = int(window_ns / binwidth_ns)
    start = max(0, central_peak_idx - window_bins)
    end = min(len(counts), central_peak_idx + window_bins + 1)
    central_counts = np.sum(counts[start:end])

    if n_k == 0 or n_n == 0:
        return np.nan, central_counts

    g2_value = (np.float64(central_counts) / np.float64(n_k)) * (
        np.float64(n_pulses) / np.float64(n_n)
    )
    return g2_value, central_counts

# ============================================================================
# R Parameter Calculation Functions
# ============================================================================

def calculate_R_parameter(g2_ii, g2_jj, g2_ij):
    """
    Calculate R parameter for Cauchy–Schwarz inequality test:

        R_ij = g2(i,j)^2 / (g2(i,i) * g2(j,j))

    Classical fields satisfy R_ij >= 1; quantum correlations can give R_ij < 1.
    """
    if g2_ii <= 0 or g2_jj <= 0:
        return np.nan
    return (g2_ij ** 2) / (g2_ii * g2_jj)


def calculate_R_parameters_vs_window(g2_results):
    """
    Calculate R_ij for harmonics (34, 35, 45) and detector combinations
    TT, TR, RT, RR.

    Returns
    -------
    R_results : dict
        R_results[combo][pair] = array over window sizes
        combo ∈ {'TT','TR','RT','RR'}
        pair  ∈ {'34','35','45'}
    """
    # Autocorrelations for each harmonic
    g2_H3 = g2_results[HARMONIC_AUTOS[3]]  # g2(H3T,H3R)
    g2_H4 = g2_results[HARMONIC_AUTOS[4]]  # g2(H4T,H4R)
    g2_H5 = g2_results[HARMONIC_AUTOS[5]]  # g2(H5T,H5R)

    n_windows = len(g2_H3)

    def g2_auto(mode):
        if mode == 3:
            return g2_H3
        if mode == 4:
            return g2_H4
        if mode == 5:
            return g2_H5
        raise ValueError("Unknown harmonic mode")

    R_results = {
        combo: {pair: np.zeros(n_windows) for pair in ['34', '35', '45']}
        for combo in ['TT', 'TR', 'RT', 'RR']
    }

    for combo, pairs_dict in CROSS_COMBOS.items():
        for pair_label, (ch_i, ch_j) in pairs_dict.items():
            # Harmonic indices from label '34', '35', '45'
            i_h = int(pair_label[0])  # 3, 4, 5
            j_h = int(pair_label[1])

            g2_ii_arr = g2_auto(i_h)
            g2_jj_arr = g2_auto(j_h)
            g2_ij_arr = g2_results[(ch_i, ch_j)]

            R_arr = np.zeros(n_windows)
            for k in range(n_windows):
                g2_ii = g2_ii_arr[k]
                g2_jj = g2_jj_arr[k]
                g2_ij = g2_ij_arr[k]
                R_arr[k] = calculate_R_parameter(g2_ii, g2_jj, g2_ij)

            R_results[combo][pair_label] = R_arr

    return R_results

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_correlation_with_windows(correlations, delays, data_dict, window_ns=4.0):
    """
    Plot G2(1,n)(t) for all channel pairs with integration windows shown.
    """
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_LARGE)
    axes = axes.flatten()

    period_ns = 1e9 / REP_RATE

    for idx, (ch1, ch2) in enumerate(CHANNEL_PAIRS):
        ax = axes[idx]

        if (ch1, ch2) not in correlations:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        time_bins, counts = correlations[(ch1, ch2)]
        time_ns = time_bins / 1000.0

        # Apply delay correction
        delay = delays.get((ch1, ch2), 0.0)
        counts_corrected = apply_time_shift_counts(time_ns, counts, delay)

        # Plot in specified range
        mask = (time_ns >= TIME_PLOT_RANGE[0]) & (time_ns <= TIME_PLOT_RANGE[1])
        ax.plot(time_ns[mask], counts_corrected[mask], 'b-', linewidth=1.5,
                label=f'G²(1,{ch2})(t)')

        # Find peaks and draw integration windows
        binwidth_ns = np.mean(np.diff(time_ns))
        min_distance_bins = int(period_ns / binwidth_ns * 0.8)

        peaks, _ = find_peaks(counts_corrected[mask],
                              distance=min_distance_bins,
                              height=np.mean(counts_corrected[mask]))

        time_plot = time_ns[mask]
        for peak_idx in peaks:
            peak_time = time_plot[peak_idx]

            if abs(peak_time) < period_ns / 2:
                color = 'red'
                alpha = 0.2
                label = f'Central (±{window_ns}ns)'
            else:
                color = 'green'
                alpha = 0.1
                label = (f'Sideband (±{window_ns}ns)'
                         if peak_idx == peaks[0] and abs(peak_time) > period_ns / 2
                         else None)

            ax.axvspan(peak_time - window_ns, peak_time + window_ns,
                       color=color, alpha=alpha, label=label)

        ax.axvline(0, color='red', linestyle='--', linewidth=1,
                   alpha=0.5, label='Zero delay')

        ax.set_xlabel('Delay (ns)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Coincidence counts', fontsize=11, fontweight='bold')
        ax.set_title(
            f'Ch{ch1} ({CHANNEL_NAMES[ch1]}) - Ch{ch2} ({CHANNEL_NAMES[ch2]})\n'
            f'Delay corrected: {-delay:+.2f} ns',
            fontsize=11, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(TIME_PLOT_RANGE)

    # axes[-1].axis('off')

    plt.suptitle(
        f'Coincidence Counts with Integration Windows (WINDOW_BAND = {window_ns} ns)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('01_correlation_traces_with_windows.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Saved: 01_correlation_traces_with_windows.png")
    plt.close()


def plot_g2_vs_window_comparison(window_values, g2_sideband_results, g2_countrate_results):
    """
    Plot g2(1,n)(0) vs WINDOW_BAND for both normalization methods.
    """
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_LARGE)
    axes = axes.flatten()

    for idx, (ch1, ch2) in enumerate(CHANNEL_PAIRS):
        ax = axes[idx]

        g2_sb = g2_sideband_results[(ch1, ch2)]
        g2_cr = g2_countrate_results[(ch1, ch2)]

        ax.plot(window_values, g2_sb, 'o-', linewidth=2, markersize=6,
                color='blue', label='Sideband norm.', alpha=0.8)
        ax.plot(window_values, g2_cr, 's-', linewidth=2, markersize=6,
                color='red', label='Countrate norm.', alpha=0.8)

        ax.set_xlabel('Integration window WINDOW_BAND (ns)', fontsize=11, fontweight='bold')
        ax.set_ylabel('g²(0)', fontsize=11, fontweight='bold')
        ax.set_title(
            f'Ch{ch1} ({CHANNEL_NAMES[ch1]}) - Ch{ch2} ({CHANNEL_NAMES[ch2]})',
            fontsize=11, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        max_g2 = max(np.nanmax(g2_sb), np.nanmax(g2_cr))
        finite_max = np.nanmax([x if np.isfinite(x) else 1.0 for x in [max_g2]])
        upper_limit = min(7.0, max(1.0, finite_max * 1.1))
        ax.set_ylim([1.0, upper_limit])
    # axes[-1].axis('off')

    plt.suptitle('g²(0) vs Integration Window: Sideband vs Countrate Normalization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('02_g2_vs_window_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 02_g2_vs_window_comparison.png")
    plt.close()


def plot_g2_ratio_vs_window(window_values, g2_sideband_results, g2_countrate_results):
    """
    Plot the ratio g2_SB / g2_CR vs WINDOW_BAND.
    """
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_LARGE)
    axes = axes.flatten()

    for idx, (ch1, ch2) in enumerate(CHANNEL_PAIRS):
        ax = axes[idx]

        g2_sb = g2_sideband_results[(ch1, ch2)]
        g2_cr = g2_countrate_results[(ch1, ch2)]

        ratio = g2_sb / g2_cr

        ax.plot(window_values, ratio, 'o-', linewidth=2, markersize=6,
                color='purple', alpha=0.8)

        ax.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                   label='Ratio = 1')

        ax.set_xlabel('Integration window WINDOW_BAND (ns)', fontsize=11, fontweight='bold')
        ax.set_ylabel('g²_SB / g²_CR', fontsize=11, fontweight='bold')
        ax.set_title(
            f'Ch{ch1} ({CHANNEL_NAMES[ch1]}) - Ch{ch2} ({CHANNEL_NAMES[ch2]})',
            fontsize=11, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        # ax.set_ylim([1.5, 3.])#ymax])

    # axes[-1].axis('off')

    plt.suptitle('Ratio of Normalization Methods: g²_SB(0) / g²_CR(0) vs Integration Window',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('03_g2_ratio_vs_window.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 03_g2_ratio_vs_window.png")
    plt.close()


def plot_R_parameters_vs_window(window_values, R_sideband, R_countrate):
    """
    Plot R_ij vs WINDOW_BAND for harmonics 34, 35, 45 and
    detector combinations TT, TR, RT, RR.

    Layout: 4 rows (TT, TR, RT, RR) × 3 columns (34, 35, 45).
    Each subplot shows SB and CR curves plus the classical limit R = 1.
    """
    combos = ['TT', 'TR', 'RT', 'RR']
    pairs = ['34', '35', '45']

    fig, axes = plt.subplots(len(combos), len(pairs), figsize=(18, 16))

    # Global y-limit over all R_ij, both normalizations
    all_vals = []
    for combo in combos:
        for pair in pairs:
            all_vals.append(R_sideband[combo][pair])
            all_vals.append(R_countrate[combo][pair])
    all_vals = np.concatenate(all_vals)
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size > 0:
        ymax = max(2.5, np.nanmax(finite) * 1.1)
    else:
        ymax = 2.5

    for i, combo in enumerate(combos):
        for j, pair in enumerate(pairs):
            ax = axes[i, j]

            R_sb = R_sideband[combo][pair]
            R_cr = R_countrate[combo][pair]

            ax.plot(window_values, R_sb, 'o-', linewidth=2, markersize=5,
                    color='blue', label='Sideband norm.', alpha=0.8)
            ax.plot(window_values, R_cr, 's-', linewidth=2, markersize=5,
                    color='red', label='Countrate norm.', alpha=0.8)

            ax.axhline(1, color='black', linestyle='--', linewidth=1.0, alpha=0.7,
                       label='Classical limit (R=1)')

            ax.set_xlabel('WINDOW_BAND (ns)', fontsize=10, fontweight='bold')
            ax.set_ylabel('R parameter', fontsize=10, fontweight='bold')
            ax.set_title(f'R{pair} ({combo})', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            if i == 0 and j == 0:
                ax.legend(fontsize=9)

            ax.set_ylim([0.5, 1.5])#ymax])

    plt.suptitle('R_ij vs Integration Window for TT/TR/RT/RR and 34/35/45',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('04_R_parameters_vs_window_4x3.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 04_R_parameters_vs_window_4x3.png")
    plt.close()

# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Main analysis workflow."""
    print("=" * 70)
    print("g²(0) & R PARAMETER WINDOW DEPENDENCE ANALYSIS")
    print("=" * 70)

    # Load data
    data_dict = load_json_data(JSON_FILE)
    correlations = data_dict['correlations']

    # Calibrate delays using COM method
    delays = calibrate_channel_delays_com(
        correlations,
        DELAY_SEARCH_RANGE_NS,
        com_window_ns=CALIBRATION_WINDOW_NS / 2,
    )

    # Plot correlations with integration windows
    print("\nPlotting correlation traces with integration windows...")
    plot_correlation_with_windows(
        correlations,
        delays,
        data_dict,
        window_ns=CALIBRATION_WINDOW_NS,
    )

    # Calculate g2 for different window sizes
    print("\nCalculating g²(0) for different integration windows...")
    print(f"  Window range: {WINDOW_BAND_VALUES[0]:.2f} - {WINDOW_BAND_VALUES[-1]:.2f} ns")
    print(f"  Number of windows: {len(WINDOW_BAND_VALUES)}")
    print(f"  Analyzing {len(ALL_PAIRS)} channel pairs")

    g2_sideband_results = {pair: [] for pair in ALL_PAIRS}
    g2_countrate_results = {pair: [] for pair in ALL_PAIRS}

    for window_ns in WINDOW_BAND_VALUES:
        for ch1, ch2 in ALL_PAIRS:
            if (ch1, ch2) not in correlations:
                g2_sideband_results[(ch1, ch2)].append(np.nan)
                g2_countrate_results[(ch1, ch2)].append(np.nan)
                continue

            time_bins, counts = correlations[(ch1, ch2)]
            time_ns = time_bins / 1000.0

            delay = delays.get((ch1, ch2), 0.0)
            counts_corrected = apply_time_shift_counts(time_ns, counts, delay)

            # Sideband normalization
            g2_sb, _, _, _ = calculate_g2_sideband(time_ns, counts_corrected, window_ns)
            g2_sideband_results[(ch1, ch2)].append(g2_sb)

            # Countrate normalization
            n_k = data_dict['total_counts'][ch1]
            n_n = data_dict['total_counts'][ch2]
            n_pulses = data_dict['n_pulses']

            g2_cr, _ = calculate_g2_countrate(
                time_ns,
                counts_corrected,
                window_ns,
                n_k,
                n_n,
                n_pulses,
            )
            g2_countrate_results[(ch1, ch2)].append(g2_cr)

    # Convert lists to arrays
    for pair in ALL_PAIRS:
        g2_sideband_results[pair] = np.array(g2_sideband_results[pair])
        g2_countrate_results[pair] = np.array(g2_countrate_results[pair])

    # Calculate R parameters for both normalization methods
    print("\nCalculating R parameters...")
    R_sideband = calculate_R_parameters_vs_window(g2_sideband_results)
    R_countrate = calculate_R_parameters_vs_window(g2_countrate_results)

    # Summary at WINDOW_BAND = 4.0 ns (use TT combination for R summary)
    print("\n" + "=" * 70)
    print("SUMMARY AT WINDOW_BAND = 4.0 ns")
    print("=" * 70)
    idx_4ns = np.argmin(np.abs(WINDOW_BAND_VALUES - 4.0))

    print("\n  g² values (Ch1 with 2–6):")
    for ch1, ch2 in CHANNEL_PAIRS:
        g2_sb = g2_sideband_results[(ch1, ch2)][idx_4ns]
        g2_cr = g2_countrate_results[(ch1, ch2)][idx_4ns]
        print(
            f"    Ch{ch1}-Ch{ch2} ({CHANNEL_NAMES[ch1]}-{CHANNEL_NAMES[ch2]}): "
            f"g²_SB = {g2_sb:.3f}, g²_CR = {g2_cr:.3f}, Ratio = {g2_sb/g2_cr:.3f}"
        )

    print("\n  R parameters (Cauchy–Schwarz, TT combination):")
    for pair_label in ['34', '35', '45']:
        R_sb = R_sideband['TT'][pair_label][idx_4ns]
        R_cr = R_countrate['TT'][pair_label][idx_4ns]
        print(
            f"    R{pair_label} (TT): R_SB = {R_sb:.3f}, "
            f"R_CR = {R_cr:.3f}, Ratio = {R_sb/R_cr:.3f}"
        )
        if R_sb < 1:
            print("       → SB: Violates classical limit (R < 1)")
        if R_cr < 1:
            print("       → CR: Violates classical limit (R < 1)")

    # Plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_g2_vs_window_comparison(WINDOW_BAND_VALUES, g2_sideband_results, g2_countrate_results)
    # plot_g2_ratio_vs_window(WINDOW_BAND_VALUES, g2_sideband_results, g2_countrate_results)
    plot_R_parameters_vs_window(WINDOW_BAND_VALUES, R_sideband, R_countrate)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  01_correlation_traces_with_windows.png")
    print("  02_g2_vs_window_comparison.png")
    print("  03_g2_ratio_vs_window.png")
    print("  04_R_parameters_vs_window_4x3.png")
    print("=" * 70)


if __name__ == '__main__':
    main()

"""
Minimal standalone script to test TimeTagger channel delays.
Tests correlation histograms between channel 1 and all other channels (2-6).

First run: No delays applied
Second run: With delays applied using setInputDelay

This helps verify if delays are working as expected.
"""

import TimeTagger
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Measurement parameters
MEASUREMENT_DURATION_SEC = 10  # seconds per measurement
CHANNELS = [1, 2, 3, 4, 5, 6]
REFERENCE_CHANNEL = 1  # Correlate all channels against this one

# Histogram parameters
BINWIDTH_PS = 50  # picoseconds
NUM_BINS = 5000
BIN_RANGE_PS = BINWIDTH_PS * NUM_BINS  # Total range in ps

# Delays to test (in picoseconds) - from your original script
TEST_DELAYS = {
    1: 0,
    2: -1000,
    3: 20000,
    4: 16500,
    5: 17800,
    6: 17300
}

# Save directory
SAVE_DIR = Path("./timetagger_delay_test")
SAVE_DIR.mkdir(exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_timestamp():    
    """Return current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def init_timetagger():
    """Initialize timetagger with basic settings"""
    try:
        tagger = TimeTagger.createTimeTagger()
        print(f"✓ TimeTagger initialized: {tagger}")
        print(f"  Serial: {tagger.getSerial()}")
        print(f"  Model: {tagger.getModel()}")
        return tagger
    except Exception as e:
        print(f"✗ Failed to initialize TimeTagger: {e}")
        return None

def set_input_delays(tagger, delays_dict):
    """
    Set delays on timetagger channels using setInputDelay.
    
    Args:
        tagger: TimeTagger object
        delays_dict: Dictionary {channel: delay_ps}
    """
    print("\nSetting channel input delays:")
    for channel, delay_ps in delays_dict.items():
        tagger.setInputDelay(channel, delay_ps)
        actual_delay = tagger.getInputDelay(channel)
        print(f"  Ch {channel}: requested={delay_ps} ps, actual={actual_delay} ps")

def clear_all_delays(tagger, channels):
    """Reset all channel delays to 0"""
    print("\nClearing all channel delays:")
    for channel in channels:
        tagger.setInputDelay(channel, 0)
        actual_delay = tagger.getInputDelay(channel)
        print(f"  Ch {channel}: {actual_delay} ps")

def measure_correlations(tagger, ref_channel, target_channels, 
                         binwidth_ps, num_bins, duration_sec):
    """
    Measure correlation histograms between reference channel and target channels.
    Uses SynchronizedMeasurements.
    
    Returns:
        dict: {channel_pair: {"time_bins": array, "counts": array}}
    """
    correlations = {}
    correlation_measurements = {}
    
    print(f"\nMeasuring correlations (duration: {duration_sec} s):")
    
    # Use synchronized measurements
    with TimeTagger.SynchronizedMeasurements(tagger) as sync_meas:
        sm_tagger = sync_meas.getTagger()
        
        # Create all correlation measurements
        for target_ch in target_channels:
            if target_ch == ref_channel:
                continue
                
            pair_name = f"Ch{ref_channel}-Ch{target_ch}"
            
            # Create correlation measurement
            corr = TimeTagger.Correlation(
                sm_tagger,
                ref_channel,
                target_ch,
                binwidth_ps,
                num_bins
            )
            correlation_measurements[pair_name] = corr
        
        # Start all measurements simultaneously
        duration_ps = int(duration_sec * 1e12)
        sync_meas.startFor(duration_ps)
        sync_meas.waitUntilFinished()
        
        # Collect results
        for pair_name, corr in correlation_measurements.items():
            time_bins = corr.getIndex()
            counts = corr.getData()
            
            correlations[pair_name] = {
                "time_bins": time_bins.copy(),
                "counts": counts.copy()
            }
            
            total_counts = np.sum(counts)
            max_counts = np.max(counts)
            peak_position_ps = time_bins[np.argmax(counts)]
            
            print(f"  {pair_name}: total={total_counts:,}, max={max_counts}, peak@{peak_position_ps} ps")
    
    return correlations

def plot_correlations(correlations_no_delay, correlations_with_delay, 
                      delays_dict, save_path):
    """
    Plot correlation histograms: before and after applying delays.
    """
    n_pairs = len(correlations_no_delay)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4*n_pairs))
    
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, pair_name in enumerate(sorted(correlations_no_delay.keys())):
        # Extract channel numbers
        ch1, ch2 = [int(x.replace('Ch', '')) for x in pair_name.split('-')]
        expected_delay = -(delays_dict.get(ch2, 0) - delays_dict.get(ch1, 0))
        
        # No delay plot
        ax1 = axes[idx, 0]
        data_no = correlations_no_delay[pair_name]
        time_ns = data_no["time_bins"] / 1000  # Convert to ns
        counts = data_no["counts"]
        
        ax1.plot(time_ns, counts, linewidth=0.8)
        ax1.set_xlabel("Time delay (ns)")
        ax1.set_ylabel("Counts")
        ax1.set_title(f"{pair_name} - NO DELAYS")
        ax1.grid(True, alpha=0.3)
        
        # Peak position
        peak_idx_no = np.argmax(counts)
        peak_time_no = time_ns[peak_idx_no]
        ax1.axvline(peak_time_no, color='red', linestyle='--', alpha=0.5, 
                   label=f'Peak @ {peak_time_no:.1f} ns')
        ax1.legend()
        
        # With delay plot
        ax2 = axes[idx, 1]
        data_with = correlations_with_delay[pair_name]
        time_ns_with = data_with["time_bins"] / 1000
        counts_with = data_with["counts"]
        
        ax2.plot(time_ns_with, counts_with, linewidth=0.8)
        ax2.set_xlabel("Time delay (ns)")
        ax2.set_ylabel("Counts")
        ax2.set_title(f"{pair_name} - WITH DELAYS (expected shift: {expected_delay/1000:.1f} ns)")
        ax2.grid(True, alpha=0.3)
        
        # Peak position
        peak_idx_with = np.argmax(counts_with)
        peak_time_with = time_ns_with[peak_idx_with]
        ax2.axvline(peak_time_with, color='red', linestyle='--', alpha=0.5,
                   label=f'Peak @ {peak_time_with:.1f} ns')
        
        # Show expected position
        expected_peak_ns = peak_time_no + expected_delay/1000
        ax2.axvline(expected_peak_ns, color='green', linestyle='--', alpha=0.5,
                   label=f'Expected @ {expected_peak_ns:.1f} ns')
        ax2.legend()
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")
    plt.show()

def save_results(results, filepath):
    """Save results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    
    for key, value in results.items():
        if key == "config":
            # Config is already JSON-serializable
            results_serializable[key] = value
        elif isinstance(value, dict):
            # This is either no_delay or with_delay dict
            results_serializable[key] = {}
            for pair_name, pair_data in value.items():
                if isinstance(pair_data, dict) and "time_bins" in pair_data:
                    results_serializable[key][pair_name] = {
                        "time_bins": pair_data["time_bins"].tolist() if isinstance(pair_data["time_bins"], np.ndarray) else pair_data["time_bins"],
                        "counts": pair_data["counts"].tolist() if isinstance(pair_data["counts"], np.ndarray) else pair_data["counts"]
                    }
                else:
                    results_serializable[key][pair_name] = pair_data
        else:
            results_serializable[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"✓ Results saved to: {filepath}")

# ============================================================================
# MAIN TEST ROUTINE
# ============================================================================

def main():
    print("=" * 70)
    print("TIMETAGGER DELAY TEST (using setInputDelay)")
    print("=" * 70)
    
    # Initialize timetagger
    tagger = init_timetagger()
    if not tagger:
        print("\n✗ Cannot proceed without timetagger")
        return
    
    # Calculate correlation channels
    target_channels = [ch for ch in CHANNELS if ch != REFERENCE_CHANNEL]
    
    print(f"\nTest Configuration:")
    print(f"  Channels: {CHANNELS}")
    print(f"  Reference channel: {REFERENCE_CHANNEL}")
    print(f"  Correlation pairs: Ch{REFERENCE_CHANNEL} vs {target_channels}")
    print(f"  Binwidth: {BINWIDTH_PS} ps")
    print(f"  Number of bins: {NUM_BINS}")
    print(f"  Total range: {BIN_RANGE_PS/1000:.1f} ns")
    print(f"  Duration per measurement: {MEASUREMENT_DURATION_SEC} s")
    
    results = {
        "config": {
            "channels": CHANNELS,
            "reference_channel": REFERENCE_CHANNEL,
            "binwidth_ps": BINWIDTH_PS,
            "num_bins": NUM_BINS,
            "duration_sec": MEASUREMENT_DURATION_SEC,
            "test_delays_ps": TEST_DELAYS
        }
    }
    
    # ========================================================================
    # TEST 1: WITHOUT DELAYS
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: CORRELATION MEASUREMENT WITHOUT DELAYS")
    print("=" * 70)
    
    clear_all_delays(tagger, CHANNELS)
    
    correlations_no_delay = measure_correlations(
        tagger, REFERENCE_CHANNEL, target_channels,
        BINWIDTH_PS, NUM_BINS, MEASUREMENT_DURATION_SEC
    )
    
    results["no_delay"] = correlations_no_delay
    
    # ========================================================================
    # TEST 2: WITH DELAYS
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: CORRELATION MEASUREMENT WITH DELAYS (setInputDelay)")
    print("=" * 70)
    
    set_input_delays(tagger, TEST_DELAYS)
    
    correlations_with_delay = measure_correlations(
        tagger, REFERENCE_CHANNEL, target_channels,
        BINWIDTH_PS, NUM_BINS, MEASUREMENT_DURATION_SEC
    )
    
    results["with_delay"] = correlations_with_delay
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: Peak Shifts")
    print("=" * 70)
    
    for pair_name in sorted(correlations_no_delay.keys()):
        # Extract channel numbers
        ch1, ch2 = [int(x.replace('Ch', '')) for x in pair_name.split('-')]
        
        # Expected delay shift
        expected_delay_ps = TEST_DELAYS.get(ch2, 0) - TEST_DELAYS.get(ch1, 0)
        
        # Measured peak positions
        peak_no_delay_ps = correlations_no_delay[pair_name]["time_bins"][
            np.argmax(correlations_no_delay[pair_name]["counts"])
        ]
        peak_with_delay_ps = correlations_with_delay[pair_name]["time_bins"][
            np.argmax(correlations_with_delay[pair_name]["counts"])
        ]
        
        # Actual shift
        actual_shift_ps = peak_with_delay_ps - peak_no_delay_ps
        
        print(f"\n{pair_name}:")
        print(f"  Peak without delay: {peak_no_delay_ps} ps ({peak_no_delay_ps/1000:.2f} ns)")
        print(f"  Peak with delay:    {peak_with_delay_ps} ps ({peak_with_delay_ps/1000:.2f} ns)")
        print(f"  Expected shift:     {expected_delay_ps} ps ({expected_delay_ps/1000:.2f} ns)")
        print(f"  Actual shift:       {actual_shift_ps} ps ({actual_shift_ps/1000:.2f} ns)")
        print(f"  Difference:         {actual_shift_ps - expected_delay_ps} ps ({(actual_shift_ps - expected_delay_ps)/1000:.2f} ns)")
        
        if abs(actual_shift_ps - expected_delay_ps) < 200:  # Within 200 ps
            print(f"  ✓ Delay working correctly!")
        else:
            print(f"  ✗ WARNING: Significant deviation from expected delay!")
    
    # ========================================================================
    # SAVE & PLOT
    # ========================================================================
    timestamp = get_timestamp()
    results_file = SAVE_DIR / f"delay_test_results_{timestamp}.json"
    plot_file = SAVE_DIR / f"delay_test_plot_{timestamp}.png"
    
    save_results(results, results_file)
    plot_correlations(correlations_no_delay, correlations_with_delay, 
                     TEST_DELAYS, plot_file)
    
    # Cleanup
    clear_all_delays(tagger, CHANNELS)
    TimeTagger.freeTimeTagger(tagger)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
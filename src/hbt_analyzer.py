"""
HBT Data Analysis and Multiplexed Visualization Pipeline

This module ingests engine metrics provided by HBTMeasurement objects to perform swept-window analyses on quantum correlation indicators ($g^{(2)}(0)$ and the Cauchy-Schwarz $R$ parameter).
It provides structured visualization grids to evaluate single-photon emitter purity and multi-mode correlation correlations.

Key Capabilities:
    * Integration window sweeps comparing sideband delay normalization vs raw countrate direct calculations.
    * Complete 5x3 multiplexed grids mapping auto-correlation metrics across multi-harmonic outputs (H3, H4, H5).
    * Automated Cauchy-Schwarz space mappings to track classical field limit crossings ($R > 1$).
    * Separated differential dataset comparators (`HBTComparator`) to track spectral jitter noise.

Author: Simon WITTMANN
Target Environment: Python 3.8+
Dependencies: numpy, matplotlib, hbt_measurements
"""

import numpy as np
import matplotlib.pyplot as plt

# Sync visual parameters for clean output presentation
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


class HBTAnalyzer:
    def __init__(self, hbt_measurement):
        """
        Initializes analysis workspace referencing an instantiated HBTMeasurement workspace.
        """
        self.measure = hbt_measurement



    def plot_g2_sweep(self, c1, c2, tau_min=0.3, tau_max=30.0, step=0.6, num_side_peaks=3):
        """Executes window sweeps evaluating g^(2)(0) calculations across active configurations."""
        tau_in_ns = np.arange(tau_min, tau_max, step)
        g2_delay = []
        g2_direct = []
        
        for tau in tau_in_ns:
            g2_delay.append(self.measure.compute_g2_delay(c1, c2, tau, num_side_peaks=num_side_peaks))
            g2_direct.append(self.measure.compute_g2_direct(c1, c2, tau))

        physical_name = self.measure._get_physical_name(c1, c2)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        max_g2 = max(max(g2_delay), max(g2_direct))
        y_top = max_g2 * 1.1
        
        ax.axhspan(0, 1, color="#2ca02c", alpha=0.1, label='Antibunching ($g^{(2)} < 1$)')
        ax.axhspan(1, 2, color="#ff7f0e", alpha=0.1, label='Bunching ($1 < g^{(2)} \leq 2$)')
        ax.axhspan(2, y_top, color="#d62728", alpha=0.1, label='Super-bunching ($g^{(2)} > 2$)')

        ax.plot(tau_in_ns, g2_delay, marker='o', linestyle='-', color="#1f77b4", markersize=5, alpha=0.9, label=r'Delay : $g^{(2)}(\tau)$')
        ax.plot(tau_in_ns, g2_direct, marker='o', linestyle='-', color="#9467bd", markersize=5, alpha=0.9, label=r'Direct : $g^{(2)}$')
                
        ax.set_ylim(bottom=0, top=y_top)
        ax.set_xlim(left=0, right=tau_max)
        ax.set_title(f"Integration Window Sweep: {physical_name}\n"
                     f"Sample: {self.measure.material} $\mid$ {self.measure.power} ({self.measure.date})", fontsize=11)
        ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
        ax.set_ylabel(r"$g^{(2)}(0)$")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fig, ax



    def plot_R_sweep(self, cross_pair, auto_pair_1, auto_pair_2, tau_min=0.3, tau_max=30.0, step=0.6, num_side_peaks=3):
        """Evaluates numerical sweeps tracking Cauchy-Schwarz R bounds across parameters."""
        tau_in_ns = np.arange(tau_min, tau_max, step)
        R_delay = []
        R_direct = []

        for tau in tau_in_ns:
            g2_del_cross = self.measure.compute_g2_delay(cross_pair[0], cross_pair[1], tau, num_side_peaks=num_side_peaks)
            g2_del_a1 = self.measure.compute_g2_delay(auto_pair_1[0], auto_pair_1[1], tau, num_side_peaks=num_side_peaks)
            g2_del_a2 = self.measure.compute_g2_delay(auto_pair_2[0], auto_pair_2[1], tau, num_side_peaks=num_side_peaks)
            R_delay.append(self.measure.compute_R_parameter(g2_del_cross, g2_del_a1, g2_del_a2))

            g2_dir_cross = self.measure.compute_g2_direct(cross_pair[0], cross_pair[1], tau)
            g2_dir_a1 = self.measure.compute_g2_direct(auto_pair_1[0], auto_pair_1[1], tau)
            g2_dir_a2 = self.measure.compute_g2_direct(auto_pair_2[0], auto_pair_2[1], tau)
            R_direct.append(self.measure.compute_R_parameter(g2_dir_cross, g2_dir_a1, g2_dir_a2))

        name_cross = self.measure._get_physical_name(cross_pair[0], cross_pair[1])

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        valid_R = [r for r in R_delay + R_direct if not np.isnan(r)]
        y_top = max(1.5, max(valid_R) * 1.1) if valid_R else 2.0
        
        ax.axhspan(0, 1, color='#9b59b6', alpha=0.15, label='Classical Regime ($R \leq 1$)')
        ax.axhline(y=1, color='#2c3e50', linestyle='--', lw=1.5, alpha=0.9, label='Classical Limit ($R = 1$)')
        ax.axhspan(1, y_top, color="#dd76b4", alpha=0.3, label='Quantum Regime ($R > 1$)')

        ax.plot(tau_in_ns, R_delay, marker='o', linestyle='-', color='navy', markersize=5, alpha=0.8, label='Delay')
        ax.plot(tau_in_ns, R_direct, marker='s', linestyle='-', color='#e67e22', markersize=5, alpha=0.8, label='Direct')
        
        ax.set_ylim(bottom=0, top=y_top)
        ax.set_xlim(left=0, right=tau_max)
        ax.set_title(f"Cauchy-Schwarz $R$ Metric Sweep $\mid$ Cross: [{name_cross}]\n"
                     f"Sample: {self.measure.material} $\mid$ {self.measure.power} ({self.measure.date})", fontsize=11)
        ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
        ax.set_ylabel(r"$R$ Parameter")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fig, ax



    def plot_g2_grid(self, tau_min=0.3, tau_max=30.0, step=0.6, num_side_peaks=3):
        """
        Generates a 5x3 plot grid evaluating correlation configurations.
        Row 1 maps harmonic auto-correlations (33, 44, 55 for RT configurations).
        Rows 2-5 cross-analyze combinations across targeted bands.
        """
        tau_in_ns = np.arange(tau_min, tau_max, step)
        inv_map = {v: k for k, v in self.measure.channel_map.items()}
        
        cross_rows = ['TT', 'TR', 'RT', 'RR']
        cross_cols = [('3', '4'), ('3', '5'), ('4', '5')]
        auto_cols = ['3', '4', '5'] 
        
        fig, axes = plt.subplots(5, 3, figsize=(18, 20), dpi=300)
        
        for i in range(5):
            for j in range(3):
                ax = axes[i, j]
                try:
                    if i == 0:
                        h = auto_cols[j]
                        ch1 = inv_map[f"H{h}R"]
                        ch2 = inv_map[f"H{h}T"]
                        title_str = f"Auto $g^{{(2)}}_{{{h}{h}}}$ (RT)"
                    else:
                        row_type = cross_rows[i-1]
                        hA, hB = cross_cols[j]
                        ch1 = inv_map[f"H{hA}{row_type[0]}"]
                        ch2 = inv_map[f"H{hB}{row_type[1]}"]
                        title_str = f"Cross $g^{{(2)}}_{{{hA}{hB}}}$ ({row_type})"
                except KeyError as e:
                    print(f"Skipping plot position ({i},{j}): Missing mapping metadata key {e}.")
                    ax.set_visible(False)
                    continue
                
                g2_delay, g2_direct = [], []
                for tau in tau_in_ns:
                    g2_delay.append(self.measure.compute_g2_delay(ch1, ch2, tau, num_side_peaks=num_side_peaks))
                    g2_direct.append(self.measure.compute_g2_direct(ch1, ch2, tau))

                valid_g2 = [g for g in g2_delay + g2_direct if not np.isnan(g)]
                y_top = max(valid_g2) * 1.1 if valid_g2 else 2.5

                ax.axhspan(0, 1, color='#2ecc71', alpha=0.1)                
                ax.axhspan(1, 2, color='#f1c40f', alpha=0.1)                
                ax.axhspan(2, y_top, color='#e74c3c', alpha=0.1)            
                ax.axhline(y=1, color='#27ae60', linestyle='--', alpha=0.9) 
                
                lbl_delay = 'Delay (Method A)' if i==0 and j==0 else ""
                lbl_direct = 'Direct (Method B)' if i==0 and j==0 else ""
                
                ax.plot(tau_in_ns, g2_delay, marker='o', linestyle='-', color='navy', markersize=5, alpha=0.8, label=lbl_delay)
                ax.plot(tau_in_ns, g2_direct, marker='s', linestyle='-', color='#c0392b', markersize=5, alpha=0.8, label=lbl_direct) 
                
                ax.set_ylim(0, y_top)
                ax.set_xlim(tau_min, tau_max)
                ax.set_title(title_str, fontsize=13)
                ax.grid(True, alpha=0.3)
                if i == 4: ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
                if j == 0: ax.set_ylabel(r"$g^{(2)}(0)$")

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2)
        plt.suptitle(f"Comprehensive Tracking Grid: Harmonics 3, 4, 5\n"
                     f"Sample: {self.measure.material} $\mid$ {self.measure.power} $\mid$ {self.measure.run_number} ({self.measure.date})", fontsize=18, y=1.01)
        plt.tight_layout()
        plt.show()
        return fig, axes



    def plot_R_grid(self, tau_min=0.3, tau_max=30.0, step=0.6, num_side_peaks=3):
        """Generates a structured multi-column layout tracking absolute Cauchy-Schwarz variables."""
        tau_in_ns = np.arange(tau_min, tau_max, step)
        inv_map = {v: k for k, v in self.measure.channel_map.items()}
        
        try:
            auto_pairs = {
                '3': (inv_map['H3R'], inv_map['H3T']),
                '4': (inv_map['H4R'], inv_map['H4T']),
                '5': (inv_map['H5R'], inv_map['H5T'])
            }
        except KeyError as e:
            print(f"Aborting matrix creation: Incomplete detector parameter arrays for key {e}.")
            return

        g2_auto_delay = {k: [] for k in auto_pairs.keys()}
        g2_auto_direct = {k: [] for k in auto_pairs.keys()}
        
        for tau in tau_in_ns:
            for key, (r_ch, t_ch) in auto_pairs.items():
                g2_auto_delay[key].append(self.measure.compute_g2_delay(r_ch, t_ch, tau, num_side_peaks=num_side_peaks))
                g2_auto_direct[key].append(self.measure.compute_g2_direct(r_ch, t_ch, tau))

        rows = ['TT', 'TR', 'RT', 'RR']
        cols = [('3', '4'), ('3', '5'), ('4', '5')]
        
        fig, axes = plt.subplots(4, 3, figsize=(18, 16), dpi=300)
        
        for i, row in enumerate(rows):
            for j, (hA, hB) in enumerate(cols):
                ax = axes[i, j]
                ch1 = inv_map[f"H{hA}{row[0]}"] 
                ch2 = inv_map[f"H{hB}{row[1]}"] 
                
                R_delay, R_direct = [], []
                for t_idx, tau in enumerate(tau_in_ns):
                    g2_del_cross = self.measure.compute_g2_delay(ch1, ch2, tau, num_side_peaks=num_side_peaks)
                    g2_dir_cross = self.measure.compute_g2_direct(ch1, ch2, tau)
                    
                    R_delay.append(self.measure.compute_R_parameter(g2_del_cross, g2_auto_delay[hA][t_idx], g2_auto_delay[hB][t_idx]))
                    R_direct.append(self.measure.compute_R_parameter(g2_dir_cross, g2_auto_direct[hA][t_idx], g2_auto_direct[hB][t_idx]))

                ax.axhspan(0, 1, color='#9b59b6', alpha=0.15)
                ax.axhline(y=1, color='#2c3e50', linestyle='--', lw=1.5, alpha=0.9)
                ax.axhspan(1, 100, color="#dd76b4", alpha=0.3)
                
                ax.plot(tau_in_ns, R_delay, marker='o', linestyle='-', color='navy', markersize=5, alpha=0.8, label='Delay' if i==0 and j==0 else "")
                ax.plot(tau_in_ns, R_direct, marker='s', linestyle='-', color='#e67e22', markersize=5, alpha=0.8, label='Direct' if i==0 and j==0 else "") 
                
                valid_R = [r for r in R_delay + R_direct if not np.isnan(r)]
                y_top = max(1.5, max(valid_R) * 1.1) if valid_R else 2.0
                ax.set_ylim(0, y_top)
                ax.set_xlim(tau_min, tau_max)
                ax.set_title(f"$R_{{{hA}{hB}}}$ ({row})")
                ax.grid(True, alpha=0.3)
                if i == 3: ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
                if j == 0: ax.set_ylabel(r"$R$ Parameter")

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=2)
        plt.suptitle(f"Cauchy-Schwarz $R$ Metric Space Correlation Matrix\n"
                     f"Sample: {self.measure.material} $\mid$ {self.measure.power} $\mid$ {self.measure.run_number} ({self.measure.date})", fontsize=18, y=1.01)
        plt.tight_layout()
        plt.show()
        return fig, axes





class HBTComparator:
    def __init__(self, run_A, run_B, label_A="Run A", label_B="Run B", comparison_type="Variable"):
        """
        Initializes the cross-dataset comparison engine using two distinct HBTMeasurement configurations.
        
        Parameters:
            run_A (HBTMeasurement): First measurement instance.
            run_B (HBTMeasurement): Second measurement instance to compare against run_A.
            label_A (str): Value/Description of condition A (e.g., "10nm Bandpass", "0 Deg").
            label_B (str): Value/Description of condition B (e.g., "50nm Broad", "90 Deg").
            comparison_type (str): The physical parameter being tested (e.g., "Optical Filter", "Polarization").
        """
        self.run_A = run_A
        self.run_B = run_B
        self.label_A = label_A
        self.label_B = label_B
        self.comparison_type = comparison_type

    def _generate_smart_title(self, base_title):
        """
        Constructs a clean, dynamic title by grouping shared baseline metadata 
        and explicitly highlighting the variable being compared.
        """
        # Group metadata if it's identical across runs, otherwise show the delta
        mat_str = self.run_A.material if self.run_A.material == self.run_B.material else f"{self.run_A.material} vs {self.run_B.material}"
        pow_str = self.run_A.power if self.run_A.power == self.run_B.power else f"{self.run_A.power} vs {self.run_B.power}"
        
        return (f"{base_title}\n"
                f"Varying {self.comparison_type}: [{self.label_A}] $\\leftrightarrow$ [{self.label_B}]\n"
                f"Baseline: {mat_str} $\\mid$ {pow_str}")

    def plot_g2_comparison(self, c1=None, c2=None, tau_min=0.3, tau_max=30.0, step=0.6, num_side_peaks=3):
        """
        Polymorphic interface to compare direct g^(2)(0) tracking data across two conditions.
        
        If c1 and c2 are provided: Renders a single high-resolution comparison chart.
        If c1 and c2 are None: Renders a complete 5x3 differential visualization matrix.
        """
        tau_in_ns = np.arange(tau_min, tau_max, step)
        
        # ------------------ BRANCH 1: MASTER MULTIPLEXED GRID ------------------
        if c1 is None or c2 is None:
            inv_map_A = {v: k for k, v in self.run_A.channel_map.items()}
            inv_map_B = {v: k for k, v in self.run_B.channel_map.items()}
            
            cross_rows = ['TT', 'TR', 'RT', 'RR']
            cross_cols = [('3', '4'), ('3', '5'), ('4', '5')]
            auto_cols = ['3', '4', '5']
            
            fig, axes = plt.subplots(5, 3, figsize=(18, 20), dpi=300)
            
            for i in range(5):
                for j in range(3):
                    ax = axes[i, j]
                    
                    try:
                        if i == 0:
                            h = auto_cols[j]
                            ch1_A = inv_map_A[f"H{h}R"]; ch2_A = inv_map_A[f"H{h}T"]
                            ch1_B = inv_map_B[f"H{h}R"]; ch2_B = inv_map_B[f"H{h}T"]
                            subplot_title = f"Auto $g^{{(2)}}_{{{h}{h}}}$ (RT)"
                        else:
                            row_type = cross_rows[i-1]
                            hA, hB = cross_cols[j]
                            ch1_A = inv_map_A[f"H{hA}{row_type[0]}"]; ch2_A = inv_map_A[f"H{hB}{row_type[1]}"]
                            ch1_B = inv_map_B[f"H{hA}{row_type[0]}"]; ch2_B = inv_map_B[f"H{hB}{row_type[1]}"]
                            subplot_title = f"Cross $g^{{(2)}}_{{{hA}{hB}}}$ ({row_type})"
                    except KeyError as e:
                        print(f"Skipping subplot context at ({i},{j}): Missing channel reference mapping {e}.")
                        ax.set_visible(False)
                        continue
                    
                    g2_A, g2_B = [], []
                    for tau in tau_in_ns:
                        g2_A.append(self.run_A.compute_g2_direct(ch1_A, ch2_A, tau))
                        g2_B.append(self.run_B.compute_g2_direct(ch1_B, ch2_B, tau))
                        
                    valid_g2 = [g for g in g2_A + g2_B if not np.isnan(g)]
                    y_top = max(valid_g2) * 1.1 if valid_g2 else 2.5
                    
                    # Apply standardized physical background bounds
                    ax.axhspan(0, 1, color='#2ecc71', alpha=0.08)                
                    ax.axhspan(1, 2, color='#f1c40f', alpha=0.08)                
                    ax.axhspan(2, y_top, color='#e74c3c', alpha=0.08)            
                    ax.axhline(y=1, color='#27ae60', linestyle='--', alpha=0.8) 
                    
                    lbl_A = self.label_A if i == 0 and j == 0 else ""
                    lbl_B = self.label_B if i == 0 and j == 0 else ""
                    
                    ax.plot(tau_in_ns, g2_A, marker='o', linestyle='-', color='#e74c3c', markersize=4, alpha=0.8, label=lbl_A)
                    ax.plot(tau_in_ns, g2_B, marker='s', linestyle='-', color='#2980b9', markersize=4, alpha=0.8, label=lbl_B)
                    
                    ax.set_ylim(0, y_top)
                    ax.set_xlim(tau_min, tau_max)
                    ax.set_title(subplot_title, fontsize=12)
                    ax.grid(True, alpha=0.3)
                    
                    if i == 4: ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
                    if j == 0: ax.set_ylabel(r"$g^{(2)}_{direct}(0)$")
                    if i == 0 and j == 0: ax.legend(loc='best')
                    
            master_title = self._generate_smart_title("Differential Spectrum Analysis Matrix")
            plt.suptitle(master_title, fontsize=18, y=1.02)
            plt.tight_layout()
            plt.show()
            return fig, axes

        # ------------------ BRANCH 2: SINGLE PAIR PLOT ------------------
        else:
            g2_A, g2_B = [], []
            for tau in tau_in_ns:
                g2_A.append(self.run_A.compute_g2_direct(c1, c2, tau))
                g2_B.append(self.run_B.compute_g2_direct(c1, c2, tau))

            physical_name = self.run_A._get_physical_name(c1, c2)

            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            max_g2 = max(max(g2_A), max(g2_B))
            y_top = max_g2 * 1.1

            ax.axhspan(0, 1, color='#2ecc71', alpha=0.1, label='Antibunching ($g^{(2)} < 1$)')
            ax.axhspan(1, 2, color='#f1c40f', alpha=0.1, label='Bunching ($1 \leq g^{(2)} \leq 2$)')
            ax.axhspan(2, y_top, color='#e74c3c', alpha=0.1, label='Super-bunching ($g^{(2)} > 2$)')
            ax.axhline(y=1, color='#27ae60', linestyle='--', alpha=0.9) 

            ax.plot(tau_in_ns, g2_A, marker='o', linestyle='-', color='#e74c3c', markersize=5, alpha=0.8, label=self.label_A)
            ax.plot(tau_in_ns, g2_B, marker='s', linestyle='-', color='#2980b9', markersize=5, alpha=0.8, label=self.label_B)
            
            ax.set_ylim(bottom=0, top=y_top)
            ax.set_xlim(left=0, right=tau_max)
            
            single_title = self._generate_smart_title(f"Differential Stability Test: {physical_name}")
            ax.set_title(single_title, fontsize=11)
            ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
            ax.set_ylabel(r"$g^{(2)}_{direct}(0)$")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            return fig, ax
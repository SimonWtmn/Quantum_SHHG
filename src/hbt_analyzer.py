import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
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
        Initializes the analyzer with a loaded HBTMeasurement object as its data engine.
        """
        self.measure = hbt_measurement



    def plot_g2_sweep(self, c1, c2, tau_min=0.3, tau_max=30.0, step=0.6, num_side_peaks=3):
        """
        Sweeps the integration window \tau_in and compares Method A (Delay/Sideband) 
        vs Method B (Direct/Countrate) for g2(0), with physical regimes shaded.
        """
        tau_in_ns = np.arange(tau_min, tau_max, step)
        g2_delay = []
        g2_direct = []
        
        for tau in tau_in_ns:
            g2_delay.append(self.measure.compute_g2_delay(c1, c2, tau, num_side_peaks=num_side_peaks))
            g2_direct.append(self.measure.compute_g2_direct(c1, c2, tau))

        key = self.measure._get_channel_key(c1, c2)
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
        ax.set_title(f"Integration Window Sweep: {physical_name}")
        ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
        ax.set_ylabel(r"$g^{(2)}(0)$")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    


    def plot_R_sweep(self, cross_pair, auto_pair_1, auto_pair_2, tau_min=0.3, tau_max=30.0, step=0.6, num_side_peaks=3):
        """
        Sweeps the integration window \tau_in and computes the Cauchy-Schwarz R parameter.
        cross_pair: tuple, e.g., (1, 3) for H3T-H4T
        auto_pair_1: tuple, e.g., (1, 2) for H3 auto-correlation
        auto_pair_2: tuple, e.g., (3, 4) for H4 auto-correlation
        """
        tau_in_ns = np.arange(tau_min, tau_max, step)
        R_delay = []
        R_direct = []

        for tau in tau_in_ns:
            # Method A: Delay (Sideband) Normalization
            g2_del_cross = self.measure.compute_g2_delay(cross_pair[0], cross_pair[1], tau, num_side_peaks=num_side_peaks)
            g2_del_a1 = self.measure.compute_g2_delay(auto_pair_1[0], auto_pair_1[1], tau, num_side_peaks=num_side_peaks)
            g2_del_a2 = self.measure.compute_g2_delay(auto_pair_2[0], auto_pair_2[1], tau, num_side_peaks=num_side_peaks)
            R_del = self.measure.compute_R_parameter(g2_del_cross, g2_del_a1, g2_del_a2)
            R_delay.append(R_del)

            # Method B: Direct (Countrate) Normalization
            g2_dir_cross = self.measure.compute_g2_direct(cross_pair[0], cross_pair[1], tau)
            g2_dir_a1 = self.measure.compute_g2_direct(auto_pair_1[0], auto_pair_1[1], tau)
            g2_dir_a2 = self.measure.compute_g2_direct(auto_pair_2[0], auto_pair_2[1], tau)
            R_dir = self.measure.compute_R_parameter(g2_dir_cross, g2_dir_a1, g2_dir_a2)
            R_direct.append(R_dir)

        name_cross = self.measure._get_physical_name(cross_pair[0], cross_pair[1])
        name_a1 = self.measure._get_physical_name(auto_pair_1[0], auto_pair_1[1])
        name_a2 = self.measure._get_physical_name(auto_pair_2[0], auto_pair_2[1])


        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        valid_R = [r for r in R_delay + R_direct if not np.isnan(r)]
        y_top = max(1.5, max(valid_R) * 1.1) if valid_R else 2.0
        
        ax.axhspan(0, 1, color='#9b59b6', alpha=0.15, label='Classical Regime ($R < 1$)')
        ax.axhline(y=1, color='#2c3e50', linestyle='--', lw=1.5, alpha=0.9, label='Classical Limit ($R = 1$)')
        ax.axhspan(1, y_top, color="#dd76b4", alpha=0.3, label='Quantum Regime ($R > 1$)')

        ax.plot(tau_in_ns, R_delay, marker='o', linestyle='-', color='navy', markersize=5, alpha=0.8, label='Delay')
        ax.plot(tau_in_ns, R_direct, marker='s', linestyle='-', color='#e67e22', markersize=5, alpha=0.8, label='Direct')
        
        ax.set_ylim(bottom=0, top=y_top)
        ax.set_xlim(left=0, right=tau_max)
        ax.set_title(f"Cauchy-Schwarz R Parameter Sweep\nCross: [{name_cross}] $\mid$ Autos: [{name_a1}], [{name_a2}]", fontsize=12)
        ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
        ax.set_ylabel(r"$R$ Parameter")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    


    def plot_R_grid(self, tau_min=0.3, tau_max=30.0, step=0.6, num_side_peaks=3):
        """
        Generates a 4x3 grid of R-parameter sweeps for all combinations of H3, H4, and H5.
        Automatically maps channels based on the internal configuration.
        """
        tau_in_ns = np.arange(tau_min, tau_max, step)
        inv_map = {v: k for k, v in self.measure.channel_map.items()}
        
        try:
            auto_pairs = {
                '3': (inv_map['H3R'], inv_map['H3T']),
                '4': (inv_map['H4R'], inv_map['H4T']),
                '5': (inv_map['H5R'], inv_map['H5T'])
            }
        except KeyError as e:
            print(f"Grid generation failed: Missing expected channel {e} in data parameters.")
            return


        print("Pre-calculating auto-correlations...")
        g2_auto_delay = {k: [] for k in auto_pairs.keys()}
        g2_auto_direct = {k: [] for k in auto_pairs.keys()}
        
        for tau in tau_in_ns:
            for key, (r_ch, t_ch) in auto_pairs.items():
                g2_auto_delay[key].append(self.measure.compute_g2_delay(r_ch, t_ch, tau, num_side_peaks=num_side_peaks))
                g2_auto_direct[key].append(self.measure.compute_g2_direct(r_ch, t_ch, tau))


        rows = ['TT', 'TR', 'RT', 'RR']
        cols = [('3', '4'), ('3', '5'), ('4', '5')]
        
        fig, axes = plt.subplots(4, 3, figsize=(18, 16), dpi=300)
        
        print("Calculating cross-correlations and plotting grid...")
        for i, row in enumerate(rows):
            for j, (hA, hB) in enumerate(cols):
                ax = axes[i, j]
                
                # Determine which specific channels we need for this subplot
                ch1 = inv_map[f"H{hA}{row[0]}"] # e.g., if hA='3', row[0]='T' -> 'H3T'
                ch2 = inv_map[f"H{hB}{row[1]}"] # e.g., if hB='4', row[1]='R' -> 'H4R'
                
                R_delay = []
                R_direct = []
                
                # Sweep the cross-correlation and calculate R
                for t_idx, tau in enumerate(tau_in_ns):
                    g2_del_cross = self.measure.compute_g2_delay(ch1, ch2, tau, num_side_peaks=num_side_peaks)
                    g2_dir_cross = self.measure.compute_g2_direct(ch1, ch2, tau)
                    
                    R_del = self.measure.compute_R_parameter(
                        g2_del_cross, g2_auto_delay[hA][t_idx], g2_auto_delay[hB][t_idx]
                    )
                    R_dir = self.measure.compute_R_parameter(
                        g2_dir_cross, g2_auto_direct[hA][t_idx], g2_auto_direct[hB][t_idx]
                    )
                    
                    R_delay.append(R_del)
                    R_direct.append(R_dir)

                # --- Plotting the Subplot ---
                # Quantum Regime Background Shading (R < 1)
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

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2)
        plt.suptitle('Cauchy-Schwarz R Parameter Grid: Harmonics 3, 4, 5', fontsize=18, y=1.01)
        plt.tight_layout()
        plt.show()
        
        return fig, axes
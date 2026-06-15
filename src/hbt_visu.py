"""
=============================================================================
HBT Data Analysis and Grid Visualization Pipeline
=============================================================================

Provides tools to generate grids, allowing for intuitive cross-comparison of 
- coherence spectra, 
- g^(2) integration sweeps, 
- Cauchy-Schwarz R parameters

Key Features
------------
* Grid Rendering: Constructs complex subplot matrices (e.g., 5x3 or 4x3) for exhaustive cross-channel and auto-correlation analysis.
* Multi-Dataset Overlays: Compares multiple experimental runs with unified, conflict-free global legends and consistent color mapping.

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquée (LOA), École Polytechnique
Date: 15/06/2026
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "figure.titlesize": 20,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

class GridVisualizer:
    def __init__(self, measurements, labels=None, comparison_variable=None):
        if not isinstance(measurements, list):
            self.runs = [measurements]
            self.labels = labels if labels else ["Dataset"]
        else:
            self.runs = measurements
            self.labels = labels if labels else [f"Run {i+1}" for i in range(len(measurements))]
        
        self.comparison_variable = comparison_variable
        self.method_colors = {'direct': '#e74c3c', 'delay': '#2980b9', 'heralded': '#27ae60'}
        self.run_colors = ['#e74c3c', '#2980b9', '#8e44ad', '#f39c12', '#2c3e50']


    def _get_title(self, base_title):
        if len(self.runs) == 1:
            return f"{base_title} | Sample: {self.runs[0].material} | {self.runs[0].power} ({self.runs[0].date})"
        else:
            var_str = f"Varying: {self.comparison_variable}" if self.comparison_variable else "Cross-Dataset Comparison"
            return f"{base_title} | {var_str} | Base Sample: {self.runs[0].material}"


    def _add_global_legend(self, fig, axes):
        """Extract legend for supblots"""
        all_axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        handles, labels = [], []
        
        for ax in all_axes:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break 
                
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), 
                       ncol=max(1, len(labels)), frameon=True)
            
        plt.tight_layout



# ---------------- 1. Coherence ----------------

    def plot_coherence(self, c1=None, c2=None, time_window_ns=5.0, xlim=None, integration_window_ns=None):
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(self.runs))))

        if xlim is None:
            x_bounds = (-time_window_ns, time_window_ns)
        elif isinstance(xlim, (int, float)):
            x_bounds = (-xlim, xlim) 
        else:
            x_bounds = xlim

        # SINGLE PLOT
        if c1 is not None and c2 is not None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            phys_name = self.runs[0]._get_physical_name(c1, c2)
            for idx, run in enumerate(self.runs):
                try:
                    ch1, ch2 = run.get_ch(self.runs[0].channel_map[c1]), run.get_ch(self.runs[0].channel_map[c2])
                    x, y = run.get_correlation_trace(ch1, ch2)
                    if x is None: continue
                    
                    t0 = run.calculate_t0_shift(ch1, ch2, (1 / run.rep_rate_hz * 1e9))
                    delta_t = x - t0
                    
                    mask = (delta_t >= x_bounds[0] - 1.0) & (delta_t <= x_bounds[1] + 1.0)
                    
                    ax.step(delta_t[mask], y[mask]*1e-3, where='mid', color=colors[idx], lw=1.2, label=self.labels[idx])
                    if len(self.runs) == 1: ax.fill_between(delta_t[mask], y[mask]*1e-3, step='mid', color=colors[idx], alpha=0.3)
                except KeyError: pass
            
            if integration_window_ns is not None:
                ax.axvspan(-integration_window_ns / 2, integration_window_ns / 2, color="#ea4432", alpha=0.3, label=r'Fen\^etre $\tau_{in}$')
            
            ax.set_title(self._get_title(f"Coherence Spectrum: {phys_name}"))
            ax.set_xlabel(r"$\Delta t$ (ns)"); ax.set_ylabel(r"Counts $N \times 10^3$")
            ax.set_xlim(x_bounds)
            ax.grid(True, alpha=0.3)
            self._add_global_legend(fig, ax)
            plt.show()
            return fig, ax

        # MASTER GRID
        cross_rows, cross_cols = ['TT', 'TR', 'RT', 'RR'], [('3', '4'), ('3', '5'), ('4', '5')]
        auto_cols = ['3', '4', '5']
        fig, axes = plt.subplots(5, 3, figsize=(18, 20), dpi=300)
        
        for i in range(5):
            for j in range(3):
                ax = axes[i, j]
                try:
                    if i == 0:
                        ch1_ref, ch2_ref = self.runs[0].get_ch(f"H{auto_cols[j]}R"), self.runs[0].get_ch(f"H{auto_cols[j]}T")
                        title = f"Coherence $H_{{{auto_cols[j]}{auto_cols[j]}}}$ (RT)"
                    else:
                        hA, hB = cross_cols[j]
                        ch1_ref, ch2_ref = self.runs[0].get_ch(f"H{hA}{cross_rows[i-1][0]}"), self.runs[0].get_ch(f"H{hB}{cross_rows[i-1][1]}")
                        title = f"Coherence $H_{{{hA}{hB}}}$ ({cross_rows[i-1]})"
                except KeyError:
                    ax.set_visible(False)
                    continue

                for idx, run in enumerate(self.runs):
                    try:
                        ch1, ch2 = run.get_ch(self.runs[0].channel_map[ch1_ref]), run.get_ch(self.runs[0].channel_map[ch2_ref])
                        x, y = run.get_correlation_trace(ch1, ch2)
                        if x is None: continue
                        
                        t0 = run.calculate_t0_shift(ch1, ch2, (1 / run.rep_rate_hz * 1e9))
                        delta_t = x - t0
                        mask = (delta_t >= x_bounds[0] - 1.0) & (delta_t <= x_bounds[1] + 1.0)
                        
                        ax.step(delta_t[mask], y[mask]*1e-3, where='mid', color=colors[idx], lw=1.2, label=self.labels[idx])
                        if len(self.runs) == 1: ax.fill_between(delta_t[mask], y[mask]*1e-3, step='mid', color=colors[idx], alpha=0.3)
                    except KeyError: pass

                if integration_window_ns is not None:
                    ax.axvspan(-integration_window_ns / 2, integration_window_ns / 2, 
                               color="#ea4432", alpha=0.3,
                               label=r'Fen\^etre $\tau_{in}$')

                ax.set_title(title)
                ax.set_xlim(x_bounds)
                ax.grid(True, alpha=0.3)
                if i == 4: ax.set_xlabel(r"$\Delta t$ (ns)")
                if j == 0: ax.set_ylabel(r"Counts $N \times 10^3$")

        plt.suptitle(self._get_title("Coherence Spectrum Matrix"), y=0.95)
        self._add_global_legend(fig, axes)
        plt.show()
        return fig, axes




    # ---------------- 2. G2 Visualization ----------------

    def plot_g2(self, c1=None, c2=None, methods=['direct'], tau_min=0.3, tau_max=30.0, step=0.6):
        tau_in_ns = np.arange(tau_min, tau_max, step)

        if c1 is not None and c2 is not None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            phys_name = self.runs[0]._get_physical_name(c1, c2)
            self._fill_ax_g2(ax, c1, c2, tau_in_ns, methods)
            ax.set_title(self._get_title(f"Integration Sweep: {phys_name}"), y=1.1)
            ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)"); ax.set_ylabel(r"$g^{(2)}(0)$")
            self._add_global_legend(fig, ax)
            plt.show()
            return fig, ax

        cross_rows, cross_cols = ['TT', 'TR', 'RT', 'RR'], [('3', '4'), ('3', '5'), ('4', '5')]
        fig, axes = plt.subplots(5, 3, figsize=(18, 20), dpi=300)
        
        for i in range(5):
            for j in range(3):
                ax = axes[i, j]
                try:
                    if i == 0:
                        ch1_ref, ch2_ref = self.runs[0].get_ch(f"H{['3', '4', '5'][j]}R"), self.runs[0].get_ch(f"H{['3', '4', '5'][j]}T")
                        title = f"Auto $g^{{(2)}}_{{{['3', '4', '5'][j]}{['3', '4', '5'][j]}}}$ (RT)"
                    else:
                        hA, hB = cross_cols[j]
                        ch1_ref, ch2_ref = self.runs[0].get_ch(f"H{hA}{cross_rows[i-1][0]}"), self.runs[0].get_ch(f"H{hB}{cross_rows[i-1][1]}")
                        title = f"Cross $g^{{(2)}}_{{{hA}{hB}}}$ ({cross_rows[i-1]})"
                except KeyError:
                    ax.set_visible(False)
                    continue
                
                self._fill_ax_g2(ax, ch1_ref, ch2_ref, tau_in_ns, methods)
                ax.set_title(title)
                if i == 4: ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
                if j == 0: ax.set_ylabel(r"$g^{(2)}(0)$")

        plt.suptitle(self._get_title("$g^{(2)}$ Sweeping Matrix"), y=0.95)
        self._add_global_legend(fig, axes)
        plt.show()
        return fig, axes

    def _fill_ax_g2(self, ax, ch1_ref, ch2_ref, tau_in_ns, methods):
        max_y = 2.5
        has_data = False
        
        for run_idx, run in enumerate(self.runs):
            try:
                c1, c2 = run.get_ch(self.runs[0].channel_map[ch1_ref]), run.get_ch(self.runs[0].channel_map[ch2_ref])
            except KeyError: continue

            for m_idx, method in enumerate(methods):
                vals = []
                for tau in tau_in_ns:
                    if method == 'delay': vals.append(run.compute_g2_delay(c1, c2, tau))
                    elif method == 'direct': vals.append(run.compute_g2_direct(c1, c2, tau))
                    elif method == 'heralded': vals.append(run.compute_g2_heralded(c1, c2, tau))

                valid = [v for v in vals if not np.isnan(v)]
                if valid: 
                    has_data = True
                    max_y = min(max(max_y, max(valid) * 1.1), 4.5)
                
                color = self.method_colors.get(method, '#333333') if len(self.runs) == 1 else self.run_colors[run_idx % len(self.run_colors)]
                lbl = (f"{method.capitalize()}" if len(self.runs) == 1 else f"{self.labels[run_idx]}" + (f" ({method})" if len(methods) > 1 else ""))
                marker = 'o' if method=='direct' else ('s' if method=='delay' else '^')
                linestyle = '-' if len(self.runs) == 1 else ('-' if m_idx == 0 else '--')
                
                ax.plot(tau_in_ns, vals, marker=marker, linestyle=linestyle, color=color, markersize=4, alpha=0.8, label=lbl)

        if not has_data: return

        ax.axhspan(0, 1, color='#2ecc71', alpha=0.05)    # Anti-bunching 
        ax.axhspan(1, 2, color='#f1c40f', alpha=0.05)    # Bunching
        ax.axhspan(2, 4, color='#e67e22', alpha=0.08)    # Super-bunching
        ax.axhspan(4, 100, color='#e74c3c', alpha=0.1)   # Non-physique
        
        text_x = tau_in_ns[-1] - (tau_in_ns[-1] - tau_in_ns[0]) * 0.02

        ax.axhline(y=1, color="#313131", linestyle='--', lw=1.5, alpha=0.6)
        ax.axhline(y=2, color="#313131", linestyle='--', lw=1.5, alpha=0.6)
        ax.axhline(y=4, color="#313131", linestyle='--', lw=1.5, alpha=0.6)
        if max_y >= 0:
            ax.text(text_x, 0.1, "Anti-bunching", color='#2ecc71', alpha=0.8, fontweight='bold', ha='right')
        if max_y >= 1:
            ax.text(text_x, 1.1, "Bunching", color='#f1c40f', alpha=0.8, fontweight='bold', ha='right')
        if max_y > 2.2:
            ax.text(text_x, 2.1, "Super-bunching", color='#e67e22', alpha=0.8, fontweight='bold', ha='right')
        if max_y >= 4.0:
            ax.text(text_x, 4.1, "Non-physical", color='#e74c3c', alpha=0.8, fontweight='bold', ha='right')

        ax.set_ylim(0, max_y)
        ax.set_xlim(tau_in_ns[0], tau_in_ns[-1])
        ax.grid(True, alpha=0.3)




    # ---------------- 3. R Parameter Visualization ----------------

    def plot_R(self, cross_pair=None, auto_pair_1=None, auto_pair_2=None, methods=['direct'], tau_min=0.3, tau_max=30.0, step=0.6):
        tau_in_ns = np.arange(tau_min, tau_max, step)

        if cross_pair is not None and auto_pair_1 is not None and auto_pair_2 is not None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            phys_name = self.runs[0]._get_physical_name(cross_pair[0], cross_pair[1])
            self._fill_ax_R(ax, cross_pair, auto_pair_1, auto_pair_2, tau_in_ns, methods)
            ax.set_title(self._get_title(f"Cauchy-Schwarz $R$ Sweep: {phys_name}"), pad=20)
            ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)"); ax.set_ylabel(r"$R$ Parameter")
            self._add_global_legend(fig, ax)
            plt.show()
            return fig, ax

        cross_rows, cross_cols = ['TT', 'TR', 'RT', 'RR'], [('3', '4'), ('3', '5'), ('4', '5')]
        fig, axes = plt.subplots(4, 3, figsize=(18, 16), dpi=300)
        
        for i, row in enumerate(cross_rows):
            for j, (hA, hB) in enumerate(cross_cols):
                ax = axes[i, j]
                try:
                    autoA_ref = (self.runs[0].get_ch(f"H{hA}R"), self.runs[0].get_ch(f"H{hA}T"))
                    autoB_ref = (self.runs[0].get_ch(f"H{hB}R"), self.runs[0].get_ch(f"H{hB}T"))
                    cross_ref = (self.runs[0].get_ch(f"H{hA}{row[0]}"), self.runs[0].get_ch(f"H{hB}{row[1]}"))
                except KeyError:
                    ax.set_visible(False)
                    continue
                
                self._fill_ax_R(ax, cross_ref, autoA_ref, autoB_ref, tau_in_ns, methods)
                ax.set_title(f"$R_{{{hA}{hB}}}$ ({row})")
                if i == 3: ax.set_xlabel(r"Integration window $\tau_{in}$ (ns)")
                if j == 0: ax.set_ylabel(r"$R$ Parameter")

        meth_title = ", ".join([m.capitalize() for m in methods])
        plt.suptitle(self._get_title(f"Cauchy-Schwarz Sweeping Matrix"), y=0.95)
        self._add_global_legend(fig, axes)
        plt.show()
        return fig, axes

    def _fill_ax_R(self, ax, cross_ref, autoA_ref, autoB_ref, tau_in_ns, methods):
        max_y = 2
        has_data = False
        
        for run_idx, run in enumerate(self.runs):
            try:
                autoA = (run.get_ch(self.runs[0].channel_map[autoA_ref[0]]), run.get_ch(self.runs[0].channel_map[autoA_ref[1]]))
                autoB = (run.get_ch(self.runs[0].channel_map[autoB_ref[0]]), run.get_ch(self.runs[0].channel_map[autoB_ref[1]]))
                cross = (run.get_ch(self.runs[0].channel_map[cross_ref[0]]), run.get_ch(self.runs[0].channel_map[cross_ref[1]]))
            except KeyError: continue
            
            for m_idx, method in enumerate(methods):
                R_vals = []
                for tau in tau_in_ns:
                    if method == 'delay':
                        c, aA, aB = run.compute_g2_delay(cross[0], cross[1], tau), run.compute_g2_delay(autoA[0], autoA[1], tau), run.compute_g2_delay(autoB[0], autoB[1], tau)
                    elif method == 'heralded':
                        c, aA, aB = run.compute_g2_heralded(cross[0], cross[1], tau), run.compute_g2_heralded(autoA[0], autoA[1], tau), run.compute_g2_heralded(autoB[0], autoB[1], tau)
                    else:
                        c, aA, aB = run.compute_g2_direct(cross[0], cross[1], tau), run.compute_g2_direct(autoA[0], autoA[1], tau), run.compute_g2_direct(autoB[0], autoB[1], tau)
                    R_vals.append(run.compute_R_parameter(c, aA, aB))

                valid = [r for r in R_vals if not np.isnan(r)]
                if valid: 
                    has_data = True
                    # Plafond à 3 pour le paramètre R
                    max_y = min(max(max_y, max(valid) * 1.1), 3.0)
                
                color = self.method_colors.get(method, '#333333') if len(self.runs) == 1 else self.run_colors[run_idx % len(self.run_colors)]
                lbl = (f"{method.capitalize()}" if len(self.runs) == 1 else f"{self.labels[run_idx]}" + (f" ({method})" if len(methods) > 1 else ""))
                marker = 'o' if method=='direct' else ('s' if method=='delay' else '^')
                linestyle = '-' if len(self.runs) == 1 else ('-' if m_idx == 0 else '--')
                
                ax.plot(tau_in_ns, R_vals, marker=marker, linestyle=linestyle, color=color, markersize=5, alpha=0.8, label=lbl)

        if not has_data: return

        ax.axhline(y=1, color='#7f8c8d', linestyle='--', lw=1.5, alpha=0.6)
        ax.axhspan(0, 1, color='#9b59b6', alpha=0.08)
        ax.axhspan(1, 100, color="#dd76b4", alpha=0.15)
        
        ax.set_ylim(0, max_y)
        ax.set_xlim(tau_in_ns[0], tau_in_ns[-1])
        ax.grid(True, alpha=0.3)
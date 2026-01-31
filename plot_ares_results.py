"""
ARES BO Prior Results - Publication Quality Plots

Generates plots in the style of the paper:
1. Convergence curves (Fig 4.9 style)
2. Final MAE vs Steps to Convergence scatter (Fig 4.10 style)
3. Results table visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os

# Try to use science plots style
try:
    import scienceplots
    plt.style.use(["science", "ieee", "no-latex"])
except ImportError:
    plt.style.use('seaborn-v0_8-whitegrid')

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_data(data_dir: str = "data/") -> Dict[str, pd.DataFrame]:
    """Load all optimization result CSVs."""
    file_mapping = {
        "Nelder-Mead": "NM_mismatched.csv",
        "BO (zero mean)": "BO_mismatched.csv",
        "BO_prior (mismatched)": "BO_prior_mismatched.csv",
        "BO_prior (matched)": "BO_prior_matched_prior_newtask.csv",
    }
    
    data = {}
    for name, filename in file_mapping.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'best_mae' not in df.columns:
                df['best_mae'] = df.groupby('run')['mae'].cummin()
            data[name] = df
            print(f"Loaded {name}: {len(df)} rows, {df['run'].nunique()} runs")
    
    return data


def plot_convergence_curves(
    data: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot convergence curves (minimum MAE over steps) - Fig 4.9 style."""
    fig, ax = plt.subplots(figsize=figsize)
    
    styles = {
        "Nelder-Mead": {"ls": "-", "color": COLORS[0], "lw": 2},
        "BO (zero mean)": {"ls": "-.", "color": COLORS[1], "lw": 2},
        "BO_prior (mismatched)": {"ls": "--", "color": COLORS[2], "lw": 2.5},
        "BO_prior (matched)": {"ls": ":", "color": COLORS[3], "lw": 2.5},
    }
    
    for name, df in data.items():
        style = styles.get(name, {"ls": "-", "color": "gray", "lw": 1.5})
        sns.lineplot(data=df, x="step", y="best_mae", ax=ax, label=name, **style)
    
   # all_best = min(df['best_mae'].min() for df in data.values())
    #ax.axhline(y=all_best, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    #ax.text(0.98, 0.02, f"Best: {all_best*1000:.3f} mm", transform=ax.transAxes,
     #       color="grey", ha="right", va="bottom", fontsize=9,
      #      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.8))
    
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Minimum MAE (m)", fontsize=12)
    ax.set_xlim(0, None)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    
    return fig


def plot_boxplot_comparison(
    data: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4),
) -> plt.Figure:
    """Boxplot comparison of final MAEs across methods."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    styles = {
        "Nelder-Mead": {"color": COLORS[0]},
        "BO (zero mean)": {"color": COLORS[1]},
        "BO_prior (mismatched)": {"color": COLORS[2]},
        "BO_prior (matched)": {"color": COLORS[3]},
    }
    
    final_maes_data = []
    improvements_data = []
    names = list(data.keys())
    
    for name, df in data.items():
        final_maes = df.groupby('run')['best_mae'].last().values * 1000
        final_maes_data.append(final_maes)
        
        imps = []
        for run_id in df['run'].unique():
            df_run = df[df['run'] == run_id].sort_values('step')
            initial = df_run['mae'].iloc[0]
            final = df_run['best_mae'].iloc[-1]
            if initial > 0:
                imps.append((initial - final) / initial * 100)
        improvements_data.append(imps)
    
    # Plot 1: Final MAE
    ax1 = axes[0]
    bp1 = ax1.boxplot(final_maes_data, tick_labels=names, patch_artist=True)
    for patch, name in zip(bp1['boxes'], names):
        patch.set_facecolor(styles[name]['color'])
        patch.set_alpha(0.6)
    ax1.set_ylabel("Final MAE (mm)")
    ax1.set_title("Final Beam Error")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Improvement
    ax2 = axes[1]
    bp2 = ax2.boxplot(improvements_data, tick_labels=names, patch_artist=True)
    for patch, name in zip(bp2['boxes'], names):
        patch.set_facecolor(styles[name]['color'])
        patch.set_alpha(0.6)
    ax2.set_ylabel("Improvement (%)")
    ax2.set_title("Beam Error Improvement")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Bar chart
    ax3 = axes[2]
    medians = [np.median(fm) for fm in final_maes_data]
    stds = [np.std(fm) for fm in final_maes_data]
    x_pos = np.arange(len(names))
    bar_colors = [styles[n]['color'] for n in names]
    bars = ax3.bar(x_pos, medians, yerr=stds, capsize=5, color=bar_colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylabel("Median Final MAE (mm)")
    ax3.set_title("Median Performance")
    ax3.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    
    return fig


def create_summary_figure(
    data: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 10),
) -> plt.Figure:
    """Create a comprehensive summary figure with multiple panels."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    styles = {
        "Nelder-Mead": {"ls": "-", "color": COLORS[0], "lw": 2},
        "BO (zero mean)": {"ls": "-.", "color": COLORS[1], "lw": 2},
        "BO_prior (mismatched)": {"ls": "--", "color": COLORS[2], "lw": 2.5},
        "BO_prior (matched)": {"ls": ":", "color": COLORS[3], "lw": 2.5},
    }
    
    names = list(data.keys())
    
    # Panel A: Convergence
    ax1 = fig.add_subplot(gs[0, 0])
    for name, df in data.items():
        style = styles[name]
        sns.lineplot(data=df, x="step", y="best_mae", ax=ax1, label=name, **style)
   # all_best = min(df['best_mae'].min() for df in data.values())
    #ax1.axhline(y=all_best, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Minimum MAE (m)")
    ax1.set_title("A) Convergence Curves")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Final MAE boxplot
    ax2 = fig.add_subplot(gs[0, 1])
    final_maes_data = [df.groupby('run')['best_mae'].last().values * 1000 for df in data.values()]
    bp = ax2.boxplot(final_maes_data, tick_labels=[n.replace(' ', '\n') for n in names], patch_artist=True)
    for patch, name in zip(bp['boxes'], names):
        patch.set_facecolor(styles[name]['color'])
        patch.set_alpha(0.6)
    ax2.set_ylabel("Final MAE (mm)")
    ax2.set_title("B) Final Error Distribution")
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel C: Improvement
    ax3 = fig.add_subplot(gs[1, 0])
    improvements_data = []
    for name, df in data.items():
        imps = []
        for run_id in df['run'].unique():
            df_run = df[df['run'] == run_id].sort_values('step')
            initial = df_run['mae'].iloc[0]
            final = df_run['best_mae'].iloc[-1]
            if initial > 0:
                imps.append((initial - final) / initial * 100)
        improvements_data.append(imps)
    bp2 = ax3.boxplot(improvements_data, tick_labels=[n.replace(' ', '\n') for n in names], patch_artist=True)
    for patch, name in zip(bp2['boxes'], names):
        patch.set_facecolor(styles[name]['color'])
        patch.set_alpha(0.6)
    ax3.set_ylabel("Improvement (%)")
    ax3.set_title("C) Beam Error Improvement")
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel D: Bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    medians = [np.median(fm) for fm in final_maes_data]
    stds = [np.std(fm) for fm in final_maes_data]
    x_pos = np.arange(len(names))
    bar_colors = [styles[n]['color'] for n in names]
    bars = ax4.bar(x_pos, medians, yerr=stds, capsize=5, color=bar_colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([n.replace(' ', '\n') for n in names])
    ax4.set_ylabel("Median Final MAE (mm)")
    ax4.set_title("D) Summary Comparison")
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, med in zip(bars, medians):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{med:.3f}', ha='center', va='bottom', fontsize=8)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    
    return fig


def plot_final_mae_vs_convergence(
    data: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    convergence_threshold: float = 1e-6,
) -> plt.Figure:
    """Plot final MAE vs steps to convergence - Fig 4.10 style."""
    n_methods = len(data)
    ncols = min(2, n_methods)
    nrows = (n_methods + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    colors = {
        "Nelder-Mead": COLORS[0],
        "BO (zero mean)": COLORS[1],
        "BO_prior (mismatched)": COLORS[2],
        "BO_prior (matched)": COLORS[3],
    }
    
    for idx, (name, df) in enumerate(data.items()):
        ax = axes[idx]
        color = colors.get(name, COLORS[idx % len(COLORS)])
        
        final_maes = []
        steps_to_conv = []
        
        for run_id in df['run'].unique():
            df_run = df[df['run'] == run_id].sort_values('step')
            best_mae_history = df_run['best_mae'].values
            
            final_maes.append(best_mae_history[-1])
            
            conv_step = len(best_mae_history) - 1
            for i in range(1, len(best_mae_history)):
                future = best_mae_history[i:]
                if np.max(future) - np.min(future) < convergence_threshold:
                    conv_step = i
                    break
            steps_to_conv.append(conv_step)
        
        final_maes = np.array(final_maes)
        steps_to_conv = np.array(steps_to_conv)
        
        ax.scatter(steps_to_conv, final_maes * 1000, alpha=0.6, s=50, c=color, label=name)
        
        ax.set_xlabel("Steps to Convergence", fontsize=10)
        ax.set_ylabel("Final MAE (mm)", fontsize=10)
        ax.set_title(name, fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        stats_text = f"Median: {np.median(final_maes)*1000:.3f} mm\nMean: {np.mean(final_maes)*1000:.3f} mm"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for idx in range(len(data), len(axes)):
        axes[idx].set_visible(False)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ARES optimization plots")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="results/")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    data = load_data(args.data_dir)
    
    if len(data) == 0:
        print("No data files found!")
        exit(1)
    
    print("\nGenerating plots...")
    
    plot_convergence_curves(data, save_path=os.path.join(args.output_dir, "convergence_curves.pdf"))
    plot_convergence_curves(data, save_path=os.path.join(args.output_dir, "convergence_curves.png"))
    plot_boxplot_comparison(data, save_path=os.path.join(args.output_dir, "boxplot_comparison.pdf"))
    create_summary_figure(data, save_path=os.path.join(args.output_dir, "summary_figure.pdf"))
    create_summary_figure(data, save_path=os.path.join(args.output_dir, "summary_figure.png"))
    plot_final_mae_vs_convergence(data, save_path=os.path.join(args.output_dir, "mae_vs_convergence.pdf"))
    
    print("\n✓ All plots generated!")

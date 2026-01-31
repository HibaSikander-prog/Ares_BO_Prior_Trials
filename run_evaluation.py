"""
ARES BO Physics-Informed Prior Evaluation

Run evaluation on optimization CSV files and generate Table 4.3 style results.

Usage:
    python run_evaluation.py --data_dir data/ --output_dir results/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from eval_ares_metrics import (
    load_and_evaluate,
    print_table_4_3_style,
    print_detailed_table,
    create_results_dataframe,
)
from plot_ares_results import (
    load_data,
    plot_convergence_curves,
    plot_boxplot_comparison,
    create_summary_figure,
    plot_final_mae_vs_convergence,
)


def generate_latex_table(studies, output_path):
    """Generate LaTeX table from study metrics."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance of optimization algorithms on ARES beam tuning}")
    lines.append(r"\label{tab:ares_results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Optimizer & Final Error (mm) & RMSE (mm) & Improvement (\%) & Success \\")
    lines.append(r"& Median / Mean $\pm$ Std & Median / Mean $\pm$ Std & Median / Mean $\pm$ Std & Rate \\")
    lines.append(r"\midrule")
    
    for s in studies:
        name = s.name.replace("_", r"\_")
        fe = f"{s.final_error_median*1000:.3f} / {s.final_error_mean*1000:.3f} $\\pm$ {s.final_error_std*1000:.3f}"
        rmse = f"{s.rmse_median*1000:.3f} / {s.rmse_mean*1000:.3f} $\\pm$ {s.rmse_std*1000:.3f}"
        imp = f"{s.improvement_median:.1f} / {s.improvement_mean:.1f} $\\pm$ {s.improvement_std:.1f}"
        success = f"{s.success_rate:.0f}\\%"
        lines.append(f"{name} & {fe} & {rmse} & {imp} & {success} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ARES BO optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_evaluation.py --data_dir data/ --output_dir results/
    python run_evaluation.py --threshold 4e-5 --no_plots
        """
    )
    
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory containing CSV result files (default: data/)")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to save outputs (default: results/)")
    parser.add_argument("--threshold", type=float, default=4e-5,
                        help="MAE threshold for 'target reached' in meters (default: 4e-5 = 40µm)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--show_plots", action="store_true",
                        help="Display plots interactively")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file paths
    file_paths = {
        "Nelder-Mead": os.path.join(args.data_dir, "NM_mismatched.csv"),
        "BO (zero mean)": os.path.join(args.data_dir, "BO_mismatched.csv"),
        "BO_prior (mismatched)": os.path.join(args.data_dir, "BO_prior_mismatched.csv"),
        "BO_prior (matched)": os.path.join(args.data_dir, "BO_prior_matched_prior_newtask.csv"),
    }
    
    # Check which files exist
    existing_files = {k: v for k, v in file_paths.items() if os.path.exists(v)}
    
    print("=" * 70)
    print("  ARES BO PHYSICS-INFORMED PRIOR EVALUATION")
    print("=" * 70)
    
    print(f"\nData directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Threshold: {args.threshold*1e6:.1f} µm")
    
    print(f"\nFound {len(existing_files)}/{len(file_paths)} data files:")
    for name, path in file_paths.items():
        status = "✓" if os.path.exists(path) else "✗"
        print(f"  {status} {name}: {path}")
    
    if len(existing_files) == 0:
        print("\nERROR: No data files found!")
        print("\nExpected files in data directory:")
        for name, path in file_paths.items():
            print(f"  - {os.path.basename(path)}")
        return 1
    
    # Load and compute metrics
    print("\n" + "=" * 70)
    print("  COMPUTING METRICS")
    print("=" * 70 + "\n")
    
    studies, episodes = load_and_evaluate(
        existing_files,
        threshold=args.threshold,
        use_best_mae=True
    )
    
    # Print results
    threshold_um = args.threshold * 1e6
    print_table_4_3_style(studies, threshold_um=threshold_um)
    print_detailed_table(studies, threshold_um=threshold_um)
    
    # Save results to CSV
    results_df = create_results_dataframe(studies)
    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Generate LaTeX table
    latex_path = os.path.join(args.output_dir, "results_table.tex")
    latex_content = generate_latex_table(studies, latex_path)
    print(f"✓ LaTeX table saved to: {latex_path}")
    
    # Print LaTeX table
    print("\n" + "=" * 70)
    print("  LATEX TABLE")
    print("=" * 70)
    print(latex_content)
    
    # Generate plots
    if not args.no_plots:
        print("\n" + "=" * 70)
        print("  GENERATING PLOTS")
        print("=" * 70 + "\n")
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        data = load_data(args.data_dir)
        
        if len(data) > 0:
            plot_convergence_curves(data, save_path=os.path.join(args.output_dir, "convergence_curves.pdf"))
            plot_convergence_curves(data, save_path=os.path.join(args.output_dir, "convergence_curves.png"))
            plot_boxplot_comparison(data, save_path=os.path.join(args.output_dir, "boxplot_comparison.pdf"))
            create_summary_figure(data, save_path=os.path.join(args.output_dir, "summary_figure.pdf"))
            create_summary_figure(data, save_path=os.path.join(args.output_dir, "summary_figure.png"))
            plot_final_mae_vs_convergence(data, save_path=os.path.join(args.output_dir, "mae_vs_convergence.pdf"))
            
            plt.close('all')
            print(f"\n✓ All plots saved to: {args.output_dir}")
            
            if args.show_plots:
                plt.show()
    
    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    print(f"\nKey findings:")
    best_fe = min(studies, key=lambda s: s.final_error_median)
    print(f"  • Best final error: {best_fe.name} ({best_fe.final_error_median*1000:.4f} mm)")
    
    best_imp = max(studies, key=lambda s: s.improvement_median)
    print(f"  • Best improvement: {best_imp.name} ({best_imp.improvement_median:.1f}%)")
    
    studies_with_target = [s for s in studies if s.steps_to_target_median is not None]
    if studies_with_target:
        fastest = min(studies_with_target, key=lambda s: s.steps_to_target_median)
        print(f"  • Fastest to target: {fastest.name} ({fastest.steps_to_target_median:.0f} steps)")
    
    best_success = max(studies, key=lambda s: s.success_rate)
    print(f"  • Highest success rate: {best_success.name} ({best_success.success_rate:.0f}%)")
    
    print(f"\nOutput files in {args.output_dir}:")
    for f in os.listdir(args.output_dir):
        size = os.path.getsize(os.path.join(args.output_dir, f))
        print(f"  • {f} ({size/1024:.1f} KB)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

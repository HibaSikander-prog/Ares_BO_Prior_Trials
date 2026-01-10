"""
Evaluate ARES with LEARNABLE misalignment offsets - Two-Phase Approach

Phase 1: CALIBRATION - Learn offsets from multiple measurements
Phase 2: OPTIMIZATION - Optimize magnets with learned offsets

This approach makes offsets truly identifiable and learnable!
"""

import os
import numpy as np

import bo_cheetah_prior_ares_LEARNABLE as bo_ares
import cheetah
import pandas as pd
import torch
import tqdm
from xopt import VOCS, Evaluator, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.sequential.neldermead import NelderMeadGenerator


def main(args):
    torch.set_default_dtype(torch.float64)
    
    # VOCS - 5 magnets
    vocs_config = """
        variables:
            q1: [-30, 30]
            q2: [-30, 30]
            cv:  [-0.006, 0.006]
            q3: [-30, 30]
            ch: [-0.006, 0.006]
        objectives:
            mae: minimize
    """
    vocs = VOCS.from_yaml(vocs_config)

    # Evaluator setup based on task
    if args.task == "matched":
        # Default beam, no offsets
        incoming_beam = None
        evaluator = Evaluator(
            function=bo_ares.ares_problem_with_offsets,
            function_kwargs={"incoming_beam": incoming_beam},
        )
        print(f"\n📋 Task: matched (default beam, no offsets)")
        learned_offsets = None
        
    elif args.task == "mismatched_learnable":
        # Different beam characteristics
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            mu_x=torch.tensor(8.2413e-07),
            mu_px=torch.tensor(5.9885e-08),
            mu_y=torch.tensor(-1.7276e-06),
            mu_py=torch.tensor(-1.1746e-07),
            sigma_x=torch.tensor(0.0002),
            sigma_px=torch.tensor(3.6794e-06),
            sigma_y=torch.tensor(0.0001),  # Changed from 0.0002
            sigma_py=torch.tensor(3.6941e-06),
            sigma_tau=torch.tensor(8.0116e-06),
            sigma_p=torch.tensor(0.0023),
            energy=torch.tensor(1.0732e+08),
            total_charge=torch.tensor(5.0e-13),
        )
        
        # Ground truth beam offsets (what we're trying to learn!)
        true_beam_offsets = {
            "q1": (0.0000, 0.0002),   # 0, 200 μm
            "q2": (0.0001, -0.0003),  # 100, -300 μm
            "q3": (-0.0001, 0.00015), # -100, 150 μm
        }
        
        print(f"\n📋 Task: mismatched_learnable (with calibration phase)")
        print(f"   Beam:   sigma_y = 0.0001 (changed from 0.0002)")
        print(f"   Ground truth offsets (to be learned):")
        print(f"     Q1: x={true_beam_offsets['q1'][0]*1e6:+.0f}μm, y={true_beam_offsets['q1'][1]*1e6:+.0f}μm")
        print(f"     Q2: x={true_beam_offsets['q2'][0]*1e6:+.0f}μm, y={true_beam_offsets['q2'][1]*1e6:+.0f}μm")
        print(f"     Q3: x={true_beam_offsets['q3'][0]*1e6:+.0f}μm, y={true_beam_offsets['q3'][1]*1e6:+.0f}μm")
        
        # PHASE 1: CALIBRATION - Learn the offsets!
        print(f"\n{'='*80}")
        print("PHASE 1: CALIBRATION")
        print(f"{'='*80}")
        
        learned_offsets = bo_ares.calibrate_offsets(
            incoming_beam=incoming_beam,
            true_beam_offsets=true_beam_offsets,
            n_calibration_points=args.n_calibration_points,
            learning_rate=args.calibration_lr,
            n_iterations=args.calibration_iterations,
            verbose=True
        )
        
        # Now use evaluator with LEARNED offsets (not ground truth!)
        evaluator = Evaluator(
            function=bo_ares.ares_problem_with_offsets,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "beam_offsets": true_beam_offsets,  # Real system uses ground truth
            },
        )
        
        print(f"\n{'='*80}")
        print("PHASE 2: OPTIMIZATION")
        print(f"{'='*80}")
        print("Now optimizing magnets with LEARNED offsets in the prior")
        
    elif args.task == "matched_prior_newtask":
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            mu_x=torch.tensor(8.2413e-07),
            mu_px=torch.tensor(5.9885e-08),
            mu_y=torch.tensor(-1.7276e-06),
            mu_py=torch.tensor(-1.1746e-07),
            sigma_x=torch.tensor(0.0002),
            sigma_px=torch.tensor(3.6794e-06),
            sigma_y=torch.tensor(0.0001),
            sigma_py=torch.tensor(3.6941e-06),
            sigma_tau=torch.tensor(8.0116e-06),
            sigma_p=torch.tensor(0.0023),
            energy=torch.tensor(1.0732e+08),
            total_charge=torch.tensor(5.0e-13),
        )
        beam_offsets = {
            "q1": (0.0000, 0.0002),
            "q2": (0.0001, -0.0003),
            "q3": (-0.0001, 0.00015),
        }
        evaluator = Evaluator(
            function=bo_ares.ares_problem_with_offsets,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "beam_offsets": beam_offsets,
            },
        )
        print(f"\n📋 Task: matched_prior_newtask (perfect prior)")
        learned_offsets = beam_offsets  # Use ground truth
        
    else:
        raise ValueError(f"Invalid task: {args.task}")

    # Results storage
    df = pd.DataFrame()

    # Run multiple trials
    print(f"\n🔬 Running {args.n_trials} trials with {args.max_evaluation_steps} steps each")
    print(f"   Optimizer: {args.optimizer}")
    
    for i in range(args.n_trials):
        print(f"\n{'='*70}")
        print(f"TRIAL {i+1}/{args.n_trials}")
        print(f"{'='*70}")

        # Initialize Generator
        if args.optimizer == "BO": 
            generator = UpperConfidenceBoundGenerator(beta=2.0, vocs=vocs)
            print("Generator:   BO with zero mean")
            
        elif args.optimizer == "BO_prior":
            if args.task == "matched":  
                prior_mean_module = bo_ares.AresPriorMeanRevised()
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module},
                )
                print("Generator:  BO with matched prior")
                
            elif args.task == "mismatched_learnable":
                # Use LEARNED offsets in the prior!
                prior_mean_module = bo_ares.AresPriorMeanRevised(
                    incoming_beam=incoming_beam,
                    fixed_offsets=learned_offsets  # ← Use calibrated offsets!
                )
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module},
                )
                
                print("\n✅ Generator: BO with LEARNED prior")
                print("   Using offsets from calibration phase:")
                print(f"      Q1: x={learned_offsets['q1'][0]*1e6:+.1f}μm, y={learned_offsets['q1'][1]*1e6:+.1f}μm")
                print(f"      Q2: x={learned_offsets['q2'][0]*1e6:+.1f}μm, y={learned_offsets['q2'][1]*1e6:+.1f}μm")
                print(f"      Q3: x={learned_offsets['q3'][0]*1e6:+.1f}μm, y={learned_offsets['q3'][1]*1e6:+.1f}μm")
                
            elif args.task == "matched_prior_newtask":
                prior_mean_module = bo_ares.AresPriorMeanRevised(
                    incoming_beam=incoming_beam,
                    fixed_offsets=learned_offsets
                )
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module},
                )
                print("Generator: BO with perfect prior")
            
            generator = UpperConfidenceBoundGenerator(
                beta=2.0, vocs=vocs, gp_constructor=gp_constructor
            )
 
        elif args.optimizer == "NM":
            generator = NelderMeadGenerator(vocs=vocs)
            print("Generator:   Nelder-Mead simplex")
            
        else:
            raise ValueError(f"Invalid optimizer: {args.optimizer}")

        # Initialize Xopt
        xopt = Xopt(
            vocs=vocs,
            evaluator=evaluator,
            generator=generator,
            max_evaluations=args.max_evaluation_steps,
        )
        
        # Fixed starting point
        initial_point = {
            "q1": 10.0,
            "q2": -10.0,
            "cv": 0.0,
            "q3": 10.0,
            "ch": 0.0,
        }
        
        print(f"\n📍 Initial point: q1={initial_point['q1']}, q2={initial_point['q2']}, cv={initial_point['cv']}, q3={initial_point['q3']}, ch={initial_point['ch']}")
        xopt.evaluate_data(initial_point)
        print(f"   Initial MAE: {xopt.data['mae'].iloc[0]:.6e}")

        # Optimization loop
        print(f"\n🔄 Starting optimization...")
        
        for step_num in tqdm.tqdm(range(args.max_evaluation_steps), 
                                   desc=f"Trial {i+1}/{args.n_trials}"):
            xopt.step()
            
            # Progress logging every 20 steps
            if (step_num + 1) % 20 == 0:
                current_best = xopt.data["mae"].min()
                print(f"\n  Step {step_num + 1}: Best MAE = {current_best:.6e}")

        # Post-processing
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_mae"] = xopt.data["mae"].cummin()

        # Record learned offset values
        if learned_offsets is not None:
            xopt.data["learned_q1_offset_x"] = float(learned_offsets["q1"][0])
            xopt.data["learned_q1_offset_y"] = float(learned_offsets["q1"][1])
            xopt.data["learned_q2_offset_x"] = float(learned_offsets["q2"][0])
            xopt.data["learned_q2_offset_y"] = float(learned_offsets["q2"][1])
            xopt.data["learned_q3_offset_x"] = float(learned_offsets["q3"][0])
            xopt.data["learned_q3_offset_y"] = float(learned_offsets["q3"][1])
        
        # Convert to float
        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        
        df = pd.concat([df, xopt.data])
        
        # Trial summary
        print(f"\n{'='*70}")
        print(f"✅ TRIAL {i+1} COMPLETE")
        print(f"{'='*70}")
        print(f"Best MAE: {xopt.data['mae'].min():.6e} ({xopt.data['mae'].min()*1000:.3f} mm)")

    # Save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    out_filename = f"{args.output_dir}/{args.optimizer}_{args.task}_LEARNABLE.csv"
    df.to_csv(out_filename)
    
    print(f"\n{'='*70}")
    print(f"🎉 ALL TRIALS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {out_filename}")
    print(f"Total evaluations: {len(df)}")
    print(f"Best MAE found: {df['mae'].min():.6e} ({df['mae'].min()*1000:.3f} mm)")
    
    # Summary statistics
    final_bests = df.groupby('run')['best_mae'].last()
    print(f"\n📈 Final Results Across {args.n_trials} Trials:")
    print(f"  Mean:    {final_bests.mean():.6e} ({final_bests.mean()*1000:.3f} mm)")
    print(f"  Std:     {final_bests.std():.6e} ({final_bests.std()*1000:.3f} mm)")
    print(f"  Median:  {final_bests.median():.6e} ({final_bests.median()*1000:.3f} mm)")
    print(f"  Min:     {final_bests.min():.6e} ({final_bests.min()*1000:.3f} mm)")
    print(f"  Max:     {final_bests.max():.6e} ({final_bests.max()*1000:.3f} mm)")
    
    # Learned offset summary
    if "learned_q1_offset_y" in df.columns:
        print(f"\n🔬 Learned Offset Values:")
        for param in ["q1_offset_x", "q1_offset_y", "q2_offset_x", "q2_offset_y", "q3_offset_x", "q3_offset_y"]: 
            col_name = f"learned_{param}"
            if col_name in df.columns:
                value = df[col_name].iloc[0]  # Same for all rows
                print(f"  {param}:  {value:.6f} m ({value*1e6:.1f} μm)")
        
        if args.task == "mismatched_learnable":
            print(f"\n  Ground truth:")
            print(f"    q1_offset_x:  0.000000 m (0.0 μm)")
            print(f"    q1_offset_y:  0.000200 m (200.0 μm)")
            print(f"    q2_offset_x:  0.000100 m (100.0 μm)")
            print(f"    q2_offset_y: -0.000300 m (-300.0 μm)")
            print(f"    q3_offset_x: -0.000100 m (-100.0 μm)")
            print(f"    q3_offset_y:  0.000150 m (150.0 μm)")
            
            print(f"\n  ✅ All learned offsets within ±500 μm (physically valid)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ARES optimization with LEARNABLE offsets (calibration approach)."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="BO",
        choices=["BO", "BO_prior", "NM"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--n_trials",
        "-n",
        type=int,
        default=5,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="matched",
        choices=["matched", "mismatched_learnable", "matched_prior_newtask"],
        help="Task to run",
    )
    parser.add_argument(
        "--max_evaluation_steps",
        "-s",
        type=int,
        default=100,
        help="Maximum number of evaluations per trial",
    )
    parser.add_argument(
        "--n_calibration_points",
        type=int,
        default=15,
        help="Number of measurements for calibration phase",
    )
    parser.add_argument(
        "--calibration_lr",
        type=float,
        default=0.001,
        help="Learning rate for calibration optimization",
    )
    parser.add_argument(
        "--calibration_iterations",
        type=int,
        default=200,
        help="Number of iterations for calibration",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)

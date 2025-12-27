"""Evaluate the ARES optimization task - REVISED VERSION

Uses FODO-style approach with beam offsets instead of misalignment attribute.

task modes:
    - matched: Optimize the ARES lattice for a matched beam (no offsets).
    - mismatched: Optimize with beam offsets simulating misalignments.
    - matched_prior_newtask: Optimize with prior that knows the true offsets.
"""

import os
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
import numpy as np

import bo_cheetah_prior_ares as bo_ares
import cheetah
import pandas as pd
import torch
import tqdm
from xopt import VOCS, Evaluator, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.sequential.neldermead import NelderMeadGenerator


def main(args):
<<<<<<< Updated upstream
    # VOCS - 5 magnets:  3 quadrupoles + 2 correctors
=======
    # VOCS - 5 magnets
>>>>>>> Stashed changes
    vocs_config = """
        variables:
            q1: [-30, 30]
            q2: [-30, 30]
            cv: [-0.006, 0.006]
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
        
    elif args.task == "mismatched":
        # Different beam characteristics
        incoming_beam = cheetah.ParameterBeam.from_parameters(
<<<<<<< Updated upstream
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(1e-3),  # Changed from 2e-3 to 1e-3
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
        # Different misalignments
        misalignment_config = {
            "AREAMQZM1": (0.0003, -0.0002),   # 0.3mm, -0.2mm
            "AREAMQZM2": (-0.0001, 0.00025),  # -0.1mm, 0.25mm
            "AREAMQZM3": (0.00015, -0.0003),  # 0.15mm, -0.3mm
=======
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
        # Beam offsets simulating misalignments
        beam_offsets = {
            "q1": (0.0000, 0.0002),   # 0, 200 μm
            "q2": (0.0001, -0.0003),  # 100, -300 μm
            "q3": (-0.0001, 0.00015), # -100, 150 μm
>>>>>>> Stashed changes
        }
        evaluator = Evaluator(
            function=bo_ares.ares_problem_with_offsets,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "beam_offsets": beam_offsets,
            },
        )
<<<<<<< Updated upstream
    elif args.task == "matched_prior_newtask": 
        # Same as mismatched but prior will be initialized correctly
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(1e-3),
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
        misalignment_config = {
            "AREAMQZM1": (0.0003, -0.0002),
            "AREAMQZM2": (-0.0001, 0.00025),
            "AREAMQZM3": (0.00015, -0.0003),
=======
        print(f"\n📋 Task: mismatched")
        print(f"   Beam: sigma_y = 0.0001 (changed from 0.0002)")
        print(f"   Beam offsets: Q1=(0, 200μm), Q2=(100, -300μm), Q3=(-100, 150μm)")
        
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
>>>>>>> Stashed changes
        }
        evaluator = Evaluator(
            function=bo_ares.ares_problem_with_offsets,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "beam_offsets": beam_offsets,
            },
        )
        print(f"\n📋 Task: matched_prior_newtask (prior knows true offsets)")
        
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
            print("Generator: BO with zero mean")
            
        elif args.optimizer == "BO_prior":
<<<<<<< Updated upstream
            if args.task == "matched": 
                # Matched case:  prior with default beam, no misalignments, not trainable
                prior_mean_module = bo_cheetah_prior.AresPriorMean()
                # Initialize all misalignments to zero (default)
                prior_mean_module.q1_misalign_x = 0.0
                prior_mean_module.q1_misalign_y = 0.0
                prior_mean_module.q2_misalign_x = 0.0
                prior_mean_module.q2_misalign_y = 0.0
                prior_mean_module.q3_misalign_x = 0.0
                prior_mean_module.q3_misalign_y = 0.0
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )
                
            elif args.task == "mismatched":
                # Mismatched case: prior trained on "old" problem, allow training
                prior_mean_module = bo_cheetah_prior.AresPriorMean()
                # Initialize with "old" misalignments (zero)
                prior_mean_module.q1_misalign_x = 0.0
                prior_mean_module.q1_misalign_y = 0.0
                prior_mean_module.q2_misalign_x = 0.0
                prior_mean_module.q2_misalign_y = 0.0
                prior_mean_module.q3_misalign_x = 0.0
                prior_mean_module.q3_misalign_y = 0.0
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae":  prior_mean_module},
                    trainable_mean_keys=["mae"],  # Allow the prior to adapt
=======
            if args.task == "matched":
                prior_mean_module = bo_ares.AresPriorMeanRevised()
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )
                print("Generator: BO with matched prior (not trainable)")
                
            elif args.task == "mismatched":
                prior_mean_module = bo_ares.AresPriorMeanRevised(
                    incoming_beam=incoming_beam
                )
                # Initialize with ZERO (wrong guess - will learn)
                prior_mean_module.q1_offset_x = 0.0
                prior_mean_module.q1_offset_y = 0.0
                prior_mean_module.q2_offset_x = 0.0
                prior_mean_module.q2_offset_y = 0.0
                prior_mean_module.q3_offset_x = 0.0
                prior_mean_module.q3_offset_y = 0.0
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module},
                    trainable_mean_keys=["mae"],  # Make it trainable!
                    use_low_noise_prior=False,
>>>>>>> Stashed changes
                )
                print("Generator: BO with mismatched prior (trainable)")
                print("  Initial offsets: all zeros (will learn true values)")
                
            elif args.task == "matched_prior_newtask":
<<<<<<< Updated upstream
                # Matched prior for new task:  correct beam and misalignments
                incoming_beam = cheetah.ParameterBeam.from_parameters(
                    sigma_x=torch.tensor(1e-4),
                    sigma_y=torch.tensor(1e-3),
                    sigma_px=torch.tensor(1e-4),
                    sigma_py=torch.tensor(1e-4),
                    energy=torch.tensor(100e6),
=======
                incoming_beam_prior = cheetah.ParameterBeam.from_parameters(
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
>>>>>>> Stashed changes
                )
                prior_mean_module = bo_ares.AresPriorMeanRevised(
                    incoming_beam=incoming_beam_prior
                )
<<<<<<< Updated upstream
                # Initialize with correct misalignments
                prior_mean_module.q1_misalign_x = 0.0003
                prior_mean_module.q1_misalign_y = -0.0002
                prior_mean_module.q2_misalign_x = -0.0001
                prior_mean_module.q2_misalign_y = 0.00025
                prior_mean_module.q3_misalign_x = 0.00015
                prior_mean_module.q3_misalign_y = -0.0003
=======
                # Initialize with CORRECT values
                prior_mean_module.q1_offset_x = 0.0000
                prior_mean_module.q1_offset_y = 0.0002
                prior_mean_module.q2_offset_x = 0.0001
                prior_mean_module.q2_offset_y = -0.0003
                prior_mean_module.q3_offset_x = -0.0001
                prior_mean_module.q3_offset_y = 0.00015
>>>>>>> Stashed changes
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )
                print("Generator: BO with matched prior (correct values, not trainable)")
            
            generator = UpperConfidenceBoundGenerator(
                beta=2.0, vocs=vocs, gp_constructor=gp_constructor
            )
            
        elif args.optimizer == "NM":
            generator = NelderMeadGenerator(vocs=vocs)
            print("Generator: Nelder-Mead simplex")
            
        else:
            raise ValueError(f"Invalid optimizer: {args.optimizer}")

        # Initialize Xopt
        xopt = Xopt(
            vocs=vocs,
            evaluator=evaluator,
            generator=generator,
            max_evaluations=args.max_evaluation_steps,
        )
        
<<<<<<< Updated upstream
         # Evaluate 5 initial points
        # Point 1: Fixed starting point
        print(f"  Evaluating initial point 1/5 (fixed)")
        xopt.evaluate_data({
=======
        # Fixed starting point
        initial_point = {
>>>>>>> Stashed changes
            "q1": 10.0,
            "q2": -10.0,
            "q3": 10.0,
<<<<<<< Updated upstream
            "cv": 0.0,
            "ch":  0.0,
        })
        
        # Points 2-5: Random starting points
        np.random.seed(args.seed + i)  # Different seed for each trial
        for j in range(4):
            print(f"  Evaluating initial point {j+2}/5 (random)")
            random_point = {
                "q1": np.random.uniform(-30, 30),
                "q2": np.random.uniform(-30, 30),
                "q3":  np.random.uniform(-30, 30),
                "cv":  np.random.uniform(-0.006, 0.006),
                "ch": np.random. uniform(-0.006, 0.006),
            }
            xopt.evaluate_data(random_point)
        
        # Start Optimization
        print(f"  Starting optimization...")
        
        # Start Optimization
        for _ in tqdm.tqdm(range(args.max_evaluation_steps)):
            xopt.step()

        # Post processing the dataframes
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_mae"] = xopt.data["mae"].cummin()
        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        df = pd.concat([df, xopt.data])
    
    # Check if the output directory exists
=======
            "ch": 0.0,
        }
        
        print(f"\n📍 Initial point: q1={initial_point['q1']}, q2={initial_point['q2']}, ...")
        xopt.evaluate_data(initial_point)
        print(f"   Initial MAE: {xopt.data['mae'].iloc[0]:.6e}")

        # Optimization loop
        print(f"\n🔄 Starting optimization...")
        
        for step_num in tqdm.tqdm(range(args.max_evaluation_steps), 
                                   desc=f"Trial {i+1}/{args.n_trials}"):
            xopt.step()

        # Post-processing
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_mae"] = xopt.data["mae"].cummin()

        # ========== NEW: Save learned offsets (like FODO drift_length) ==========
        if args.optimizer == "BO_prior" and args.task in ["mismatched", "matched_prior_newtask"]:
            try:
                model = xopt.generator.model
                gp_model = model.models[0]
                learned_mean = gp_model.mean_module._model
                
                # Extract learned offset values
                xopt.data["learned_q1_offset_x"] = float(learned_mean.q1_offset_x.item())
                xopt.data["learned_q1_offset_y"] = float(learned_mean.q1_offset_y.item())
                xopt.data["learned_q2_offset_x"] = float(learned_mean.q2_offset_x.item())
                xopt.data["learned_q2_offset_y"] = float(learned_mean.q2_offset_y.item())
                xopt.data["learned_q3_offset_x"] = float(learned_mean.q3_offset_x.item())
                xopt.data["learned_q3_offset_y"] = float(learned_mean.q3_offset_y.item())
                
                print(f"\n  Final learned offsets (m):")
                print(f"    Q1: x={learned_mean.q1_offset_x.item():+.6f}, y={learned_mean.q1_offset_y.item():+.6f}")
                print(f"    Q2: x={learned_mean.q2_offset_x.item():+.6f}, y={learned_mean.q2_offset_y.item():+.6f}")
                print(f"    Q3: x={learned_mean.q3_offset_x.item():+.6f}, y={learned_mean.q3_offset_y.item():+.6f}")
                
                if args.task == "mismatched":
                    print(f"\n  Ground truth offsets (m):")
                    print(f"    Q1: x=+0.000000, y=+0.000200")
                    print(f"    Q2: x=+0.000100, y=-0.000300")
                    print(f"    Q3: x=-0.000100, y=+0.000150")
                    
                    # Calculate errors
                    errors = [
                        abs(learned_mean.q1_offset_x.item() - 0.000000),
                        abs(learned_mean.q1_offset_y.item() - 0.000200),
                        abs(learned_mean.q2_offset_x.item() - 0.000100),
                        abs(learned_mean.q2_offset_y.item() + 0.000300),
                        abs(learned_mean.q3_offset_x.item() + 0.000100),
                        abs(learned_mean.q3_offset_y.item() - 0.000150),
                    ]
                    avg_error = sum(errors) / len(errors)
                    print(f"\n  Average learning error: {avg_error*1e6:.1f} μm")
                    
            except Exception as e:
                print(f"  ⚠️  Could not save learned offsets: {e}")
                import traceback
                traceback.print_exc()
        # ========== END NEW CODE ==========
        
        # Convert to float
        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        
        df = pd.concat([df, xopt.data])
        
        # Trial summary
        print(f"\n{'='*70}")
        print(f"TRIAL {i+1} COMPLETE")
        print(f"{'='*70}")
        print(f"Best MAE: {xopt.data['mae'].min():.6e}")

    # Save results
>>>>>>> Stashed changes
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    out_filename = f"{args.output_dir}/{args.optimizer}_{args.task}_revised.csv"
    df.to_csv(out_filename)
<<<<<<< Updated upstream
    print(f"Results saved to {out_filename}")
=======
    
    print(f"\n{'='*70}")
    print(f"✅ ALL TRIALS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {out_filename}")
    print(f"Total evaluations: {len(df)}")
    print(f"Best MAE found: {df['mae'].min():.6e}")
    
    # Summary statistics
    final_bests = df.groupby('run')['best_mae'].last()
    print(f"\nFinal Results Across {args.n_trials} Trials:")
    print(f"  Mean:   {final_bests.mean():.6e}")
    print(f"  Std:    {final_bests.std():.6e}")
    print(f"  Median: {final_bests.median():.6e}")
    print(f"  Min:    {final_bests.min():.6e}")
    print(f"  Max:    {final_bests.max():.6e}")
    
    # ========== NEW: Offset learning summary ==========
    if args.optimizer == "BO_prior" and "learned_q1_offset_y" in df.columns:
        print(f"\nLearned Offset Summary:")
        final_offsets = df.groupby('run')['learned_q1_offset_y'].last()
        print(f"  Q1 Y offset: {final_offsets.mean():.6f} ± {final_offsets.std():.6f} m")
        
        if args.task == "mismatched":
            print(f"  Ground truth: 0.000200 m")
            print(f"  Mean error: {abs(final_offsets.mean() - 0.0002)*1e6:.1f} μm")
    # ========== END NEW CODE ==========
>>>>>>> Stashed changes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ARES optimization task - REVISED with FODO-style offsets."
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
        choices=["matched", "mismatched", "matched_prior_newtask"],
        help="Task to run",
    )
    parser.add_argument(
        "--max_evaluation_steps",
        "-s",
        type=int,
        default=130,
        help="Maximum number of evaluations per trial",
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
    main(args)
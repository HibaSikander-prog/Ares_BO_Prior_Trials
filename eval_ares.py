<<<<<<< Updated upstream
"""Evaluate the ARES optimization task using different optimizers.

task modes:
    - matched:  Optimize the ARES lattice for a matched beam.
    - mismatched:  Optimize ARES lattice with mismatched prior (different beam and misalignments).
    - matched_prior_newtask: Optimize with a prior matched to the new task.
=======
"""Evaluate the ARES optimization task with MISALIGNMENTS (not offsets)

Following FODO pattern exactly:
- Manual segment creation
- Learnable misalignment parameters (like drift_length)
- Direct assignment to element properties

task modes:
    - matched: Optimize with default beam, no misalignments
    - mismatched: Optimize with different beam + misalignments (prior starts wrong, learns)
    - matched_prior_newtask: Optimize with prior that has correct beam + misalignments
>>>>>>> Stashed changes
"""

import os

import numpy as np

import bo_cheetah_prior_ares as bo_cheetah_prior
import cheetah
import pandas as pd
import torch
import tqdm
from xopt import VOCS, Evaluator, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.sequential.neldermead import NelderMeadGenerator


def main(args):
    # VOCS - 5 magnets:  3 quadrupoles + 2 correctors
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
        # Default beam, no misalignments
        incoming_beam = None
        misalignment_config = None  # Will default to zeros
        evaluator = Evaluator(
<<<<<<< Updated upstream
            function=bo_cheetah_prior.ares_problem,
            function_kwargs={"incoming_beam": incoming_beam},
        )
=======
            function=bo_ares.ares_problem,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "misalignment_config": misalignment_config
            },
        )
        print(f"\n📋 Task: matched (default beam, no misalignments)")
        
>>>>>>> Stashed changes
    elif args.task == "mismatched":
        # Different beam (sigma_y changed)
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
        }
        evaluator = Evaluator(
            function=bo_cheetah_prior.ares_problem,
=======
            mu_x=torch.tensor(8.2413e-07),
            mu_px=torch.tensor(5.9885e-08),
            mu_y=torch.tensor(-1.7276e-06),
            mu_py=torch.tensor(-1.1746e-07),
            sigma_x=torch.tensor(0.0002),
            sigma_px=torch.tensor(3.6794e-06),
            sigma_y=torch.tensor(0.0001),  # CHANGED from default 0.0002
            sigma_py=torch.tensor(3.6941e-06),
            sigma_tau=torch.tensor(8.0116e-06),
            sigma_p=torch.tensor(0.0023),
            energy=torch.tensor(1.0732e+08),
            total_charge=torch.tensor(5.0e-13),
        )
        # Ground truth misalignments
        misalignment_config = {
            "AREAMQZM1": (0.0000, 0.0002),   # 0, 200 μm
            "AREAMQZM2": (0.0001, -0.0003),  # 100, -300 μm
            "AREAMQZM3": (-0.0001, 0.00015), # -100, 150 μm
        }
        evaluator = Evaluator(
            function=bo_ares.ares_problem,
>>>>>>> Stashed changes
            function_kwargs={
                "incoming_beam": incoming_beam,
                "misalignment_config": misalignment_config,
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
        }
        evaluator = Evaluator(
            function=bo_cheetah_prior.ares_problem,
=======
        print(f"\n📋 Task: mismatched")
        print(f"   Beam: sigma_y = 0.0001 (changed from 0.0002)")
        print(f"   Misalignments:")
        print(f"     Q1: ({misalignment_config['AREAMQZM1'][0]*1e6:+.0f}, {misalignment_config['AREAMQZM1'][1]*1e6:+.0f}) μm")
        print(f"     Q2: ({misalignment_config['AREAMQZM2'][0]*1e6:+.0f}, {misalignment_config['AREAMQZM2'][1]*1e6:+.0f}) μm")
        print(f"     Q3: ({misalignment_config['AREAMQZM3'][0]*1e6:+.0f}, {misalignment_config['AREAMQZM3'][1]*1e6:+.0f}) μm")
        
    elif args.task == "matched_prior_newtask":
        # Same as mismatched for evaluator
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            mu_x=torch.tensor(8.2413e-07),
            mu_px=torch.tensor(5.9885e-08),
            mu_y=torch.tensor(-1.7276e-06),
            mu_py=torch.tensor(-1.1746e-07),
            sigma_x=torch.tensor(0.0002),
            sigma_px=torch.tensor(3.6794e-06),
            sigma_y=torch.tensor(0.0001),  # CHANGED
            sigma_py=torch.tensor(3.6941e-06),
            sigma_tau=torch.tensor(8.0116e-06),
            sigma_p=torch.tensor(0.0023),
            energy=torch.tensor(1.0732e+08),
            total_charge=torch.tensor(5.0e-13),
        )
        misalignment_config = {
            "AREAMQZM1": (0.0000, 0.0002),
            "AREAMQZM2": (0.0001, -0.0003),
            "AREAMQZM3": (-0.0001, 0.00015),
        }
        evaluator = Evaluator(
            function=bo_ares.ares_problem,
>>>>>>> Stashed changes
            function_kwargs={
                "incoming_beam": incoming_beam,
                "misalignment_config": misalignment_config,
            },
        )
<<<<<<< Updated upstream
=======
        print(f"\n📋 Task: matched_prior_newtask (prior knows true beam + misalignments)")
        
>>>>>>> Stashed changes
    else:
        raise ValueError(f"Invalid task: {args.task}")

    # Empty dataframe to store results
    df = pd.DataFrame()

    # Run n_trials
    for i in range(args.n_trials):
        print(f"Trial {i+1}/{args.n_trials}")

        # Initialize Generator
        if args.optimizer == "BO":
            # Standard BO without prior
            generator = UpperConfidenceBoundGenerator(beta=2.0, vocs=vocs)
            
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
=======
            if args.task == "matched":
                # Matched case: default beam, no misalignments, NOT trainable
                prior_mean_module = bo_ares.AresPriorMean()
                # Misalignments default to 0.0 (correct for this task)
>>>>>>> Stashed changes
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                    # NOT trainable
                )
                
            elif args.task == "mismatched":
<<<<<<< Updated upstream
                # Mismatched case: prior trained on "old" problem, allow training
                prior_mean_module = bo_cheetah_prior.AresPriorMean()
                # Initialize with "old" misalignments (zero)
=======
                # Mismatched case: DEFAULT beam (wrong sigma_y), trainable misalignments
                # Prior uses DEFAULT beam (sigma_y=0.0002) which is WRONG for this task
                prior_mean_module = bo_ares.AresPriorMean()
                
                # Initialize misalignments to ZERO (wrong guess - will learn via training)
                # Like FODO: drift_length=0.5 when truth is 0.7
>>>>>>> Stashed changes
                prior_mean_module.q1_misalign_x = 0.0
                prior_mean_module.q1_misalign_y = 0.0
                prior_mean_module.q2_misalign_x = 0.0
                prior_mean_module.q2_misalign_y = 0.0
                prior_mean_module.q3_misalign_x = 0.0
                prior_mean_module.q3_misalign_y = 0.0
                
                gp_constructor = StandardModelConstructor(
<<<<<<< Updated upstream
                    mean_modules={"mae":  prior_mean_module},
                    trainable_mean_keys=["mae"],  # Allow the prior to adapt
                )
                
            elif args.task == "matched_prior_newtask":
                # Matched prior for new task:  correct beam and misalignments
                incoming_beam = cheetah.ParameterBeam.from_parameters(
                    sigma_x=torch.tensor(1e-4),
                    sigma_y=torch.tensor(1e-3),
                    sigma_px=torch.tensor(1e-4),
                    sigma_py=torch.tensor(1e-4),
                    energy=torch.tensor(100e6),
                )
                prior_mean_module = bo_cheetah_prior.AresPriorMean(
                    incoming_beam=incoming_beam
                )
                # Initialize with correct misalignments
                prior_mean_module.q1_misalign_x = 0.0003
                prior_mean_module.q1_misalign_y = -0.0002
                prior_mean_module.q2_misalign_x = -0.0001
                prior_mean_module.q2_misalign_y = 0.00025
                prior_mean_module.q3_misalign_x = 0.00015
                prior_mean_module.q3_misalign_y = -0.0003
=======
                    mean_modules={"mae": prior_mean_module},
                    trainable_mean_keys=["mae"],  # TRAINABLE - learns corrections
                    use_low_noise_prior=False,
                )
                print("Generator: BO with mismatched prior (trainable)")
                print("  Prior beam: sigma_y = 0.0002 (WRONG - should be 0.0001)")
                print("  Initial misalignments: all zeros (will learn true values)")
                
            elif args.task == "matched_prior_newtask":
                # Matched case: CORRECT beam AND CORRECT misalignments
                # Create beam matching evaluator
                incoming_beam_prior = cheetah.ParameterBeam.from_parameters(
                    mu_x=torch.tensor(8.2413e-07),
                    mu_px=torch.tensor(5.9885e-08),
                    mu_y=torch.tensor(-1.7276e-06),
                    mu_py=torch.tensor(-1.1746e-07),
                    sigma_x=torch.tensor(0.0002),
                    sigma_px=torch.tensor(3.6794e-06),
                    sigma_y=torch.tensor(0.0001),  # CORRECT (matches evaluator)
                    sigma_py=torch.tensor(3.6941e-06),
                    sigma_tau=torch.tensor(8.0116e-06),
                    sigma_p=torch.tensor(0.0023),
                    energy=torch.tensor(1.0732e+08),
                    total_charge=torch.tensor(5.0e-13),
                )
                prior_mean_module = bo_ares.AresPriorMean(
                    incoming_beam=incoming_beam_prior
                )
                
                # Set CORRECT misalignments directly on elements (like FODO)
                # NOT using property setters!
                prior_mean_module.Q1.misalignment = torch.tensor([0.0000, 0.0002], dtype=torch.float32)
                prior_mean_module.Q2.misalignment = torch.tensor([0.0001, -0.0003], dtype=torch.float32)
                prior_mean_module.Q3.misalignment = torch.tensor([-0.0001, 0.00015], dtype=torch.float32)
>>>>>>> Stashed changes
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )
<<<<<<< Updated upstream
=======
                print("Generator: BO with matched prior (correct beam + misalignments, not trainable)")
>>>>>>> Stashed changes
            
            generator = UpperConfidenceBoundGenerator(
                beta=2.0, vocs=vocs, gp_constructor=gp_constructor
            )
            
        elif args.optimizer == "NM":
            # Nelder-Mead baseline
            generator = NelderMeadGenerator(vocs=vocs)
            
        else:
            raise ValueError(f"Invalid optimizer: {args.optimizer}")

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
        # Fixed starting point (same for all trials, like FODO)
        initial_point = {
>>>>>>> Stashed changes
            "q1": 10.0,
            "q2": -10.0,
            "q3": 10.0,
            "cv": 0.0,
            "ch":  0.0,
        })
        
<<<<<<< Updated upstream
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
=======
        print(f"\n📍 Initial point: q1={initial_point['q1']}, q2={initial_point['q2']}, cv={initial_point['cv']}, q3={initial_point['q3']}, ch={initial_point['ch']}")
        xopt.evaluate_data(initial_point)
        print(f"   Initial MAE: {xopt.data['mae'].iloc[0]:.6e}")

        # Optimization loop
        print(f"\n🔄 Starting optimization...")
>>>>>>> Stashed changes
        
        # Start Optimization
        print(f"  Starting optimization...")
        
        # Start Optimization
        for _ in tqdm.tqdm(range(args.max_evaluation_steps)):
            xopt.step()

        # Post processing the dataframes
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_mae"] = xopt.data["mae"].cummin()
<<<<<<< Updated upstream
=======

        # Save learned misalignments (like FODO saves learned drift_length)
        if args.optimizer == "BO_prior" and args.task in ["mismatched", "matched_prior_newtask"]:
            try:
                model = xopt.generator.model
                gp_model = model.models[0]
                learned_mean = gp_model.mean_module._model
                
                # Extract learned values
                xopt.data["learned_q1_misalign_x"] = float(learned_mean.q1_misalign_x.item())
                xopt.data["learned_q1_misalign_y"] = float(learned_mean.q1_misalign_y.item())
                xopt.data["learned_q2_misalign_x"] = float(learned_mean.q2_misalign_x.item())
                xopt.data["learned_q2_misalign_y"] = float(learned_mean.q2_misalign_y.item())
                xopt.data["learned_q3_misalign_x"] = float(learned_mean.q3_misalign_x.item())
                xopt.data["learned_q3_misalign_y"] = float(learned_mean.q3_misalign_y.item())
                
                print(f"\n  Final learned misalignments (m):")
                print(f"    Q1: x={learned_mean.q1_misalign_x.item():+.6f}, y={learned_mean.q1_misalign_y.item():+.6f}")
                print(f"    Q2: x={learned_mean.q2_misalign_x.item():+.6f}, y={learned_mean.q2_misalign_y.item():+.6f}")
                print(f"    Q3: x={learned_mean.q3_misalign_x.item():+.6f}, y={learned_mean.q3_misalign_y.item():+.6f}")
                
                if args.task == "mismatched":
                    print(f"\n  Ground truth misalignments (m):")
                    print(f"    Q1: x=+0.000000, y=+0.000200")
                    print(f"    Q2: x=+0.000100, y=-0.000300")
                    print(f"    Q3: x=-0.000100, y=+0.000150")
                    
                    # Calculate learning errors
                    errors = [
                        abs(learned_mean.q1_misalign_x.item() - 0.000000),
                        abs(learned_mean.q1_misalign_y.item() - 0.000200),
                        abs(learned_mean.q2_misalign_x.item() - 0.000100),
                        abs(learned_mean.q2_misalign_y.item() + 0.000300),
                        abs(learned_mean.q3_misalign_x.item() + 0.000100),
                        abs(learned_mean.q3_misalign_y.item() - 0.000150),
                    ]
                    avg_error = sum(errors) / len(errors)
                    print(f"\n  Average learning error: {avg_error*1e6:.1f} μm")
                    
            except Exception as e:
                print(f"  ⚠️  Could not save learned misalignments: {e}")
                import traceback
                traceback.print_exc()
        
        # Convert to float
>>>>>>> Stashed changes
        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        df = pd.concat([df, xopt.data])
    
    # Check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
<<<<<<< Updated upstream
    out_filename = f"{args.output_dir}/{args.optimizer}_{args.task}.csv"

    df.to_csv(out_filename)
    print(f"Results saved to {out_filename}")
=======
    
    out_filename = f"{args.output_dir}/{args.optimizer}_{args.task}.csv"
    df.to_csv(out_filename)
    
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
    
    # Misalignment learning summary
    if args.optimizer == "BO_prior" and "learned_q1_misalign_y" in df.columns:
        print(f"\nLearned Misalignment Summary:")
        final_vals = df.groupby('run')['learned_q1_misalign_y'].last()
        print(f"  Q1 Y misalignment: {final_vals.mean():.6f} ± {final_vals.std():.6f} m")
        
        if args.task == "mismatched":
            print(f"  Ground truth: 0.000200 m")
            print(f"  Mean error: {abs(final_vals.mean() - 0.0002)*1e6:.1f} μm")
>>>>>>> Stashed changes


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

<<<<<<< Updated upstream
    parser = argparse.ArgumentParser(description="Run ARES optimization task.")
=======
    parser = argparse.ArgumentParser(
        description="Run ARES optimization with MISALIGNMENTS (FODO style)."
    )
>>>>>>> Stashed changes
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
        default=10,
        help="Number of trials to run for each optimizer.",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="matched",
        choices=["matched", "mismatched", "matched_prior_newtask"],
        help="Task to run.See bo_cheetah_prior.py for options.",
    )
    parser.add_argument(
        "--max_evaluation_steps",
        "-s",
        type=int,
        default=50,
        help="Maximum number of evaluations to run for each trial.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n_workers",
        "-w",
        type=int,
        default=mp.cpu_count() - 1,
        help="Number of workers to use for parallel evaluation.",
    )
    args = parser.parse_args()
    main(args)
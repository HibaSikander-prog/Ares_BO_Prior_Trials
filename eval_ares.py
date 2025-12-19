"""Evaluate the ARES optimization task using different optimizers.

task modes:
    - matched:  Optimize the ARES lattice for a matched beam.
    - mismatched:  Optimize ARES lattice with mismatched prior (different beam and misalignments).
    - matched_prior_newtask: Optimize with a prior matched to the new task.
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
    # Using 'combined' objective which includes beam size + position error
    vocs_config = """
        variables:
            q1: [-30, 30]
            q2: [-30, 30]
            cv: [-0.006, 0.006]
            q3: [-30, 30]
            ch: [-0.006, 0.006]
        objectives:
            combined: minimize
    """
    vocs = VOCS.from_yaml(vocs_config)

    # Evaluator setup based on task
    if args.task == "matched":
        # Default beam, no misalignments
        incoming_beam = None
        evaluator = Evaluator(
            function=bo_cheetah_prior.ares_problem,
            function_kwargs={"incoming_beam": incoming_beam},
        )
    elif args.task == "mismatched":
        # Different beam characteristics
        incoming_beam = cheetah.ParticleBeam.from_parameters(
            num_particles=10000,
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(1e-3),  # Changed from 2e-3 to 1e-3
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
        # Different misalignments (using larger values for better signal)
        misalignment_config = {
            "AREAMQZM1": (0.0003, -0.0002),   # 0.3mm, -0.2mm
            "AREAMQZM2": (-0.0001, 0.00025),  # -0.1mm, 0.25mm
            "AREAMQZM3": (0.00015, -0.0003),  # 0.15mm, -0.3mm
        }
        evaluator = Evaluator(
            function=bo_cheetah_prior.ares_problem,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "misalignment_config": misalignment_config,
            },
        )
    elif args.task == "matched_prior_newtask":  
        # Same as mismatched but prior will be initialized correctly
        incoming_beam = cheetah.ParticleBeam.from_parameters(
            num_particles=10000,
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
            function_kwargs={
                "incoming_beam": incoming_beam,
                "misalignment_config": misalignment_config,
            },
        )
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
                    mean_modules={"combined": prior_mean_module}
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
                    mean_modules={"combined": prior_mean_module},
                    trainable_mean_keys=["combined"], 
                     use_low_noise_prior=False,  
                )
                
            elif args.task == "matched_prior_newtask":
                # Matched prior for new task:  correct beam and misalignments
                incoming_beam = cheetah.ParticleBeam.from_parameters(
                    num_particles=10000,
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
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"combined": prior_mean_module}
                )
            
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
        
        # CRITICAL FIX: Use only 1 fixed starting point (like FODO example)
        print(f"  Evaluating initial point...")
        xopt.evaluate_data({
            "q1": 10.0,
            "q2": -10.0,
            "q3": 10.0,
            "cv": 0.0,
            "ch": 0.0,
        })
        
        # Start Optimization
        print(f"  Starting optimization...")
        for step_num in tqdm.tqdm(range(args.max_evaluation_steps)):
            xopt.step()

            # === FIX: Extract and persist learned parameters ===
            if args.optimizer == "BO_prior" and args.task == "mismatched":
                try:
                    # Access the trained GP model
                    model = xopt.generator.model
                    gp_model = model.models[0]  # First model in ModelListGP
                    custom_mean = gp_model.mean_module  # CustomMean wrapper
                    trained_prior = custom_mean._model  # Our AresPriorMean inside
                    
                    # Extract learned parameters
                    q1x = trained_prior.q1_misalign_x.item()
                    q1y = trained_prior.q1_misalign_y.item()
                    q2x = trained_prior.q2_misalign_x.item()
                    q2y = trained_prior.q2_misalign_y.item()
                    q3x = trained_prior.q3_misalign_x.item()
                    q3y = trained_prior.q3_misalign_y.item()
                    
                    # Copy back to the original prior_mean_module to persist
                    with torch.no_grad():
                        prior_mean_module.raw_q1_misalign_x.copy_(trained_prior.raw_q1_misalign_x)
                        prior_mean_module.raw_q1_misalign_y.copy_(trained_prior.raw_q1_misalign_y)
                        prior_mean_module.raw_q2_misalign_x.copy_(trained_prior.raw_q2_misalign_x)
                        prior_mean_module.raw_q2_misalign_y.copy_(trained_prior.raw_q2_misalign_y)
                        prior_mean_module.raw_q3_misalign_x.copy_(trained_prior.raw_q3_misalign_x)
                        prior_mean_module.raw_q3_misalign_y.copy_(trained_prior.raw_q3_misalign_y)
                    
                    if step_num == 0:
                        print(f"\n=== After Step {step_num + 1}:  Extracted & Persisted ===")
                        print(f"  Q1: x={q1x:.6f}, y={q1y:.6f}")
                        print(f"  Q2: x={q2x:.6f}, y={q2y:.6f}")
                        print(f"  Q3: x={q3x:.6f}, y={q3y:.6f}")
                        print("===")
                except Exception as e: 
                    print(f"Warning: Could not extract parameters:  {e}")
            # === END FIX ===
            
            # Print progress every 20 steps
            if (step_num + 1) % 10 == 0 and args.optimizer == "BO_prior" and args.task == "mismatched":
                current_best = xopt.data["combined"].min()
                print(f"    Step {step_num + 1}:  Best combined = {current_best:.6e}")
                # Also print current learned misalignments
                print(f"    Current misalignments:  Q1_x={prior_mean_module.q1_misalign_x.item():.6f}")

        # Post processing the dataframes
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_combined"] = xopt.data["combined"].cummin()

        # Also track best_mae for comparison
        if "mae" in xopt.data.columns:
            xopt.data["best_mae"] = xopt.data["mae"].cummin()

        # CRITICAL FIX: Save learned misalignments if using BO_prior
        if args.optimizer == "BO_prior" and args.task in ["mismatched", "matched_prior_newtask"]:
            # Get the current learned misalignments from the prior
            xopt.data["q1_misalign_x"] = float(prior_mean_module.q1_misalign_x.item())
            xopt.data["q1_misalign_y"] = float(prior_mean_module.q1_misalign_y.item())
            xopt.data["q2_misalign_x"] = float(prior_mean_module.q2_misalign_x.item())
            xopt.data["q2_misalign_y"] = float(prior_mean_module.q2_misalign_y.item())
            xopt.data["q3_misalign_x"] = float(prior_mean_module.q3_misalign_x.item())
            xopt.data["q3_misalign_y"] = float(prior_mean_module.q3_misalign_y.item())
        
        # Also track best_mae for comparison with old results
      #  if "mae" in xopt.data.columns:
       #     xopt.data["best_mae"] = xopt.data["mae"].cummin()
        
        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        df = pd.concat([df, xopt.data])
        
        # Debug: print learned misalignments
        if args.optimizer == "BO_prior": 
            print(f"\n  === Trial {i+1} Summary ===")
            print(f"  Best combined objective: {xopt.data['combined'].min():.6e}")
            if args.task in ["mismatched", "matched_prior_newtask"]:
                print(f"  Learned misalignments:")
                print(f"    Q1: x={prior_mean_module.q1_misalign_x.item():.6f}m, y={prior_mean_module.q1_misalign_y.item():.6f}m")
                print(f"    Q2: x={prior_mean_module.q2_misalign_x.item():.6f}m, y={prior_mean_module.q2_misalign_y.item():.6f}m")
                print(f"    Q3: x={prior_mean_module.q3_misalign_x.item():.6f}m, y={prior_mean_module.q3_misalign_y.item():.6f}m")
                if args.task == "mismatched":
                    print(f"  Ground truth misalignments:")
                    print(f"    Q1: x=0.000300m, y=-0.000200m")
                    print(f"    Q2: x=-0.000100m, y=0.000250m")
                    print(f"    Q3: x=0.000150m, y=-0.000300m")
    
    # Check if the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_filename = f"{args.output_dir}/{args.optimizer}_{args.task}.csv"

    df.to_csv(out_filename)
    print(f"\nResults saved to {out_filename}")


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Run ARES optimization task.")
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
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)
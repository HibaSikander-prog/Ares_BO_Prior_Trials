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
    # Using 'mae' objective which includes beam size + position error
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
        evaluator = Evaluator(
            function=bo_cheetah_prior.ares_problem,
            function_kwargs={"incoming_beam": incoming_beam},
        )
    elif args.task == "mismatched":
        # Different beam characteristics
        incoming_beam = cheetah.ParticleBeam.from_parameters(
            num_particles=1000,
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(1e-3),  # Changed from 2e-3 to 1e-3
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
        # Different misalignments (using larger values for better signal)
        misalignment_config = {
            "AREAMQZM1": (0.000000, 0.000100),   # 0, 100 μm
            "AREAMQZM2": (0.000030, -0.000120),  # 30, -120 μm
            "AREAMQZM3": (-0.000040, 0.000080),  # -40, 80 μm
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
            num_particles=1000,
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(1e-3),
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
        misalignment_config = {
            "AREAMQZM1": (0.000000, 0.000100),   # 0, 100 μm
            "AREAMQZM2": (0.000030, -0.000120),  # 30, -120 μm
            "AREAMQZM3": (-0.000040, 0.000080),  # -40, 80 μm
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
                # All misalignments default to 0.0 (matched to evaluator)
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                    # NOT trainable
                )
                
            elif args.task == "mismatched":
                # Mismatched case: prior starts with WRONG values, learns correct ones
                prior_mean_module = bo_cheetah_prior.AresPriorMean(
                    incoming_beam=incoming_beam
                )
                # Initialize with ZERO (wrong initial guess)
                # Like FODO's drift_length=0.5 when truth is 0.7
                prior_mean_module.q1_misalign_x = 0.0
                prior_mean_module.q1_misalign_y = 0.0
                prior_mean_module.q2_misalign_x = 0.0
                prior_mean_module.q2_misalign_y = 0.0
                prior_mean_module.q3_misalign_x = 0.0
                prior_mean_module.q3_misalign_y = 0.0

                 
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module},
                    trainable_mean_keys=["mae"], 
                     use_low_noise_prior=False,  
                )
                
            elif args.task == "matched_prior_newtask":
                # Matched prior for new task:  correct beam and misalignments
                incoming_beam = cheetah.ParticleBeam.from_parameters(
                    num_particles=1000,
                    sigma_x=torch.tensor(1e-4),
                    sigma_y=torch.tensor(1e-3),
                    sigma_px=torch.tensor(1e-4),
                    sigma_py=torch.tensor(1e-4),
                    energy=torch.tensor(100e6),
                )
                prior_mean_module = bo_cheetah_prior.AresPriorMean(
                    incoming_beam=incoming_beam
                )
                # Initialize with CORRECT values (matched to evaluator)
                # Like FODO's drift_length=0.7 when truth is 0.7
                prior_mean_module.q1_misalign_x = 0.000000
                prior_mean_module.q1_misalign_y = 0.000100
                prior_mean_module.q2_misalign_x = 0.000030
                prior_mean_module.q2_misalign_y = -0.000120
                prior_mean_module.q3_misalign_x = -0.000040
                prior_mean_module.q3_misalign_y = 0.000080
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                    # NOT trainable (already correct)
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

            # Diagnostic logging for mismatched task
            if (step_num + 1) % 10 == 0 and args.optimizer == "BO_prior" and args.task == "mismatched":
                current_best = xopt.data["mae"].min()
                print(f"\n    Step {step_num + 1}: Best mae = {current_best:.6e}")
                
                # Try to extract learned misalignments
                try: 
                    model = xopt.generator.model
                    gp_model = model.models[0]
                    learned_mean = gp_model.mean_module._model
                    
                    print(f"    Learned misalignments:")
                    print(f"      Q1: x={learned_mean.q1_misalign_x.item():.6f}, y={learned_mean.q1_misalign_y.item():.6f}")
                    print(f"      Q2: x={learned_mean.q2_misalign_x.item():.6f}, y={learned_mean.q2_misalign_y.item():.6f}")
                    print(f"      Q3: x={learned_mean.q3_misalign_x.item():.6f}, y={learned_mean.q3_misalign_y.item():.6f}")
                    print(f"    Target:")
                    print(f"      Q1: x=0.000000, y=0.000100")
                    print(f"      Q2: x=0.000030, y=-0.000120")
                    print(f"      Q3: x=-0.000040, y=0.000080")
                  
                    
                except Exception as e:
                    print(f"    (Could not extract learned values: {e})")

        # Post processing the dataframes
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_mae"] = xopt.data["mae"].cummin()

        # Also track best_mae for comparison
        if "mae" in xopt.data.columns:
            xopt.data["best_mae"] = xopt.data["mae"].cummin()

        # Save learned misalignments if using BO_prior
        if args.optimizer == "BO_prior" and args.task in ["mismatched", "matched_prior_newtask"]:
            try:
                model = xopt.generator.model
                gp_model = model.models[0]
                learned_mean = gp_model.mean_module._model
                
                # Save final learned values to dataframe
                xopt.data["q1_misalign_x"] = float(learned_mean.q1_misalign_x.item())
                xopt.data["q1_misalign_y"] = float(learned_mean.q1_misalign_y.item())
                xopt.data["q2_misalign_x"] = float(learned_mean.q2_misalign_x.item())
                xopt.data["q2_misalign_y"] = float(learned_mean.q2_misalign_y.item())
                xopt.data["q3_misalign_x"] = float(learned_mean.q3_misalign_x.item())
                xopt.data["q3_misalign_y"] = float(learned_mean.q3_misalign_y.item())
            except Exception as e:
                print(f"  Warning: Could not save learned misalignments: {e}")
        
        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        df = pd.concat([df, xopt.data])
        
        # Print trial summary
        print(f"\n  === Trial {i+1} Summary ===")
        print(f"  Best mae: {xopt.data['mae'].min():.6e}")
        
        if args.optimizer == "BO_prior" and args.task in ["mismatched", "matched_prior_newtask"]:
            try:
                print(f"  Final learned misalignments:")
                print(f"    Q1: x={learned_mean.q1_misalign_x.item():.6f}m, y={learned_mean.q1_misalign_y.item():.6f}m")
                print(f"    Q2: x={learned_mean.q2_misalign_x.item():.6f}m, y={learned_mean.q2_misalign_y.item():.6f}m")
                print(f"    Q3: x={learned_mean.q3_misalign_x.item():.6f}m, y={learned_mean.q3_misalign_y.item():.6f}m")
                
                if args.task == "mismatched":
                    print(f"  Ground truth misalignments:")
                    print(f"    Q1: x=0.000000m, y=0.000100m")
                    print(f"    Q2: x=0.000030m, y=-0.000120m")
                    print(f"    Q3: x=-0.000040m, y=0.000080m")
            except: 
                pass

    # Save results
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
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
        incoming_beam = cheetah.ParameterBeam.from_parameters(
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
            function_kwargs={
                "incoming_beam": incoming_beam,
                "misalignment_config": misalignment_config,
            },
        )
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
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )
            
            generator = UpperConfidenceBoundGenerator(
                beta=0.5, vocs=vocs, gp_constructor=gp_constructor
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
        
         # Evaluate 5 initial points
        # Point 1: Fixed starting point
        print(f"  Evaluating initial point 1/5 (fixed)")
        xopt.evaluate_data({
            "q1": 10.0,
            "q2": -10.0,
            "q3": 10.0,
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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_filename = f"{args.output_dir}/{args.optimizer}_{args.task}.csv"

    df.to_csv(out_filename)
    print(f"Results saved to {out_filename}")


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
    main(args)
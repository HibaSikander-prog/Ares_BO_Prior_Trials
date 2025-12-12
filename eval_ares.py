"""Evaluate the ARES optimization task using different optimizers.

task modes:
    - matched:  Optimize the ARES lattice for a matched beam with no misalignments.
    - mismatched:  Optimize ARES lattice with misaligned quadrupoles and different beam.
    - matched_prior_newtask: Optimize with correct prior for the misaligned case.
"""

import os

import bo_cheetah_prior_ares
import cheetah
import numpy as np
import pandas as pd
import torch
import tqdm
from xopt import VOCS, Evaluator, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.sequential.neldermead import NelderMeadGenerator


def main(args):
    # VOCS - 5D optimization problem
    vocs_config = """
        variables: 
            q1: [-30, 30]
            q2: [-30, 30]
            q3: [-30, 30]
            cv: [-0.006, 0.006]
            ch: [-0.006, 0.006]
        objectives:
            mae: minimize
    """
    vocs = VOCS.from_yaml(vocs_config)
    
    # Set random seed for reproducibility of misalignments
    np.random.seed(42)
    
    # Generate random misalignments in range [-0.5mm, 0.5mm] = [-0.0005m, 0.0005m]
    misalignment_values = {
        "q1_misalignment_x": np.random.uniform(-0.0005, 0.0005),
        "q1_misalignment_y": np.random.uniform(-0.0005, 0.0005),
        "q2_misalignment_x": np.random.uniform(-0.0005, 0.0005),
        "q2_misalignment_y": np.random.uniform(-0.0005, 0.0005),
        "q3_misalignment_x": np.random.uniform(-0.0005, 0.0005),
        "q3_misalignment_y":  np.random.uniform(-0.0005, 0.0005),
    }
    
    print(f"Task:  {args.task}")
    print(f"Generated misalignments (mm):")
    for key, val in misalignment_values.items():
        print(f"  {key}: {val * 1000:.4f}")
    
    # Evaluator
    if args.task == "matched": 
        incoming_beam = None
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_problem,
            function_kwargs={"incoming_beam": incoming_beam},
        )
    elif args.task == "mismatched":
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor(1e-3),
            sigma_y=torch.tensor(1e-3),
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_problem,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "lattice_config": misalignment_values,
            },
        )
    elif args.task == "matched_prior_newtask":
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor(1e-3),
            sigma_y=torch.tensor(1e-3),
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_problem,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "lattice_config":  misalignment_values,
            },
        )
    
    df = pd.DataFrame()
    
    # Run n_trials
    for i in range(args.n_trials):
        print(f"Trial {i+1}/{args.n_trials}")
        
        # Initialize Generator
        if args.optimizer == "BO":
            generator = UpperConfidenceBoundGenerator(beta=2.0, vocs=vocs)
        elif args.optimizer == "BO_prior":
            # Create prior mean module
            if args.task == "matched": 
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean()
                # All misalignments are 0.0 by default
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )
            elif args.task == "mismatched": 
                # Prior is mismatched - uses default beam and 0.0 misalignments
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean()
                # Make prior trainable to adapt
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module},
                    trainable_mean_keys=["mae"],
                )
            elif args.task == "matched_prior_newtask":
                # Prior is correctly matched to the new task
                incoming_beam = cheetah.ParameterBeam.from_parameters(
                    sigma_x=torch.tensor(1e-3),
                    sigma_y=torch.tensor(1e-3),
                    sigma_px=torch.tensor(1e-4),
                    sigma_py=torch.tensor(1e-4),
                    energy=torch.tensor(100e6),
                )
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean(
                    incoming_beam=incoming_beam
                )
                # Set misalignments to match the true values
                prior_mean_module.raw_q1_misalignment_x.data = torch.tensor(
                    misalignment_values["q1_misalignment_x"]
                )
                prior_mean_module.raw_q1_misalignment_y.data = torch.tensor(
                    misalignment_values["q1_misalignment_y"]
                )
                prior_mean_module.raw_q2_misalignment_x.data = torch.tensor(
                    misalignment_values["q2_misalignment_x"]
                )
                prior_mean_module.raw_q2_misalignment_y.data = torch.tensor(
                    misalignment_values["q2_misalignment_y"]
                )
                prior_mean_module.raw_q3_misalignment_x.data = torch.tensor(
                    misalignment_values["q3_misalignment_x"]
                )
                prior_mean_module.raw_q3_misalignment_y.data = torch.tensor(
                    misalignment_values["q3_misalignment_y"]
                )
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )
            generator = UpperConfidenceBoundGenerator(
                beta=2.0, vocs=vocs, gp_constructor=gp_constructor
            )
        elif args.optimizer == "NM":
            generator = NelderMeadGenerator(vocs=vocs)
        else:
            raise ValueError(f"Invalid optimizer: {args.optimizer}")
        
        xopt = Xopt(
            vocs=vocs,
            evaluator=evaluator,
            generator=generator,
            max_evaluations=args.max_evaluation_steps,
        )
        
        # Generate starting points
        # First point is fixed
        starting_points = [
            {"q1": -10.0, "q2": 10.0, "q3":  10.0, "cv": 0.0, "ch": 0.0}
        ]
        
        # Generate additional random starting points if n_initial > 1
        for _ in range(args.n_initial - 1):
            starting_points.append({
                "q1": np.random.uniform(-30, 30),
                "q2": np.random.uniform(-30, 30),
                "q3": np.random.uniform(-30, 30),
                "cv": np.random.uniform(-0.006, 0.006),
                "ch": np.random.uniform(-0.006, 0.006),
            })
        
        # Evaluate starting points
        for start_point in starting_points:
            xopt.evaluate_data(start_point)
        
        # Run optimization
        for _ in tqdm.tqdm(range(args.max_evaluation_steps)):
            xopt.step()
        
        # Post processing
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_mae"] = xopt.data["mae"].cummin()
        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        df = pd.concat([df, xopt.data])
    
    # Save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_filename = f"{args.output_dir}/{args.optimizer}_{args.task}.csv"
    df.to_csv(out_filename)
    print(f"Results saved to: {out_filename}")


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
        help="Task to run.",
    )
    parser.add_argument(
        "--max_evaluation_steps",
        "-s",
        type=int,
        default=50,
        help="Maximum number of evaluations to run for each trial.",
    )
    parser.add_argument(
        "--n_initial",
        type=int,
        default=5,
        help="Number of initial points (first is fixed, rest are random).",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data_ares/",
        help="Output directory for results.",
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
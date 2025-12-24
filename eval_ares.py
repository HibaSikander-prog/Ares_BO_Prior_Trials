"""Evaluate the ARES optimization task using different optimizers.

task modes:
    - matched:  Optimize the ARES lattice for a matched beam.
    - mismatched:  Optimize ARES lattice with mismatched prior (different beam and misalignments).
    - matched_prior_newtask: Optimize with a prior matched to the new task.
"""

import os
import time
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
<<<<<<< Updated upstream
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(1e-3),  # Changed from 2e-3 to 1e-3
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
=======
        #incoming_beam = cheetah.ParticleBeam.from_parameters(
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            #num_particles=10000,  # Use ParticleBeam for bmadx tracking
            mu_x=torch.tensor(8.2413e-07),
            mu_px=torch.tensor(5.9885e-08),
            mu_y=torch.tensor(-1.7276e-06),
            mu_py=torch.tensor(-1.1746e-07),
            sigma_x=torch.tensor(0.0002),
            sigma_px=torch.tensor(3.6794e-06),
            sigma_y=torch.tensor(0.0001),  # Changed to 0.0001 to from 0.0002
            sigma_py=torch.tensor(3.6941e-06),
            sigma_tau=torch.tensor(8.0116e-06),
            sigma_p=torch.tensor(0.0023),
            energy=torch.tensor(1.0732e+08),
            total_charge=torch.tensor(5.0e-13),
>>>>>>> Stashed changes
        )
        # Different misalignments
        misalignment_config = {
<<<<<<< Updated upstream
            "AREAMQZM1": (0.0003, -0.0002),   # 0.3mm, -0.2mm
            "AREAMQZM2": (-0.0001, 0.00025),  # -0.1mm, 0.25mm
            "AREAMQZM3": (0.00015, -0.0003),  # 0.15mm, -0.3mm
=======
            "AREAMQZM1": (0.0000, 0.0002),   # 0, 200 μm (was 100 μm)
            "AREAMQZM2": (0.0001, -0.0003),  # 100, -300 μm (was 30, -120 μm)
            "AREAMQZM3": (-0.0001, 0.00015), # -100, 150 μm (was -40, 80 μm)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
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
        #incoming_beam = cheetah.ParticleBeam.from_parameters(
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            #num_particles=10000,  # Use ParticleBeam for bmadx tracking
            mu_x=torch.tensor(8.2413e-07),
            mu_px=torch.tensor(5.9885e-08),
            mu_y=torch.tensor(-1.7276e-06),
            mu_py=torch.tensor(-1.1746e-07),
            sigma_x=torch.tensor(0.0002),
            sigma_px=torch.tensor(3.6794e-06),
            sigma_y=torch.tensor(0.0001),  # Changed to 0.0001 to from 0.0002
            sigma_py=torch.tensor(3.6941e-06),
            sigma_tau=torch.tensor(8.0116e-06),
            sigma_p=torch.tensor(0.0023),
            energy=torch.tensor(1.0732e+08),
            total_charge=torch.tensor(5.0e-13),
        )
        misalignment_config = {
            "AREAMQZM1": (0.0000, 0.0002),   # 0, 200 μm
            "AREAMQZM2": (0.0001, -0.0003),  # 100, -300 μm
            "AREAMQZM3": (-0.0001, 0.00015), # -100, 150 μm
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
                incoming_beam = cheetah.ParameterBeam.from_parameters(
                    sigma_x=torch.tensor(1e-4),
                    sigma_y=torch.tensor(1e-3),
                    sigma_px=torch.tensor(1e-4),
                    sigma_py=torch.tensor(1e-4),
                    energy=torch.tensor(100e6),
=======
                #incoming_beam = cheetah.ParticleBeam.from_parameters(
                incoming_beam = cheetah.ParameterBeam.from_parameters(
                    #num_particles=10000,  # Use ParticleBeam for bmadx tracking
                    mu_x=torch.tensor(8.2413e-07),
                    mu_px=torch.tensor(5.9885e-08),
                    mu_y=torch.tensor(-1.7276e-06),
                    mu_py=torch.tensor(-1.1746e-07),
                    sigma_x=torch.tensor(0.0002),
                    sigma_px=torch.tensor(3.6794e-06),
                    sigma_y=torch.tensor(0.0001),  # Changed to 0.0001 to from 0.0002
                    sigma_py=torch.tensor(3.6941e-06),
                    sigma_tau=torch.tensor(8.0116e-06),
                    sigma_p=torch.tensor(0.0023),
                    energy=torch.tensor(1.0732e+08),
                    total_charge=torch.tensor(5.0e-13),
>>>>>>> Stashed changes
                )
                prior_mean_module = bo_cheetah_prior.AresPriorMean(
                    incoming_beam=incoming_beam
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
                # Initialize with CORRECT values (matched to evaluator)
                # Like FODO's drift_length=0.7 when truth is 0.7
                prior_mean_module.q1_misalign_x = 0.0000
                prior_mean_module.q1_misalign_y = 0.0002
                prior_mean_module.q2_misalign_x = 0.0001
                prior_mean_module.q2_misalign_y = -0.0003
                prior_mean_module.q3_misalign_x = -0.0001
                prior_mean_module.q3_misalign_y = 0.00015
>>>>>>> Stashed changes
                
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
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
        
<<<<<<< Updated upstream
         # Evaluate 5 initial points
        # Point 1: Fixed starting point
        print(f"  Evaluating initial point 1/5 (fixed)")
        xopt.evaluate_data({
=======
        # CRITICAL FIX: Use only 1 fixed starting point (like FODO example)
        print(f"  Evaluating initial point...")
        initial_point = {
>>>>>>> Stashed changes
            "q1": 10.0,
            "q2": -10.0,
            "cv": 0.0,
<<<<<<< Updated upstream
            "ch":  0.0,
        })
=======
            "q3": 10.0,
            "ch": 0.0,
        }

        print(f"[INIT] Initial point: {initial_point}")


        xopt.evaluate_data(initial_point)
        print(f"[INIT] Initial MAE: {xopt.data['mae'].iloc[0]:.6e}")

        # Start Optimization
        print(f"\n[OPT] Starting optimization for {args.max_evaluation_steps} steps...")

>>>>>>> Stashed changes
        
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

<<<<<<< Updated upstream
=======
            # DIAGNOSTIC: Progress logging every 10 steps
            if (step_num + 1) % 10 == 0:
                current_best = xopt.data["mae"].min()
                
                print(f"\n  Step {step_num + 1}/{args.max_evaluation_steps}:")
                print(f"    Best MAE so far: {current_best:.6e}")
                
                # Extract and display learned misalignments for BO_prior
                if args.optimizer == "BO_prior" and args.task in ["mismatched", "matched_prior_newtask"]:
                    try: 
                        model = xopt.generator.model
                        gp_model = model.models[0]
                        learned_mean = gp_model.mean_module._model
                        
                        print(f"    Current learned misalignments (m):")
                        print(f"      Q1: x={learned_mean.q1_misalign_x.item():.6f}, y={learned_mean.q1_misalign_y.item():.6f}")
                        print(f"      Q2: x={learned_mean.q2_misalign_x.item():.6f}, y={learned_mean.q2_misalign_y.item():.6f}")
                        print(f"      Q3: x={learned_mean.q3_misalign_x.item():.6f}, y={learned_mean.q3_misalign_y.item():.6f}")
                        
                        if args.task == "mismatched":
                            print(f"    Ground truth (m):")
                            print(f"      Q1: x=0.000000, y=0.000200")
                            print(f"      Q2: x=0.000100, y=-0.000300")
                            print(f"      Q3: x=-0.000100, y=0.000150")
                    except Exception as e:
                        print(f"    (Could not extract learned values: {e})")

>>>>>>> Stashed changes
        # Post processing the dataframes
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_mae"] = xopt.data["mae"].cummin()
<<<<<<< Updated upstream
        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        df = pd.concat([df, xopt.data])
    
    # Check if the output directory exists
=======

        # Also track best_mae for comparison
        #if "mae" in xopt.data.columns:
         #   xopt.data["best_mae"] = xopt.data["mae"].cummin()

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
        
        # DIAGNOSTIC: Print comprehensive trial summary
        print(f"\n{'='*70}")
        print(f"TRIAL {i+1} SUMMARY")
        print(f"{'='*70}")
        print(f"Best MAE achieved: {xopt.data['mae'].min():.6e}")
        
        if args.optimizer == "BO_prior" and args.task in ["mismatched", "matched_prior_newtask"]:
            try:
                print(f"  Final learned misalignments:")
                print(f"    Q1: x={learned_mean.q1_misalign_x.item():.6f}m, y={learned_mean.q1_misalign_y.item():.6f}m")
                print(f"    Q2: x={learned_mean.q2_misalign_x.item():.6f}m, y={learned_mean.q2_misalign_y.item():.6f}m")
                print(f"    Q3: x={learned_mean.q3_misalign_x.item():.6f}m, y={learned_mean.q3_misalign_y.item():.6f}m")
                
                if args.task == "mismatched":
                    print(f"\nGround truth misalignments (m):")
                    print(f"  Q1: x=0.000000, y=0.000200")
                    print(f"  Q2: x=0.000100, y=-0.000300")
                    print(f"  Q3: x=-0.000100, y=0.000150")

                    # Calculate error
                    error_q1_x = abs(learned_mean.q1_misalign_x.item() - 0.000000)
                    error_q1_y = abs(learned_mean.q1_misalign_y.item() - 0.000200)
                    error_q2_x = abs(learned_mean.q2_misalign_x.item() - 0.000100)
                    error_q2_y = abs(learned_mean.q2_misalign_y.item() + 0.000300)
                    error_q3_x = abs(learned_mean.q3_misalign_x.item() + 0.000100)
                    error_q3_y = abs(learned_mean.q3_misalign_y.item() - 0.000150)
                    
                    avg_error = (error_q1_x + error_q1_y + error_q2_x + error_q2_y + error_q3_x + error_q3_y) / 6
                    print(f"\nMisalignment Learning Error:")
                    print(f"  Average absolute error: {avg_error:.6f} m ({avg_error*1e6:.1f} μm)")
                    print(f"  Q1 errors: x={error_q1_x*1e6:.1f}μm, y={error_q1_y*1e6:.1f}μm")
                    print(f"  Q2 errors: x={error_q2_x*1e6:.1f}μm, y={error_q2_y*1e6:.1f}μm")
                    print(f"  Q3 errors: x={error_q3_x*1e6:.1f}μm, y={error_q3_y*1e6:.1f}μm")
            except: 
                pass

    # Save results
>>>>>>> Stashed changes
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
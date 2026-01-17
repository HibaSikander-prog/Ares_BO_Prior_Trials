"""Find the global minimum for the ARES problem."""
import torch
import numpy as np
from scipy.optimize import differential_evolution, minimize

def main():
    import bo_cheetah_prior_ares as bo_cheetah_prior
    import cheetah

    # Set up the same beam and misalignments as matched_prior_newtask
    # Use float32 to match Cheetah's default dtype
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        mu_x=torch.tensor(8.2413e-07, dtype=torch.float32),
        mu_px=torch.tensor(5.9885e-08, dtype=torch.float32),
        mu_y=torch.tensor(-1.7276e-06, dtype=torch.float32),
        mu_py=torch.tensor(-1.1746e-07, dtype=torch.float32),
        sigma_x=torch.tensor(0.0002, dtype=torch.float32),
        sigma_px=torch.tensor(3.6794e-06, dtype=torch.float32),
        sigma_y=torch.tensor(0.0001, dtype=torch.float32),  # Changed
        sigma_py=torch.tensor(3.6941e-06, dtype=torch.float32),
        sigma_tau=torch.tensor(8.0116e-06, dtype=torch.float32),
        sigma_p=torch.tensor(0.0023, dtype=torch.float32),
        energy=torch.tensor(1.0732e+08, dtype=torch.float32),
        total_charge=torch.tensor(5.0e-13, dtype=torch.float32),
    )

    misalignment_config = {
        "AREAMQZM1": (0.0000, 0.0002),
        "AREAMQZM2": (0.0001, -0.0003),
        "AREAMQZM3": (-0.0001, 0.00015),
    }

    def objective(x):
        """Objective function for scipy optimizer."""
        input_param = {
            "q1": float(x[0]),
            "q2": float(x[1]),
            "cv":  float(x[2]),
            "q3": float(x[3]),
            "ch": float(x[4]),
        }
        result = bo_cheetah_prior.ares_problem(
            input_param,
            incoming_beam=incoming_beam,
            misalignment_config=misalignment_config,
        )
        return float(result["mae"])

    # Bounds from VOCS
    bounds = [
        (-30, 30),       # q1
        (-30, 30),       # q2
        (-0.006, 0.006), # cv
        (-30, 30),       # q3
        (-0.006, 0.006), # ch
    ]

    print("Finding global minimum using differential evolution...")
    print("This may take a few minutes...\n")

    # Use differential evolution for global optimization
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=1000,
        tol=1e-10,
        disp=True,
        workers=1,  # Single process - avoids Windows multiprocessing issues
        updating='immediate',
    )

    print("\n" + "="*70)
    print("GLOBAL MINIMUM FOUND")
    print("="*70)
    print(f"Minimum MAE: {result.fun:.6e} ({result.fun*1000:.6f} mm)")
    print(f"\nOptimal parameters:")
    print(f"  q1: {result.x[0]:.6f}")
    print(f"  q2: {result.x[1]:.6f}")
    print(f"  cv:  {result.x[2]:.6f}")
    print(f"  q3: {result.x[3]:.6f}")
    print(f"  ch: {result.x[4]:.6f}")

    # Verify with a local refinement
    print("\nRefining with L-BFGS-B...")
    result_refined = minimize(
        objective,
        result.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-12}
    )

    print(f"\nRefined Minimum MAE: {result_refined.fun:.6e} ({result_refined.fun*1000:.6f} mm)")
    print(f"\nRefined optimal parameters:")
    print(f"  q1: {result_refined.x[0]:.6f}")
    print(f"  q2: {result_refined.x[1]:.6f}")
    print(f"  cv: {result_refined.x[2]:.6f}")
    print(f"  q3: {result_refined.x[3]:.6f}")
    print(f"  ch: {result_refined.x[4]:.6f}")

    # Also check what happens at the initial point
    initial_point = {"q1": 10.0, "q2": -10.0, "cv": 0.0, "q3": 10.0, "ch": 0.0}
    initial_result = bo_cheetah_prior.ares_problem(
        initial_point,
        incoming_beam=incoming_beam,
        misalignment_config=misalignment_config,
    )
    print(f"\nInitial point MAE: {initial_result['mae']:.6e} ({float(initial_result['mae'])*1000:.6f} mm)")

    # Test a grid of points near zero to understand the landscape
    print("\n" + "="*70)
    print("TESTING POINTS NEAR ZERO")
    print("="*70)
    test_points = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 5.0, 0.0, 0.0, 0.0],
    ]
    
    for pt in test_points: 
        val = objective(pt)
        print(f"  {pt} -> MAE = {val:.6e} ({val*1000:.4f} mm)")


if __name__ == '__main__':
    main()
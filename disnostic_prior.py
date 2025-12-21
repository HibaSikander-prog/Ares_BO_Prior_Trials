"""Diagnostic script to verify the prior is working correctly."""

import torch
import cheetah
import numpy as np
import bo_cheetah_prior_ares as bo_cheetah_prior

print("="*70)
print("DIAGNOSTIC 1: Test Prior vs Actual Evaluator")
print("="*70)

# Define test points
test_points = [
    {"q1": 10.0, "q2": -10.0, "q3": 10.0, "cv": 0.0, "ch": 0.0},  # Initial point
    {"q1": 0.0, "q2": 0.0, "q3": 0.0, "cv": 0.0, "ch": 0.0},      # Center
    {"q1": 20.0, "q2": -20.0, "q3": 20.0, "cv": 0.0, "ch": 0.0},  # Stronger magnets
]

# Setup mismatched task
incoming_beam = cheetah.ParticleBeam.from_parameters(
    num_particles=1000,
    sigma_x=torch.tensor(1e-4),
    sigma_y=torch.tensor(1e-3),
    sigma_px=torch.tensor(1e-4),
    sigma_py=torch.tensor(1e-4),
    energy=torch.tensor(100e6),
)

misalignment_config = {
    "AREAMQZM1": (0.000000, 0.000100),
    "AREAMQZM2": (0.000030, -0.000120),
    "AREAMQZM3": (-0.000040, 0.000080),
}

# Create prior with WRONG initial guess (mismatched case)
prior_mean_module = bo_cheetah_prior.AresPriorMean(incoming_beam=incoming_beam)
prior_mean_module.q1_misalign_x = 0.0
prior_mean_module.q1_misalign_y = 0.0
prior_mean_module.q2_misalign_x = 0.0
prior_mean_module.q2_misalign_y = 0.0
prior_mean_module.q3_misalign_x = 0.0
prior_mean_module.q3_misalign_y = 0.0

print("\nPrior initialized with ZERO misalignments")
print("Ground truth misalignments:")
print(f"  Q1: x=0.000000, y=0.000100")
print(f"  Q2: x=0.000030, y=-0.000120")
print(f"  Q3: x=-0.000040, y=0.000080")

print("\n" + "-"*70)
print(f"{'Test Point':<30s} {'Prior Pred':<15s} {'True Value':<15s} {'Error':<15s}")
print("-"*70)

for i, point in enumerate(test_points):
    # Evaluate with true evaluator
    true_result = bo_cheetah_prior.ares_problem(
        point, 
        incoming_beam=incoming_beam,
        misalignment_config=misalignment_config
    )
    true_mae = true_result["mae"]
    
    # Evaluate with prior
    X = torch.tensor([[point["q1"], point["q2"], point["cv"], point["q3"], point["ch"]]])
    prior_pred = prior_mean_module.forward(X).item()
    
    error = abs(prior_pred - true_mae)
    error_pct = (error / true_mae) * 100
    
    point_str = f"Point {i+1}: q1={point['q1']:.0f}"
    print(f"{point_str:<30s} {prior_pred:.6e}   {true_mae:.6e}   {error:.6e} ({error_pct:.1f}%)")

print("="*70)

# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 2: Test Misalignment Sensitivity")
print("="*70)

test_point = {"q1": 10.0, "q2": -10.0, "q3": 10.0, "cv": 0.0, "ch": 0.0}

# No misalignment
result_no_mis = bo_cheetah_prior.ares_problem(
    test_point, 
    incoming_beam=incoming_beam,
    misalignment_config=None
)

# With misalignment
result_with_mis = bo_cheetah_prior.ares_problem(
    test_point, 
    incoming_beam=incoming_beam,
    misalignment_config=misalignment_config
)

print(f"\nTest point: q1=10, q2=-10, q3=10, cv=0, ch=0")
print(f"MAE without misalignment:  {result_no_mis['mae']:.6e}")
print(f"MAE with misalignment:     {result_with_mis['mae']:.6e}")
print(f"Absolute difference:       {abs(result_with_mis['mae'] - result_no_mis['mae']):.6e}")
print(f"Relative difference:       {abs(result_with_mis['mae'] - result_no_mis['mae']) / result_no_mis['mae'] * 100:.2f}%")

if abs(result_with_mis['mae'] - result_no_mis['mae']) / result_no_mis['mae'] < 0.01:
    print("\n⚠️  WARNING: Misalignments have <1% effect! Prior may not help much.")
else:
    print("\n✓ Misalignments have significant effect. Prior should help.")

print("="*70)

# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 3: Check Initial Point Quality")
print("="*70)

initial_point = {"q1": 10.0, "q2": -10.0, "q3": 10.0, "cv": 0.0, "ch": 0.0}
result_initial = bo_cheetah_prior.ares_problem(
    initial_point,
    incoming_beam=incoming_beam,
    misalignment_config=misalignment_config
)

# Try some random points for comparison
n_random = 50
random_results = []
np.random.seed(42)

for _ in range(n_random):
    random_point = {
        "q1": np.random.uniform(-30, 30),
        "q2": np.random.uniform(-30, 30),
        "q3": np.random.uniform(-30, 30),
        "cv": np.random.uniform(-0.006, 0.006),
        "ch": np.random.uniform(-0.006, 0.006),
    }
    result = bo_cheetah_prior.ares_problem(
        random_point,
        incoming_beam=incoming_beam,
        misalignment_config=misalignment_config
    )
    random_results.append(result["mae"])

random_results = np.array(random_results)

print(f"\nInitial point MAE:        {result_initial['mae']:.6e}")
print(f"Random points mean MAE:   {random_results.mean():.6e}")
print(f"Random points median MAE: {np.median(random_results):.6e}")
print(f"Random points min MAE:    {random_results.min():.6e}")
print(f"Random points max MAE:    {random_results.max():.6e}")

percentile = (random_results > result_initial['mae']).sum() / len(random_results) * 100
print(f"\nInitial point is better than {percentile:.1f}% of random points")

if percentile > 90:
    print("✓ Initial point is EXCELLENT - explains fast initial convergence")
elif percentile > 70:
    print("✓ Initial point is GOOD")
elif percentile > 50:
    print("→ Initial point is AVERAGE")
else:
    print("✗ Initial point is POOR")

print("="*70)

# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC 4: Compare Matched vs Mismatched Prior")
print("="*70)

# Matched prior (correct misalignments)
prior_matched = bo_cheetah_prior.AresPriorMean(incoming_beam=incoming_beam)
prior_matched.q1_misalign_x = 0.000000
prior_matched.q1_misalign_y = 0.000100
prior_matched.q2_misalign_x = 0.000030
prior_matched.q2_misalign_y = -0.000120
prior_matched.q3_misalign_x = -0.000040
prior_matched.q3_misalign_y = 0.000080

# Mismatched prior (zero misalignments)
prior_mismatched = bo_cheetah_prior.AresPriorMean(incoming_beam=incoming_beam)
prior_mismatched.q1_misalign_x = 0.0
prior_mismatched.q1_misalign_y = 0.0
prior_mismatched.q2_misalign_x = 0.0
prior_mismatched.q2_misalign_y = 0.0
prior_mismatched.q3_misalign_x = 0.0
prior_mismatched.q3_misalign_y = 0.0

print("\nComparing predictions at test points:")
print("-"*70)
print(f"{'Point':<20s} {'Matched Prior':<15s} {'Mismatched Prior':<15s} {'True Value':<15s}")
print("-"*70)

for i, point in enumerate(test_points):
    X = torch.tensor([[point["q1"], point["q2"], point["cv"], point["q3"], point["ch"]]])
    
    pred_matched = prior_matched.forward(X).item()
    pred_mismatched = prior_mismatched.forward(X).item()
    
    true_result = bo_cheetah_prior.ares_problem(
        point, 
        incoming_beam=incoming_beam,
        misalignment_config=misalignment_config
    )
    true_mae = true_result["mae"]
    
    print(f"Point {i+1:<16d} {pred_matched:.6e}   {pred_mismatched:.6e}   {true_mae:.6e}")

matched_better = 0
mismatched_better = 0

for point in test_points:
    X = torch.tensor([[point["q1"], point["q2"], point["cv"], point["q3"], point["ch"]]])
    pred_matched = prior_matched.forward(X).item()
    pred_mismatched = prior_mismatched.forward(X).item()
    
    true_result = bo_cheetah_prior.ares_problem(
        point, 
        incoming_beam=incoming_beam,
        misalignment_config=misalignment_config
    )
    true_mae = true_result["mae"]
    
    error_matched = abs(pred_matched - true_mae)
    error_mismatched = abs(pred_mismatched - true_mae)
    
    if error_matched < error_mismatched:
        matched_better += 1
    else:
        mismatched_better += 1

print(f"\nMatched prior is more accurate on {matched_better}/{len(test_points)} points")
print(f"Mismatched prior is more accurate on {mismatched_better}/{len(test_points)} points")

if matched_better > mismatched_better:
    print("✓ Matched prior provides better predictions")
else:
    print("⚠️  WARNING: Matched prior is NOT better! Something may be wrong.")

print("="*70)
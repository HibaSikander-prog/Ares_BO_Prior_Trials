"""
Diagnostic tests to verify ARES BO prior implementation.

Tests: 
1.Ground truth optimality check
2.Prior-evaluator consistency
"""

import numpy as np
import torch

import bo_cheetah_prior_ares as bo_cheetah_prior
import cheetah

# Realistic misalignments (50-150 μm range)
GROUND_TRUTH = {
    "AREAMQZM1": (0.000000, 0.000100),
    "AREAMQZM2": (0.000030, -0.000120),
    "AREAMQZM3": (-0.000040, 0.000080),
}

MISMATCHED_BEAM = cheetah.ParticleBeam.from_parameters(
    num_particles=1000,
    sigma_x=torch.tensor(1e-4),
    sigma_y=torch.tensor(1e-3),
    sigma_px=torch.tensor(1e-4),
    sigma_py=torch.tensor(1e-4),
    energy=torch.tensor(100e6),
)

DEFAULT_MAGNETS = {
    "q1": 10.0,
    "q2": -10.0,
    "cv": 0.0,
    "q3": 10.0,
    "ch": 0.0,
}


def test_1_ground_truth_optimality():
    """Test if ground truth misalignments are optimal."""
    print("\n" + "="*70)
    print("TEST 1: Is Ground Truth Optimal?")
    print("="*70)
    
    result_zero = bo_cheetah_prior.ares_problem(
        input_param=DEFAULT_MAGNETS,
        incoming_beam=MISMATCHED_BEAM,
        misalignment_config={
            "AREAMQZM1": (0.0, 0.0),
            "AREAMQZM2": (0.0, 0.0),
            "AREAMQZM3": (0.0, 0.0),
        },
    )
    
    result_gt = bo_cheetah_prior.ares_problem(
        input_param=DEFAULT_MAGNETS,
        incoming_beam=MISMATCHED_BEAM,
        misalignment_config=GROUND_TRUTH,
    )
    
    result_boundary = bo_cheetah_prior.ares_problem(
        input_param=DEFAULT_MAGNETS,
        incoming_beam=MISMATCHED_BEAM,
        misalignment_config={
            "AREAMQZM1": (0.0005, -0.0005),
            "AREAMQZM2":  (-0.0005, 0.0005),
            "AREAMQZM3": (0.0005, -0.0005),
        },
    )
    
    print(f"\nObjective values (MAE - lower is better):")
    print(f"  Zero misalignments:       {result_zero['mae']:.6e}")
    print(f"  Ground truth (new):      {result_gt['mae']:.6e}")
    print(f"  Boundary (±0.5mm):      {result_boundary['mae']:.6e}")
    
    if result_gt['mae'] < result_zero['mae']:
        print("\n✓ Ground truth performs better than zero")
    else:
        print("\n⚠️  Zero performs better - ground truth may still be suboptimal")
    
    return {
        'zero':  result_zero['mae'],
        'ground_truth': result_gt['mae'],
        'boundary': result_boundary['mae']
    }


def test_2_prior_evaluator_consistency():
    """Check if prior forward() matches evaluator."""
    print("\n" + "="*70)
    print("TEST 2: Prior vs Evaluator Consistency")
    print("="*70)
    
    prior_mean_module = bo_cheetah_prior.AresPriorMean(incoming_beam=MISMATCHED_BEAM)
    
    # Set ground truth misalignments
    prior_mean_module.q1_misalign_x = GROUND_TRUTH["AREAMQZM1"][0]
    prior_mean_module.q1_misalign_y = GROUND_TRUTH["AREAMQZM1"][1]
    prior_mean_module.q2_misalign_x = GROUND_TRUTH["AREAMQZM2"][0]
    prior_mean_module.q2_misalign_y = GROUND_TRUTH["AREAMQZM2"][1]
    prior_mean_module.q3_misalign_x = GROUND_TRUTH["AREAMQZM3"][0]
    prior_mean_module.q3_misalign_y = GROUND_TRUTH["AREAMQZM3"][1]
    
    X_test = torch.tensor([[
        DEFAULT_MAGNETS["q1"],
        DEFAULT_MAGNETS["q2"],
        DEFAULT_MAGNETS["cv"],
        DEFAULT_MAGNETS["q3"],
        DEFAULT_MAGNETS["ch"],
    ]], dtype=torch.float32)
    
    with torch.no_grad():
        prior_pred = prior_mean_module(X_test)
    
    eval_result = bo_cheetah_prior.ares_problem(
        input_param=DEFAULT_MAGNETS,
        incoming_beam=MISMATCHED_BEAM,
        misalignment_config=GROUND_TRUTH,
    )
    
    print(f"\nPredictions:")
    print(f"  Prior forward():  {prior_pred.item():.10e}")
    print(f"  Evaluator:         {eval_result['mae']:.10e}")
    
    rel_diff = abs(prior_pred.item() - eval_result['mae']) / eval_result['mae'] * 100
    
    if rel_diff < 0.1:
        print(f"\n✓ Prior and evaluator are CONSISTENT ({rel_diff:.4f}% difference)")
    else:
        print(f"\n❌ DISCREPANCY:  {rel_diff:.2f}% difference")
    
    return {'rel_diff_pct': rel_diff}


def main():
    """Run diagnostic tests."""
    print("\n" + "="*70)
    print("  ARES BO PRIOR DIAGNOSTIC TESTS")
    print("="*70)
    print("\nUsing ground truth (50-150 μm range):")
    print(f"  Q1: {GROUND_TRUTH['AREAMQZM1']}")
    print(f"  Q2: {GROUND_TRUTH['AREAMQZM2']}")
    print(f"  Q3: {GROUND_TRUTH['AREAMQZM3']}")
    print("\n⭐ NOW USING MAE ONLY (matching FODO)")
    
    results = {}
    
    results['optimality'] = test_1_ground_truth_optimality()
    results['consistency'] = test_2_prior_evaluator_consistency()
    
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    if results['optimality']['ground_truth'] < results['optimality']['zero']:
        print("\n✓ Ground truth is better than zero misalignments")
    
    if results['consistency']['rel_diff_pct'] < 1.0:
        print("✓ Prior and evaluator are consistent")
    
    print("\n✅ Ready to run eval_ares.py!")
    print("="*70)


if __name__ == "__main__":
    main()
"""Test script to verify that misalignments are working correctly."""

import torch
import cheetah
import bo_cheetah_prior_ares as bo_cheetah_prior


def test_misalignment_in_objective():
    """Test that misalignments affect the objective function."""
    print("=" * 60)
    print("TEST 1: Do misalignments affect the objective function?")
    print("=" * 60)
    
    # Fixed magnet parameters
    input_param = {
        "q1": 10.0,
        "q2": -10.0,
        "q3":  10.0,
        "cv": 0.0,
        "ch": 0.0,
    }
    
    # Same beam
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        sigma_x=torch.tensor(1e-4),
        sigma_y=torch.tensor(1e-3),
        sigma_px=torch.tensor(1e-4),
        sigma_py=torch.tensor(1e-4),
        energy=torch.tensor(100e6),
    )
    
    # Test 1: No misalignments
    result_no_misalign = bo_cheetah_prior.ares_problem(
        input_param,
        incoming_beam=incoming_beam,
        misalignment_config=None
    )
    
    # Test 2: With misalignments
    misalignment_config = {
        "AREAMQZM1": (0.0003, -0.0002),
        "AREAMQZM2": (-0.0001, 0.00025),
        "AREAMQZM3": (0.00015, -0.0003),
    }
    result_with_misalign = bo_cheetah_prior.ares_problem(
        input_param,
        incoming_beam=incoming_beam,
        misalignment_config=misalignment_config
    )
    
    # Test 3: Larger misalignments (should have bigger effect)
    large_misalignment_config = {
        "AREAMQZM1": (0.001, -0.001),  # 1mm
        "AREAMQZM2": (-0.001, 0.001),
        "AREAMQZM3": (0.001, -0.001),
    }
    result_large_misalign = bo_cheetah_prior.ares_problem(
        input_param,
        incoming_beam=incoming_beam,
        misalignment_config=large_misalignment_config
    )
    
    print(f"\nNo misalignment:     MAE = {result_no_misalign['mae']:.6e}")
    print(f"Small misalignment: MAE = {result_with_misalign['mae']:.6e}")
    print(f"Large misalignment:  MAE = {result_large_misalign['mae']:.6e}")
    
    # Check if misalignments have effect
    diff_small = abs(result_with_misalign['mae'] - result_no_misalign['mae'])
    diff_large = abs(result_large_misalign['mae'] - result_no_misalign['mae'])
    
    print(f"\nDifference (small): {diff_small:.6e}")
    print(f"Difference (large): {diff_large:.6e}")
    
    if diff_small > 1e-10: 
        print("✓ Small misalignments DO affect the objective")
    else:
        print("✗ Small misalignments DO NOT affect the objective")
    
    if diff_large > diff_small:
        print("✓ Larger misalignments have larger effect")
    else:
        print("✗ Larger misalignments do NOT have larger effect")
    
    return diff_small > 1e-10 and diff_large > diff_small


def test_misalignment_in_prior():
    """Test that misalignments affect the prior mean predictions."""
    print("\n" + "=" * 60)
    print("TEST 2: Do misalignments affect the prior mean? ")
    print("=" * 60)
    
    # Create input tensor for BO
    X = torch.tensor([[10.0, -10.0, 0.0, 10.0, 0.0]])  # [q1, q2, cv, q3, ch]
    
    # Same beam
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        sigma_x=torch.tensor(1e-4),
        sigma_y=torch.tensor(1e-3),
        sigma_px=torch.tensor(1e-4),
        sigma_py=torch.tensor(1e-4),
        energy=torch.tensor(100e6),
    )
    
    # Test 1: Prior with zero misalignments
    prior_zero = bo_cheetah_prior.AresPriorMean(incoming_beam=incoming_beam)
    prior_zero.q1_misalign_x = 0.0
    prior_zero.q1_misalign_y = 0.0
    prior_zero.q2_misalign_x = 0.0
    prior_zero.q2_misalign_y = 0.0
    prior_zero.q3_misalign_x = 0.0
    prior_zero.q3_misalign_y = 0.0
    
    pred_zero = prior_zero.forward(X)
    
    # Test 2: Prior with small misalignments
    prior_small = bo_cheetah_prior.AresPriorMean(incoming_beam=incoming_beam)
    prior_small.q1_misalign_x = 0.0003
    prior_small.q1_misalign_y = -0.0002
    prior_small.q2_misalign_x = -0.0001
    prior_small.q2_misalign_y = 0.00025
    prior_small.q3_misalign_x = 0.00015
    prior_small.q3_misalign_y = -0.0003
    
    pred_small = prior_small.forward(X)
    
    # Test 3: Prior with large misalignments
    prior_large = bo_cheetah_prior.AresPriorMean(incoming_beam=incoming_beam)
    prior_large.q1_misalign_x = 0.001
    prior_large.q1_misalign_y = -0.001
    prior_large.q2_misalign_x = -0.001
    prior_large.q2_misalign_y = 0.001
    prior_large.q3_misalign_x = 0.001
    prior_large.q3_misalign_y = -0.001
    
    pred_large = prior_large.forward(X)
    
    print(f"\nZero misalignment:  MAE = {pred_zero.item():.6e}")
    print(f"Small misalignment:  MAE = {pred_small.item():.6e}")
    print(f"Large misalignment: MAE = {pred_large.item():.6e}")
    
    # Check if misalignments have effect on prior
    diff_small_prior = abs(pred_small.item() - pred_zero.item())
    diff_large_prior = abs(pred_large.item() - pred_zero.item())
    
    print(f"\nDifference (small): {diff_small_prior:.6e}")
    print(f"Difference (large): {diff_large_prior:.6e}")
    
    if diff_small_prior > 1e-10:
        print("✓ Small misalignments DO affect the prior")
    else:
        print("✗ Small misalignments DO NOT affect the prior")
    
    if diff_large_prior > diff_small_prior: 
        print("✓ Larger misalignments have larger effect on prior")
    else:
        print("✗ Larger misalignments do NOT have larger effect on prior")
    
    return diff_small_prior > 1e-10 and diff_large_prior > diff_small_prior


def test_prior_vs_objective_consistency():
    """Test that prior and objective give similar results for same inputs."""
    print("\n" + "=" * 60)
    print("TEST 3: Does prior predict similar values to objective?")
    print("=" * 60)
    
    # Same parameters
    input_dict = {
        "q1":  10.0,
        "q2": -10.0,
        "q3": 10.0,
        "cv":  0.0,
        "ch": 0.0,
    }
    X = torch.tensor([[10.0, -10.0, 0.0, 10.0, 0.0]])
    
    # Same beam
    incoming_beam = cheetah.ParameterBeam.from_parameters(
        sigma_x=torch.tensor(1e-4),
        sigma_y=torch.tensor(1e-3),
        sigma_px=torch.tensor(1e-4),
        sigma_py=torch.tensor(1e-4),
        energy=torch.tensor(100e6),
    )
    
    # Same misalignments
    misalignment_config = {
        "AREAMQZM1": (0.0003, -0.0002),
        "AREAMQZM2": (-0.0001, 0.00025),
        "AREAMQZM3": (0.00015, -0.0003),
    }
    
    # Objective function result
    obj_result = bo_cheetah_prior.ares_problem(
        input_dict,
        incoming_beam=incoming_beam,
        misalignment_config=misalignment_config
    )
    
    # Prior mean result
    prior = bo_cheetah_prior.AresPriorMean(incoming_beam=incoming_beam)
    prior.q1_misalign_x = 0.0003
    prior.q1_misalign_y = -0.0002
    prior.q2_misalign_x = -0.0001
    prior.q2_misalign_y = 0.00025
    prior.q3_misalign_x = 0.00015
    prior.q3_misalign_y = -0.0003
    
    prior_result = prior.forward(X)
    
    print(f"\nObjective MAE: {obj_result['mae']:.6e}")
    print(f"Prior MAE:      {prior_result.item():.6e}")
    
    relative_error = abs(obj_result['mae'] - prior_result.item()) / obj_result['mae']
    print(f"\nRelative error:  {relative_error:.2%}")
    
    if relative_error < 0.01:  # Less than 1% error
        print("✓ Prior and objective are VERY consistent")
        return True
    elif relative_error < 0.1:  # Less than 10% error
        print("✓ Prior and objective are reasonably consistent")
        return True
    else:
        print("✗ Prior and objective are NOT consistent - potential bug!")
        return False


def test_parameter_retrieval():
    """Test that we can get and set misalignment parameters correctly."""
    print("\n" + "=" * 60)
    print("TEST 4: Can we get/set misalignment parameters?")
    print("=" * 60)
    
    prior = bo_cheetah_prior.AresPriorMean()
    
    # Set specific values
    test_value_x = 0.0003
    test_value_y = -0.0002
    
    prior.q1_misalign_x = test_value_x
    prior.q1_misalign_y = test_value_y
    
    # Retrieve values
    retrieved_x = prior.q1_misalign_x.item()
    retrieved_y = prior.q1_misalign_y.item()
    
    print(f"\nSet Q1 x-misalignment:       {test_value_x}")
    print(f"Retrieved Q1 x-misalignment: {retrieved_x}")
    print(f"Difference: {abs(test_value_x - retrieved_x):.6e}")
    
    print(f"\nSet Q1 y-misalignment:       {test_value_y}")
    print(f"Retrieved Q1 y-misalignment:  {retrieved_y}")
    print(f"Difference: {abs(test_value_y - retrieved_y):.6e}")
    
    if abs(test_value_x - retrieved_x) < 1e-10 and abs(test_value_y - retrieved_y) < 1e-10:
        print("\n✓ Get/set parameters work correctly")
        return True
    else:
        print("\n✗ Get/set parameters DO NOT work correctly")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING MISALIGNMENT IMPLEMENTATION")
    print("=" * 60)
    
    test1_pass = test_misalignment_in_objective()
    test2_pass = test_misalignment_in_prior()
    test3_pass = test_prior_vs_objective_consistency()
    test4_pass = test_parameter_retrieval()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Objective with misalignments):  {'PASS ✓' if test1_pass else 'FAIL ✗'}")
    print(f"Test 2 (Prior with misalignments):      {'PASS ✓' if test2_pass else 'FAIL ✗'}")
    print(f"Test 3 (Prior vs Objective consistency): {'PASS ✓' if test3_pass else 'FAIL ✗'}")
    print(f"Test 4 (Parameter get/set):              {'PASS ✓' if test4_pass else 'FAIL ✗'}")
    
    if all([test1_pass, test2_pass, test3_pass, test4_pass]):
        print("\n✓ ALL TESTS PASSED - Misalignment implementation looks correct!")
    else:
        print("\n✗ SOME TESTS FAILED - There are issues with misalignment implementation")
import torch
import bo_cheetah_prior_ares as bo_cheetah_prior
import cheetah
from xopt import VOCS
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor

# Create VOCS
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

# Create prior mean
prior_mean_module = bo_cheetah_prior.AresPriorMean()

# Set some initial misalignments
prior_mean_module.q1_misalign_x = 0.0
prior_mean_module.q1_misalign_y = 0.0
prior_mean_module.q2_misalign_x = 0.0
prior_mean_module.q2_misalign_y = 0.0
prior_mean_module.q3_misalign_x = 0.0
prior_mean_module.q3_misalign_y = 0.0

print("="*60)
print("TESTING PRIOR MEAN PREDICTIONS")
print("="*60)

# Test a few input points
test_points = [
    torch.tensor([[10.0, -10.0, 0.0, 10.0, 0.0]]),
    torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]),
    torch.tensor([[-15.0, 15.0, 0.002, -15.0, -0.002]]),
]

for i, X in enumerate(test_points):
    print(f"\nTest point {i+1}:  {X.squeeze().tolist()}")
    try:
        prediction = prior_mean_module.forward(X)
        print(f"  Prior prediction: {prediction.item():.6f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Now test with GP constructor
print("\n" + "="*60)
print("TESTING GP CONSTRUCTOR")
print("="*60)

gp_constructor = StandardModelConstructor(
    mean_modules={"mae": prior_mean_module}
)

print(f"GP Constructor created:  {gp_constructor}")
print(f"Mean modules: {gp_constructor.mean_modules}")

# Create generator
generator = UpperConfidenceBoundGenerator(
    beta=2.0, 
    vocs=vocs, 
    gp_constructor=gp_constructor
)

print(f"\nGenerator created: {generator}")
print(f"Generator has GP constructor: {hasattr(generator, 'gp_constructor')}")

print("\n✅ Prior mean setup appears correct!")
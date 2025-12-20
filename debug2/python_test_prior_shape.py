import torch
import bo_cheetah_prior_ares as bo_cheetah_prior

# Create prior mean module
prior_mean_module = bo_cheetah_prior.AresPriorMean()

# Set some misalignment values
prior_mean_module.q1_misalign_x = 0.0001
prior_mean_module.q1_misalign_y = -0.0002

# Check the shapes
print("=" * 50)
print("CHECKING MISALIGNMENT PARAMETER SHAPES")
print("=" * 50)

print(f"\nq1_misalign_x:")
print(f"  Type: {type(prior_mean_module.q1_misalign_x)}")
print(f"  Value: {prior_mean_module.q1_misalign_x}")
print(f"  Shape: {prior_mean_module.q1_misalign_x.shape}")
print(f"  Is scalar (0-dim)?:  {prior_mean_module.q1_misalign_x.dim() == 0}")

print(f"\nq1_misalign_y:")
print(f"  Type: {type(prior_mean_module.q1_misalign_y)}")
print(f"  Value:  {prior_mean_module.q1_misalign_y}")
print(f"  Shape:  {prior_mean_module.q1_misalign_y.shape}")
print(f"  Is scalar (0-dim)?: {prior_mean_module.q1_misalign_y.dim() == 0}")

# Test torch.stack
stacked = torch.stack([
    prior_mean_module.q1_misalign_x,
    prior_mean_module.q1_misalign_y
])
print(f"\nAfter torch.stack([q1_misalign_x, q1_misalign_y]):")
print(f"  Shape: {stacked.shape}")
print(f"  Value: {stacked}")
print(f"  Dimensions: {stacked.dim()}")

# Compare with what your problem function does
problem_style = torch.tensor([0.0001, -0.0002])
print(f"\nProblem function style (torch.tensor([dx, dy])):")
print(f"  Shape: {problem_style.shape}")
print(f"  Value:  {problem_style}")
print(f"  Dimensions: {problem_style.dim()}")

# Check if they match
print(f"\nDo they match? ")
print(f"  Same shape?: {stacked.shape == problem_style.shape}")
print(f"  Same dimensions?: {stacked.dim() == problem_style.dim()}")

# Test what the forward method would do
print("\n" + "=" * 50)
print("TESTING FORWARD METHOD")
print("=" * 50)

# Create a dummy input
X = torch.tensor([[10.0, -10.0, 0.001, 10.0, -0.001]])  # Shape: (1, 5)
print(f"\nInput X shape: {X.shape}")

try:
    output = prior_mean_module.forward(X)
    print(f"✅ Forward pass succeeded!")
    print(f"Output:  {output}")
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Forward pass failed with error:")
    print(f"   {type(e).__name__}: {e}")
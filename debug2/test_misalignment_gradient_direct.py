"""Test if misalignment_matrix itself supports gradients."""
import torch
from cheetah.track_methods import misalignment_matrix

print("Testing misalignment_matrix gradient support...")

# Create misalignment with requires_grad
misalign = torch.tensor([0.0001, 0.0002], requires_grad=True)

print(f"Misalignment:  {misalign}")
print(f"Requires grad: {misalign.requires_grad}")

# Get entry and exit matrices
R_entry, R_exit = misalignment_matrix(misalign)

print(f"\nR_entry shape: {R_entry.shape}")
print(f"R_entry requires_grad: {R_entry.requires_grad}")
print(f"R_entry[0, 6]:  {R_entry[0, 6]} (should be -{misalign[0]})")
print(f"R_entry[2, 6]: {R_entry[2, 6]} (should be -{misalign[1]})")

# Create a dummy beam vector
beam = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], requires_grad=False)

# Apply entry matrix
beam_shifted = R_entry @ beam

print(f"\nBeam after R_entry:  {beam_shifted}")
print(f"Beam_shifted requires_grad: {beam_shifted.requires_grad}")

# Compute a loss (e.g., beam position)
loss = beam_shifted[0]**2 + beam_shifted[2]**2

print(f"\nLoss: {loss.item()}")
print(f"Loss requires_grad: {loss.requires_grad}")
print(f"Loss grad_fn: {loss.grad_fn}")

# Backward
loss.backward()

print(f"\nMisalignment gradient: {misalign.grad}")

if misalign.grad is None:
    print("❌ GRADIENT IS NONE - misalignment_matrix doesn't support gradients!")
elif (misalign.grad.abs() < 1e-9).all():
    print("⚠️  GRADIENT IS ZERO - either true zero or numerical issue")
else:
    print(f"✅ GRADIENT IS NON-ZERO: {misalign.grad}")
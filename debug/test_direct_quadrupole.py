import cheetah
import torch

print("Testing misalignment on directly-created quadrupole (not from JSON)...")

# Create quadrupoles directly (not from JSON)
quad1 = cheetah.Quadrupole(
    length=torch.tensor(0.2),
    k1=torch.tensor(10.0),
    misalignment=torch.tensor([0.0, 0.0]),
    name="Q1"
)

drift = cheetah.Drift(length=torch.tensor(0.5), name="D1")

segment = cheetah.Segment([quad1, drift])

# Create beam
incoming_beam = cheetah.ParticleBeam.from_parameters(
    num_particles=100000,
    sigma_x=torch.tensor(1e-4),
    sigma_y=torch.tensor(1e-3),
    sigma_px=torch.tensor(1e-4),
    sigma_py=torch.tensor(1e-4),
    energy=torch.tensor(100e6),
)

# Test 1: No misalignment
quad1.misalignment = torch.tensor([0.0, 0.0])
out_beam_no = segment(incoming_beam)
mae_no = 0.5 * (out_beam_no.sigma_x.abs() + out_beam_no.sigma_y.abs())

# Test 2: With misalignment
quad1.misalignment = torch.tensor([0.001, -0.001])
out_beam_yes = segment(incoming_beam)
mae_yes = 0.5 * (out_beam_yes.sigma_x.abs() + out_beam_yes.sigma_y.abs())

print(f"MAE without misalignment: {mae_no.item():.6e}")
print(f"MAE with misalignment:    {mae_yes.item():.6e}")
print(f"Difference:              {abs(mae_yes.item() - mae_no.item()):.6e}")

if abs(mae_yes.item() - mae_no.item()) > 1e-6:
    print("\n✓ Misalignments work on directly-created quadrupoles!")
else:
    print("\n✗ Misalignments don't work even on directly-created quadrupoles")
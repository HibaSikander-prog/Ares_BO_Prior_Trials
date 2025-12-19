import cheetah
import torch

# Load from JSON
ares_segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
ares_ea = ares_segment.subcell("AREASOLA1", "AREABSCR1")

print("Testing bmadx with LARGE misalignments (10mm instead of 1mm)...")

# Change to bmadx
ares_ea.AREAMQZM1.tracking_method = "bmadx"
ares_ea.AREAMQZM2.tracking_method = "bmadx"
ares_ea.AREAMQZM3.tracking_method = "bmadx"

# Use ParticleBeam
incoming_beam = cheetah.ParticleBeam.from_parameters(
    num_particles=50000,  
    sigma_x=torch.tensor(1e-4),
    sigma_y=torch.tensor(1e-3),
    sigma_px=torch.tensor(1e-4),
    sigma_py=torch.tensor(1e-4),
    energy=torch.tensor(100e6),
)

ares_ea.AREAMQZM1.k1 = torch.tensor(10.0)
ares_ea.AREAMQZM2.k1 = torch.tensor(-10.0)
ares_ea.AREAMQZM3.k1 = torch.tensor(10.0)
ares_ea.AREAMCVM1.angle = torch.tensor(0.0)
ares_ea.AREAMCHM1.angle = torch.tensor(0.0)

# Test 1: No misalignment
ares_ea.AREAMQZM1.misalignment = torch.tensor([0.0, 0.0])
ares_ea.AREAMQZM2.misalignment = torch.tensor([0.0, 0.0])
ares_ea.AREAMQZM3.misalignment = torch.tensor([0.0, 0.0])
out_beam_no = ares_ea(incoming_beam)
mae_no = 0.5 * (out_beam_no.sigma_x.abs() + out_beam_no.sigma_y.abs())

# Test 2: Small misalignment (1mm)
ares_ea.AREAMQZM1.misalignment = torch.tensor([0.001, -0.001])
ares_ea.AREAMQZM2.misalignment = torch.tensor([-0.001, 0.001])
ares_ea.AREAMQZM3.misalignment = torch.tensor([0.001, -0.001])
out_beam_small = ares_ea(incoming_beam)
mae_small = 0.5 * (out_beam_small.sigma_x.abs() + out_beam_small.sigma_y.abs())

# Test 3: LARGE misalignment (10mm)
ares_ea.AREAMQZM1.misalignment = torch.tensor([0.010, -0.010])
ares_ea.AREAMQZM2.misalignment = torch.tensor([-0.010, 0.010])
ares_ea.AREAMQZM3.misalignment = torch.tensor([0.010, -0.010])
out_beam_large = ares_ea(incoming_beam)
mae_large = 0.5 * (out_beam_large.sigma_x.abs() + out_beam_large.sigma_y.abs())

print(f"\nNo misalignment (0mm):     MAE = {mae_no.item():.6e}")
print(f"Small misalignment (1mm):  MAE = {mae_small.item():.6e}")
print(f"Large misalignment (10mm): MAE = {mae_large.item():.6e}")

diff_small = abs(mae_small.item() - mae_no.item())
diff_large = abs(mae_large.item() - mae_no.item())

print(f"\nDifference (1mm):  {diff_small:.6e}")
print(f"Difference (10mm): {diff_large:.6e}")
print(f"Ratio (large/small): {diff_large/diff_small if diff_small > 0 else 'N/A':.2f}")

if diff_large > 10 * diff_small:
    print("\n✓ Misalignments ARE working - effect scales with size!")
elif diff_large > 1e-6:
    print("\n⚠ Misalignments have SOME effect but it's very small")
else:
    print("\n✗ Misalignments don't seem to work")

# Also check beam centroid shift (mu_x, mu_y should change with misalignment)
print(f"\n--- Checking beam centroid (should shift with misalignment) ---")
print(f"No misalignment:  mu_x={out_beam_no.mu_x.item():.6e}, mu_y={out_beam_no.mu_y.item():.6e}")
print(f"Large misalignment: mu_x={out_beam_large.mu_x.item():.6e}, mu_y={out_beam_large.mu_y.item():.6e}")

mu_x_diff = abs(out_beam_large.mu_x.item() - out_beam_no.mu_x.item())
mu_y_diff = abs(out_beam_large.mu_y.item() - out_beam_no.mu_y.item())

print(f"Centroid shift:  Δmu_x={mu_x_diff:.6e}, Δmu_y={mu_y_diff:.6e}")

if mu_x_diff > 1e-6 or mu_y_diff > 1e-6:
    print("\n✓✓✓ SUCCESS!  Misalignments cause beam centroid shift!  ✓✓✓")
else:
    print("\n✗ No centroid shift detected")
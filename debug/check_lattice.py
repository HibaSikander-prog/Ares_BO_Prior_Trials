import cheetah
import torch

# Load lattice
ares_segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
ares_ea = ares_segment.subcell("AREASOLA1", "AREABSCR1")

# Check what misalignment values exist
print("AREAMQZM1 misalignment:", ares_ea.AREAMQZM1.misalignment)
print("AREAMQZM2 misalignment:", ares_ea.AREAMQZM2.misalignment)
print("AREAMQZM3 misalignment:", ares_ea.AREAMQZM3.misalignment)

# Try setting misalignment
print("\nSetting AREAMQZM1 misalignment to [0.001, -0.001]...")
ares_ea.AREAMQZM1.misalignment = torch.tensor([0.001, -0.001])
print("AREAMQZM1 misalignment after setting:", ares_ea.AREAMQZM1.misalignment)

# Check if it's actually being used
incoming_beam = cheetah.ParameterBeam.from_parameters(
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

# Run with no misalignment
ares_ea.AREAMQZM1.misalignment = torch.tensor([0.0, 0.0])
out_beam_no_misalign = ares_ea(incoming_beam)
mae_no_misalign = 0.5 * (out_beam_no_misalign.sigma_x.abs() + out_beam_no_misalign.sigma_y.abs())

# Run with misalignment
ares_ea.AREAMQZM1.misalignment = torch.tensor([0.001, -0.001])
out_beam_with_misalign = ares_ea(incoming_beam)
mae_with_misalign = 0.5 * (out_beam_with_misalign.sigma_x.abs() + out_beam_with_misalign.sigma_y.abs())

print(f"\nMAE without misalignment: {mae_no_misalign.item():.6e}")
print(f"MAE with misalignment:     {mae_with_misalign.item():.6e}")
print(f"Difference:               {abs(mae_with_misalign.item() - mae_no_misalign.item()):.6e}")
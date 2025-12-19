import cheetah
import torch

# Load from JSON
ares_segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
ares_ea = ares_segment.subcell("AREASOLA1", "AREABSCR1")

print("Tracking methods from JSON:")
print(f"AREAMQZM1: {ares_ea.AREAMQZM1.tracking_method}")

# Change to 'linear' (which supports misalignment via transfer maps)
print("\nChanging all quadrupoles to 'linear' tracking...")
ares_ea.AREAMQZM1.tracking_method = "linear"
ares_ea.AREAMQZM2.tracking_method = "linear"
ares_ea.AREAMQZM3.tracking_method = "linear"

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

# Test with no misalignment
ares_ea.AREAMQZM1.misalignment = torch.tensor([0.0, 0.0])
ares_ea.AREAMQZM2.misalignment = torch.tensor([0.0, 0.0])
ares_ea.AREAMQZM3.misalignment = torch.tensor([0.0, 0.0])
out_beam_no = ares_ea(incoming_beam)
mae_no = 0.5 * (out_beam_no.sigma_x.abs() + out_beam_no.sigma_y.abs())

# Test with misalignment
ares_ea.AREAMQZM1.misalignment = torch.tensor([0.001, -0.001])
ares_ea.AREAMQZM2.misalignment = torch.tensor([-0.001, 0.001])
ares_ea.AREAMQZM3.misalignment = torch.tensor([0.001, -0.001])
out_beam_yes = ares_ea(incoming_beam)
mae_yes = 0.5 * (out_beam_yes.sigma_x.abs() + out_beam_yes.sigma_y.abs())

print(f"\nWith 'linear' tracking:")
print(f"MAE without misalignment: {mae_no.item():.6e}")
print(f"MAE with misalignment:    {mae_yes.item():.6e}")
print(f"Difference:              {abs(mae_yes.item() - mae_no.item()):.6e}")

if abs(mae_yes.item() - mae_no.item()) > 1e-8:
    print("\n✓✓✓ SUCCESS!  Misalignments work with 'linear' tracking!  ✓✓✓")
    print("\nThis is the solution:")
    print("1.Change tracking_method from 'cheetah' to 'linear' in your JSON")
    print("2.OR set tracking_method after loading the lattice")
else:
    print("\n✗ Still doesn't work")
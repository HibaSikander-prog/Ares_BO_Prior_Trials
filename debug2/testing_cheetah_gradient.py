import cheetah
import torch

print("Testing if Cheetah supports gradients for misalignments...")

beam = cheetah.ParameterBeam.from_parameters(
    sigma_x=torch.tensor(1e-4),
    sigma_y=torch.tensor(2e-3),
    sigma_px=torch.tensor(1e-4),
    sigma_py=torch.tensor(1e-4),
    energy=torch.tensor(100e6),
)

segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
ares_ea = segment.subcell("AREASOLA1", "AREABSCR1")

# Test 1: Can we set misalignment as a tensor with requires_grad? 
misalign = torch.tensor([0.0001, 0.0002], requires_grad=True)
ares_ea.AREAMQZM1.k1 = torch.tensor(10.0)
ares_ea.AREAMQZM2.k1 = torch.tensor(-10.0)
ares_ea.AREAMCVM1.angle = torch.tensor(0.0)
ares_ea.AREAMQZM3.k1 = torch.tensor(10.0)
ares_ea.AREAMCHM1.angle = torch.tensor(0.0)

ares_ea.AREAMQZM1.misalignment = misalign

print(f"Misalignment tensor: {misalign}")
print(f"Requires grad: {misalign.requires_grad}")

# Simulate
out_beam = ares_ea(beam)
output = 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())

print(f"Output:  {output.item():.6f}")
print(f"Output requires_grad: {output.requires_grad}")

# Try backward
try:
    output.backward()
    print(f"\n✅ Backward pass succeeded!")
    print(f"Misalignment gradient: {misalign.grad}")
    
    if misalign.grad is None:
        print("❌ But gradient is None - Cheetah doesn't compute gradients for misalignments!")
    else:
        print("✅ Gradient computed successfully!")
except Exception as e: 
    print(f"\n❌ Backward pass failed: {e}")
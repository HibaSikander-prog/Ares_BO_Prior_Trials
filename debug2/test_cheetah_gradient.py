"""Minimal test to find where Cheetah breaks gradients."""

import torch
import torch.nn as nn
import cheetah

print("="*60)
print("MINIMAL CHEETAH GRADIENT TEST")
print("="*60)

# Load ARES lattice
print("\n1.Loading ARES lattice...")
ares_segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
ares_ea = ares_segment.subcell("AREASOLA1", "AREABSCR1")
print("   ✓ Lattice loaded")

# Create beam with requires_grad
print("\n2.Creating ParameterBeam...")
beam = cheetah.ParameterBeam.from_parameters(
    sigma_x=torch.tensor(1e-4),
    sigma_y=torch.tensor(2e-3),
    sigma_px=torch.tensor(1e-4),
    sigma_py=torch.tensor(1e-4),
    energy=torch.tensor(100e6),
)
print(f"   beam.mu requires_grad: {beam.mu.requires_grad}")
print(f"   beam.cov requires_grad: {beam.cov.requires_grad}")

# Set magnet strengths as parameters
print("\n3.Setting magnet strengths as nn.Parameter...")
ares_ea.AREAMQZM1.k1 = nn.Parameter(torch.tensor(10.0))
ares_ea.AREAMQZM2.k1 = nn.Parameter(torch.tensor(-10.0))
ares_ea.AREAMCVM1.angle = nn.Parameter(torch.tensor(0.0))
ares_ea.AREAMQZM3.k1 = nn.Parameter(torch.tensor(10.0))
ares_ea.AREAMCHM1.angle = nn.Parameter(torch.tensor(0.0))
print("   ✓ Magnets set as parameters")
print(f"   AREAMQZM1.k1 requires_grad: {ares_ea.AREAMQZM1.k1.requires_grad}")

# Set misalignments as parameters
print("\n4.Setting misalignments as nn.Parameter...")
ares_ea.AREAMQZM1.misalignment = nn.Parameter(torch.tensor([0.0001, -0.0001]))
print(f"   AREAMQZM1.misalignment requires_grad: {ares_ea.AREAMQZM1.misalignment.requires_grad}")

# Track beam
print("\n5.Tracking beam through lattice...")
try:
    out_beam = ares_ea.track(beam)
    print("   ✓ Tracking successful")
    print(f"   out_beam.mu requires_grad: {out_beam.mu.requires_grad}")
    print(f"   out_beam.cov requires_grad: {out_beam.cov.requires_grad}")
    print(f"   out_beam.mu grad_fn: {out_beam.mu.grad_fn}")
    
    # Try to compute objective
    print("\n6.Computing objective...")
    beam_size = 0.5 * (out_beam.sigma_x + out_beam.sigma_y)
    position_error = (out_beam.mu_x**2 + out_beam.mu_y**2).sqrt()
    objective = beam_size + 100.0 * position_error
    
    print(f"   objective value: {objective.item():.6f}")
    print(f"   objective requires_grad: {objective.requires_grad}")
    print(f"   objective grad_fn: {objective.grad_fn}")
    
    # Try backward
    print("\n7.Testing backward pass...")
    if objective.requires_grad:
        objective.backward()
        print("   ✓ Backward successful!")
        print(f"   AREAMQZM1.k1.grad: {ares_ea.AREAMQZM1.k1.grad}")
        print(f"   AREAMQZM1.misalignment.grad: {ares_ea.AREAMQZM1.misalignment.grad}")
    else:
        print("   ✗ Objective does not require grad - cannot backward!")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
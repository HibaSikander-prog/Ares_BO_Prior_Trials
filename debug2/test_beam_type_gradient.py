"""Test gradients with both ParameterBeam and ParticleBeam."""
import torch
import cheetah

print("="*60)
print("GRADIENT TEST: ParameterBeam vs ParticleBeam")
print("="*60)

def test_with_beam_type(beam_type_name):
    print(f"\n{'='*60}")
    print(f"Testing with {beam_type_name}")
    print(f"{'='*60}")
    
    # Create quadrupole with misalignment
    quad = cheetah.Quadrupole(
        length=torch.tensor(0.2),
        k1=torch.tensor(10.0),
        misalignment=torch.tensor([0.0001, 0.0002], requires_grad=True),
        tracking_method='cheetah'
    )
    
    # Create incoming beam
    if beam_type_name == "ParameterBeam":
        beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(1e-3),
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
    else:  # ParticleBeam
        beam = cheetah.ParticleBeam.from_parameters(
            num_particles=10000,
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(1e-3),
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
    
    print(f"Beam type: {type(beam).__name__}")
    print(f"Misalignment requires_grad: {quad.misalignment.requires_grad}")
    
    # Track
    out_beam = quad.track(beam)
    
    # Compute beam size
    beam_size = 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
    
    print(f"Beam size:  {beam_size.item():.6e}")
    print(f"Beam size requires_grad: {beam_size.requires_grad}")
    print(f"Beam size grad_fn: {beam_size.grad_fn}")
    
    # Try backward
    if beam_size.requires_grad:
        beam_size.backward()
        print(f"Misalignment gradient:  {quad.misalignment.grad}")
        
        if quad.misalignment.grad is None:
            print(f"❌ Gradient is None")
            return False
        elif (quad.misalignment.grad.abs() < 1e-9).all():
            print(f"⚠️  Gradient is ZERO:  {quad.misalignment.grad}")
            return False
        else: 
            print(f"✅ Gradient is NON-ZERO!")
            return True
    else: 
        print(f"❌ beam_size doesn't require grad - graph is broken")
        return False

# Test both
param_beam_works = test_with_beam_type("ParameterBeam")
particle_beam_works = test_with_beam_type("ParticleBeam")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"ParameterBeam gradients work: {param_beam_works}")
print(f"ParticleBeam gradients work: {particle_beam_works}")

if param_beam_works and particle_beam_works:
    print("\n✅ BOTH beam types support gradients!")
elif param_beam_works and not particle_beam_works: 
    print("\n⚠️  Only ParameterBeam works")
elif not param_beam_works and particle_beam_works:
    print("\n⚠️  Only ParticleBeam works - this is why I suggested switching")
else:
    print("\n❌ NEITHER works - there's a deeper issue")
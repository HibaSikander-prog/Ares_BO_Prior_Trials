"""Test if the prior mean parameters are actually trainable with Cheetah."""

import torch
import torch.nn as nn
import bo_cheetah_prior_ares as bo_cheetah_prior
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

print("="*60)
print("DIAGNOSTIC TEST:  Cheetah-Based Prior Mean Parameter Training")
print("="*60)

# 1.Create prior mean module
print("\n1.Creating AresPriorMean module...")
beam_params = {
    "sigma_x": 1e-4,
    "sigma_y":  1e-3,
    "sigma_px": 1e-4,
    "sigma_py":  1e-4,
    "energy": 100e6,
}
prior_mean = bo_cheetah_prior.AresPriorMean(beam_params=beam_params)

# Check initial values
print(f"   Initial q1_misalign_x: {prior_mean.q1_misalign_x.item():.6f}")
print(f"   Is parameter?  {isinstance(prior_mean.q1_misalign_x, torch.nn.Parameter)}")
print(f"   Requires grad? {prior_mean.q1_misalign_x.requires_grad}")

# 2.Test forward pass
print("\n2.Testing forward pass...")
X_test = torch.tensor([[10.0, -10.0, 0.0, 10.0, 0.0]])
output = prior_mean(X_test)
print(f"   Output:  {output.item():.6f}")
print(f"   Output requires_grad: {output.requires_grad}")
print(f"   Output grad_fn: {output.grad_fn}")

# 3.Test gradient computation
print("\n3.Testing gradient computation...")
prior_mean.zero_grad()
output = prior_mean(X_test)
try:
    output.backward()
    print(f"   ✓ Backward pass successful!")
    print(f"   q1_misalign_x gradient: {prior_mean.q1_misalign_x.grad}")
    if prior_mean.q1_misalign_x.grad is not None:
        print(f"   ✓ GRADIENTS ARE FLOWING!")
    else:
        print(f"   ✗ Gradient is None")
except Exception as e:
    print(f"   ✗ Backward pass failed: {e}")

# 4.Create GP model and test training
print("\n4.Creating GP model with trainable mean...")
train_X = torch.tensor([
    [10.0, -10.0, 0.0, 10.0, 0.0],
    [8.0, -8.0, 0.001, 9.0, -0.001],
    [12.0, -12.0, -0.001, 11.0, 0.001],
])
train_Y = torch.tensor([[0.1], [0.09], [0.11]])

try:
    model = SingleTaskGP(train_X, train_Y, mean_module=prior_mean)
    print(f"   ✓ Model created successfully")
    print(f"   Model type: {type(model)}")
    
    # Check if parameters are in the model
    print("\n5.Checking model parameters...")
    for name, param in model.named_parameters():
        if 'misalign' in name:
            print(f"   Found:  {name}, requires_grad={param.requires_grad}, value={param.item():.6f}")
    
    # 6.Test MLL and optimization step
    print("\n6.Testing optimization step...")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    
    # Store initial values
    initial_q1_x = prior_mean.q1_misalign_x.item()
    
    # Compute loss
    output = model(train_X)
    loss = -mll(output, train_Y.squeeze(-1))
    print(f"   Initial loss: {loss.item():.6f}")
    
    # Take optimization step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    print("\n7.Checking gradients after backward...")
    for name, param in model.named_parameters():
        if 'misalign' in name and param.grad is not None:
            print(f"   {name}: grad={param.grad.item():.6e}")
    
    optimizer.step()
    
    # Check if parameters changed
    print("\n8.Checking if parameters changed after optimizer step...")
    final_q1_x = prior_mean.q1_misalign_x.item()
    changed = abs(final_q1_x - initial_q1_x) > 1e-10
    print(f"   q1_misalign_x:  {initial_q1_x:.6f} -> {final_q1_x:.6f} {'✓ CHANGED' if changed else '✗ NO CHANGE'}")
    
    if changed:
        print("\n✓✓✓ SUCCESS!  Parameters are trainable!  ✓✓✓")
    else:
        print("\n✗✗✗ PROBLEM:  Parameters did not change ✗✗✗")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC TEST COMPLETE")
print("="*60)
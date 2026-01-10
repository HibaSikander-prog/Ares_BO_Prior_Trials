"""
ARES BO with LEARNABLE Misalignment Offsets - Using Calibration Approach

Key innovation: Two-phase learning to make offsets identifiable
1. Calibration Phase: Learn offsets from multiple fixed magnet configurations
2. Optimization Phase: Optimize magnets with learned offsets

This solves the identifiability problem by:
- Taking measurements at multiple diverse magnet settings
- Offsets cannot be "compensated" across all settings simultaneously
- Direct minimization of offset error (not MLL)
"""

from typing import Dict, Optional, List, Tuple

import cheetah
import torch
import torch.nn as nn
from gpytorch.constraints.constraints import Interval
from gpytorch.means import Mean
from gpytorch.priors import SmoothedBoxPrior
import numpy as np


def ensure_beam_float64(beam: cheetah.Beam) -> cheetah.Beam:
    """Ensure beam and all its internal tensors are float64."""
    if isinstance(beam, cheetah.ParameterBeam):
        return cheetah.ParameterBeam.from_parameters(
            mu_x=beam.mu_x.to(dtype=torch.float64),
            mu_px=beam.mu_px.to(dtype=torch.float64),
            mu_y=beam.mu_y.to(dtype=torch.float64),
            mu_py=beam.mu_py.to(dtype=torch.float64),
            sigma_x=beam.sigma_x.to(dtype=torch.float64),
            sigma_px=beam.sigma_px.to(dtype=torch.float64),
            sigma_y=beam.sigma_y.to(dtype=torch.float64),
            sigma_py=beam.sigma_py.to(dtype=torch.float64),
            sigma_tau=beam.sigma_tau.to(dtype=torch.float64),
            sigma_p=beam.sigma_p.to(dtype=torch.float64),
            energy=beam.energy.to(dtype=torch.float64),
            total_charge=beam.total_charge.to(dtype=torch.float64),
        )
    elif isinstance(beam, cheetah.ParticleBeam):
        return cheetah.ParticleBeam(
            particles=beam.particles.to(dtype=torch.float64),
            energy=beam.energy.to(dtype=torch.float64),
            particle_charges=beam.particle_charges.to(dtype=torch.float64) if beam.particle_charges is not None else None,
            device=beam.particles.device,
        )
    else:
        return beam.to(dtype=torch.float64)


def ares_problem_with_offsets(
    input_param: Dict[str, float],
    incoming_beam: Optional[cheetah.Beam] = None,
    beam_offsets: Optional[Dict[str, tuple]] = None,
) -> Dict[str, float]: 
    """Simulate ARES accelerator with beam position offsets."""
    if incoming_beam is None:
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            mu_x=torch.tensor(8.2413e-07),
            mu_px=torch.tensor(5.9885e-08),
            mu_y=torch.tensor(-1.7276e-06),
            mu_py=torch.tensor(-1.1746e-07),
            sigma_x=torch.tensor(0.0002),
            sigma_px=torch.tensor(3.6794e-06),
            sigma_y=torch.tensor(0.0002),
            sigma_py=torch.tensor(3.6941e-06),
            sigma_tau=torch.tensor(8.0116e-06),
            sigma_p=torch.tensor(0.0023),
            energy=torch.tensor(1.0732e+08),
            total_charge=torch.tensor(5.0e-13),
        )
    
    incoming_beam = ensure_beam_float64(incoming_beam)
    
    if beam_offsets is None:
        beam_offsets = {"q1": (0.0, 0.0), "q2": (0.0, 0.0), "q3": (0.0, 0.0)}
    
    q1_offset_x, q1_offset_y = beam_offsets.get("q1", (0.0, 0.0))
    q2_offset_x, q2_offset_y = beam_offsets.get("q2", (0.0, 0.0))
    q3_offset_x, q3_offset_y = beam_offsets.get("q3", (0.0, 0.0))
    
    segment_elements = [
        cheetah.Drift(length=torch.tensor(0.17504000663757324, dtype=torch.float64), name="Drift_to_Q1"),
        cheetah.Quadrupole(
            length=torch.tensor(0.12200000137090683, dtype=torch.float64),
            k1=torch.tensor(input_param["q1"], dtype=torch.float64),
            name="AREAMQZM1"
        ),
        cheetah.Drift(length=torch.tensor(0.42800000309944153, dtype=torch.float64), name="Drift_Q1_to_Q2"),
        cheetah.Quadrupole(
            length=torch.tensor(0.12200000137090683, dtype=torch.float64),
            k1=torch.tensor(input_param["q2"], dtype=torch.float64),
            name="AREAMQZM2"
        ),
        cheetah.Drift(length=torch.tensor(0.20399999618530273, dtype=torch.float64), name="Drift_Q2_to_CV"),
        cheetah.VerticalCorrector(
            length=torch.tensor(0.019999999552965164, dtype=torch.float64),
            angle=torch.tensor(input_param["cv"], dtype=torch.float64),
            name="AREAMCVM1"
        ),
        cheetah.Drift(length=torch.tensor(0.20399999618530273, dtype=torch.float64), name="Drift_CV_to_Q3"),
        cheetah.Quadrupole(
            length=torch.tensor(0.12200000137090683, dtype=torch.float64),
            k1=torch.tensor(input_param["q3"], dtype=torch.float64),
            name="AREAMQZM3"
        ),
        cheetah.Drift(length=torch.tensor(0.17900000512599945, dtype=torch.float64), name="Drift_Q3_to_CH"),
        cheetah.HorizontalCorrector(
            length=torch.tensor(0.019999999552965164, dtype=torch.float64),
            angle=torch.tensor(input_param["ch"], dtype=torch.float64),
            name="AREAMCHM1"
        ),
        cheetah.Drift(length=torch.tensor(0.44999998807907104, dtype=torch.float64), name="Drift_to_screen"),
    ]
    
    ares_segment = cheetah.Segment(elements=segment_elements)
    
    current_beam = incoming_beam
    
    # To Q1
    current_beam = ares_segment.elements[0](current_beam)
    # Q1 with misalignment
    current_beam = apply_beam_offset(current_beam, q1_offset_x, q1_offset_y)
    current_beam = ares_segment.elements[1](current_beam)
    current_beam = apply_beam_offset(current_beam, -q1_offset_x, -q1_offset_y)
    
    # To Q2
    current_beam = ares_segment.elements[2](current_beam)
    # Q2 with misalignment
    current_beam = apply_beam_offset(current_beam, q2_offset_x, q2_offset_y)
    current_beam = ares_segment.elements[3](current_beam)
    current_beam = apply_beam_offset(current_beam, -q2_offset_x, -q2_offset_y)
    
    # Through corrector section
    current_beam = ares_segment.elements[4](current_beam)
    current_beam = ares_segment.elements[5](current_beam)
    current_beam = ares_segment.elements[6](current_beam)
    
    # Q3 with misalignment
    current_beam = apply_beam_offset(current_beam, q3_offset_x, q3_offset_y)
    current_beam = ares_segment.elements[7](current_beam)
    current_beam = apply_beam_offset(current_beam, -q3_offset_x, -q3_offset_y)
    
    # Final section
    current_beam = ares_segment.elements[8](current_beam)
    current_beam = ares_segment.elements[9](current_beam)
    current_beam = ares_segment.elements[10](current_beam)
    
    out_beam = current_beam
    
    ares_beam_mae = 0.25 * (
        out_beam.mu_x.abs() + 
        out_beam.sigma_x.abs() + 
        out_beam.mu_y.abs() + 
        out_beam.sigma_y.abs()
    )
    
    return {
        "mae": ares_beam_mae.detach().numpy(),
        "mu_x": out_beam.mu_x.detach().numpy(),
        "mu_y": out_beam.mu_y.detach().numpy(),
        "sigma_x":  out_beam.sigma_x.detach().numpy(),
        "sigma_y": out_beam.sigma_y.detach().numpy(),
    }


def apply_beam_offset(beam:  cheetah.Beam, offset_x: float, offset_y: float) -> cheetah.Beam:
    """Apply position offset to beam (simulates misalignment effect)."""
    if isinstance(beam, cheetah.ParameterBeam):
        new_beam = cheetah.ParameterBeam.from_parameters(
            mu_x=beam.mu_x + torch.tensor(offset_x, dtype=beam.mu_x.dtype),
            mu_px=beam.mu_px,
            mu_y=beam.mu_y + torch.tensor(offset_y, dtype=beam.mu_y.dtype),
            mu_py=beam.mu_py,
            sigma_x=beam.sigma_x,
            sigma_px=beam.sigma_px,
            sigma_y=beam.sigma_y,
            sigma_py=beam.sigma_py,
            sigma_tau=beam.sigma_tau,
            sigma_p=beam.sigma_p,
            energy=beam.energy,
            total_charge=beam.total_charge,
        )
        return ensure_beam_float64(new_beam)
    else:
        return beam


def calibrate_offsets(
    incoming_beam: cheetah.Beam,
    true_beam_offsets: Dict[str, Tuple[float, float]],
    n_calibration_points: int = 15,
    learning_rate: float = 0.001,
    n_iterations: int = 200,
    verbose: bool = True
) -> Dict[str, Tuple[float, float]]:
    """
    Learn misalignment offsets from measurements at multiple magnet configurations.
    
    This is the KEY to making offsets learnable:
    - Takes measurements at many different magnet settings
    - Offsets cannot be compensated across all settings simultaneously
    - Direct optimization of offset values to match observations
    
    Args:
        incoming_beam: The beam configuration
        true_beam_offsets: Ground truth offsets (simulates real measurements)
        n_calibration_points: Number of diverse magnet configurations to measure
        learning_rate: Learning rate for offset optimization
        n_iterations: Number of optimization iterations
        verbose: Print progress
        
    Returns:
        learned_offsets: Dictionary with learned offset values
    """
    
    if verbose:
        print("\n" + "="*80)
        print("CALIBRATION PHASE: Learning Misalignment Offsets")
        print("="*80)
        print(f"Strategy: Measure at {n_calibration_points} diverse magnet configurations")
        print(f"Ground truth offsets:")
        for key, (x, y) in true_beam_offsets.items():
            print(f"  {key}: x={x*1e6:+.1f}μm, y={y*1e6:+.1f}μm")
    
    # Generate diverse calibration points (magnet settings)
    # Use a spread of values across the operating range
    np.random.seed(42)  # Reproducible
    calibration_configs = []
    
    for i in range(n_calibration_points):
        config = {
            "q1": np.random.uniform(-25, 25),
            "q2": np.random.uniform(-25, 25),
            "cv": np.random.uniform(-0.005, 0.005),
            "q3": np.random.uniform(-25, 25),
            "ch": np.random.uniform(-0.005, 0.005),
        }
        calibration_configs.append(config)
    
    # Take "measurements" at these configurations (simulate real data)
    measurements = []
    for config in calibration_configs:
        result = ares_problem_with_offsets(
            config, 
            incoming_beam=incoming_beam,
            beam_offsets=true_beam_offsets
        )
        measurements.append(result)
    
    if verbose:
        print(f"\nCollected {len(measurements)} measurements")
        print(f"MAE range: [{min(m['mae'] for m in measurements)*1000:.3f}, {max(m['mae'] for m in measurements)*1000:.3f}] mm")
    
    # Initialize learnable offset parameters
    learned_q1_offset_x = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))
    learned_q1_offset_y = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))
    learned_q2_offset_x = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))
    learned_q2_offset_y = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))
    learned_q3_offset_x = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))
    learned_q3_offset_y = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, requires_grad=True))
    
    params = [learned_q1_offset_x, learned_q1_offset_y, 
              learned_q2_offset_x, learned_q2_offset_y,
              learned_q3_offset_x, learned_q3_offset_y]
    
    # Optimizer for offset learning
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    # Optimization loop
    if verbose:
        print(f"\nOptimizing offsets ({n_iterations} iterations)...")
        print("Progress: ", end="")
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        # Compute predictions with current offset estimates
        total_loss = 0.0
        
        for config, measurement in zip(calibration_configs, measurements):
            # Build current offset guess
            current_offsets = {
                "q1": (learned_q1_offset_x.item(), learned_q1_offset_y.item()),
                "q2": (learned_q2_offset_x.item(), learned_q2_offset_y.item()),
                "q3": (learned_q3_offset_x.item(), learned_q3_offset_y.item()),
            }
            
            # Predict what we would measure
            prediction = ares_problem_with_offsets(
                config,
                incoming_beam=incoming_beam,
                beam_offsets=current_offsets
            )
            
            # Loss: difference between prediction and actual measurement
            # Use all observables for better identifiability
            loss = (
                (prediction["mae"] - measurement["mae"])**2 +
                0.1 * (prediction["mu_x"] - measurement["mu_x"])**2 +
                0.1 * (prediction["mu_y"] - measurement["mu_y"])**2 +
                0.1 * (prediction["sigma_x"] - measurement["sigma_x"])**2 +
                0.1 * (prediction["sigma_y"] - measurement["sigma_y"])**2
            )
            
            total_loss += loss
        
        # Backprop through physics simulation
        total_loss_tensor = torch.tensor(total_loss, requires_grad=True)
        
        # Manual gradient computation (since we're going through numpy)
        # We need to compute finite differences
        eps = 1e-7
        grads = []
        
        for param in params:
            original_value = param.item()
            
            # Forward difference
            param.data = torch.tensor(original_value + eps, dtype=torch.float64)
            loss_plus = 0.0
            for config, measurement in zip(calibration_configs, measurements):
                current_offsets = {
                    "q1": (learned_q1_offset_x.item(), learned_q1_offset_y.item()),
                    "q2": (learned_q2_offset_x.item(), learned_q2_offset_y.item()),
                    "q3": (learned_q3_offset_x.item(), learned_q3_offset_y.item()),
                }
                prediction = ares_problem_with_offsets(config, incoming_beam, current_offsets)
                loss_plus += (
                    (prediction["mae"] - measurement["mae"])**2 +
                    0.1 * (prediction["mu_x"] - measurement["mu_x"])**2 +
                    0.1 * (prediction["mu_y"] - measurement["mu_y"])**2 +
                    0.1 * (prediction["sigma_x"] - measurement["sigma_x"])**2 +
                    0.1 * (prediction["sigma_y"] - measurement["sigma_y"])**2
                )
            
            # Restore and compute gradient
            param.data = torch.tensor(original_value, dtype=torch.float64)
            grad = (loss_plus - total_loss) / eps
            grads.append(grad)
        
        # Manual parameter update with gradient clipping and constraints
        with torch.no_grad():
            for param, grad in zip(params, grads):
                # Clip gradients
                grad_clipped = np.clip(grad, -1.0, 1.0)
                
                # Update
                param.data -= learning_rate * grad_clipped
                
                # Project onto constraints (±0.0005 m = ±500 μm)
                param.data = torch.clamp(param.data, -0.0005, 0.0005)
        
        # Print progress
        if verbose and (iteration + 1) % 20 == 0:
            print(f"{iteration+1}", end="." if (iteration + 1) < n_iterations else "\n")
    
    # Extract learned values
    learned_offsets = {
        "q1": (learned_q1_offset_x.item(), learned_q1_offset_y.item()),
        "q2": (learned_q2_offset_x.item(), learned_q2_offset_y.item()),
        "q3": (learned_q3_offset_x.item(), learned_q3_offset_y.item()),
    }
    
    if verbose:
        print("\n" + "="*80)
        print("CALIBRATION COMPLETE")
        print("="*80)
        print("\nLearned offsets:")
        for key, (x, y) in learned_offsets.items():
            print(f"  {key}: x={x*1e6:+.1f}μm, y={y*1e6:+.1f}μm")
        
        print("\nGround truth offsets:")
        for key, (x, y) in true_beam_offsets.items():
            print(f"  {key}: x={x*1e6:+.1f}μm, y={y*1e6:+.1f}μm")
        
        print("\nErrors:")
        errors = []
        for key in learned_offsets.keys():
            ex = abs(learned_offsets[key][0] - true_beam_offsets[key][0]) * 1e6
            ey = abs(learned_offsets[key][1] - true_beam_offsets[key][1]) * 1e6
            print(f"  {key}: x_error={ex:.1f}μm, y_error={ey:.1f}μm")
            errors.extend([ex, ey])
        
        avg_error = np.mean(errors)
        print(f"\nAverage absolute error: {avg_error:.1f} μm")
        
        if avg_error < 50:
            print("✅ EXCELLENT calibration! (error < 50 μm)")
        elif avg_error < 100:
            print("✅ GOOD calibration! (error < 100 μm)")
        else:
            print("⚠️  Calibration could be better (error > 100 μm)")
    
    return learned_offsets


class AresPriorMeanRevised(Mean):
    """
    ARES Lattice as a prior mean function with FIXED offsets.
    
    Offsets are learned separately in calibration phase, then fixed here.
    """

    def __init__(self, incoming_beam: Optional[cheetah.Beam] = None, 
                 fixed_offsets: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__()
        
        if incoming_beam is None:
            incoming_beam = cheetah.ParameterBeam.from_parameters(
                mu_x=torch.tensor(8.2413e-07),
                mu_px=torch.tensor(5.9885e-08),
                mu_y=torch.tensor(-1.7276e-06),
                mu_py=torch.tensor(-1.1746e-07),
                sigma_x=torch.tensor(0.0002),
                sigma_px=torch.tensor(3.6794e-06),
                sigma_y=torch.tensor(0.0002),
                sigma_py=torch.tensor(3.6941e-06),
                sigma_tau=torch.tensor(8.0116e-06),
                sigma_p=torch.tensor(0.0023),
                energy=torch.tensor(1.0732e+08),
                total_charge=torch.tensor(5.0e-13),
            )
        
        incoming_beam = ensure_beam_float64(incoming_beam)
        self.incoming_beam = incoming_beam
        
        # Build ARES segment
        self.D0 = cheetah.Drift(length=torch.tensor(0.175, dtype=torch.float64), name="Drift_to_Q1")
        self.Q1 = cheetah.Quadrupole(
            length=torch.tensor(0.122, dtype=torch.float64), 
            k1=torch.tensor(0.0, dtype=torch.float64), 
            name="AREAMQZM1"
        )
        self.D1 = cheetah.Drift(length=torch.tensor(0.428, dtype=torch.float64), name="Drift_Q1_to_Q2")
        self.Q2 = cheetah.Quadrupole(
            length=torch.tensor(0.122, dtype=torch.float64), 
            k1=torch.tensor(0.0, dtype=torch.float64), 
            name="AREAMQZM2"
        )
        self.D2 = cheetah.Drift(length=torch.tensor(0.204, dtype=torch.float64), name="Drift_Q2_to_CV")
        self.CV = cheetah.VerticalCorrector(
            length=torch.tensor(0.020, dtype=torch.float64), 
            angle=torch.tensor(0.0, dtype=torch.float64), 
            name="AREAMCVM1"
        )
        self.D3 = cheetah.Drift(length=torch.tensor(0.204, dtype=torch.float64), name="Drift_CV_to_Q3")
        self.Q3 = cheetah.Quadrupole(
            length=torch.tensor(0.122, dtype=torch.float64), 
            k1=torch.tensor(0.0, dtype=torch.float64), 
            name="AREAMQZM3"
        )
        self.D4 = cheetah.Drift(length=torch.tensor(0.179, dtype=torch.float64), name="Drift_Q3_to_CH")
        self.CH = cheetah.HorizontalCorrector(
            length=torch.tensor(0.020, dtype=torch.float64), 
            angle=torch.tensor(0.0, dtype=torch.float64), 
            name="AREAMCHM1"
        )
        self.D5 = cheetah.Drift(length=torch.tensor(0.450, dtype=torch.float64), name="Drift_final")
        
        self.segment = cheetah.Segment(elements=[
            self.D0, self.Q1, self.D1, self.Q2, self.D2,
            self.CV, self.D3, self.Q3, self.D4, self.CH, self.D5
        ])
        
        # Store FIXED offsets (learned from calibration)
        if fixed_offsets is None:
            fixed_offsets = {
                "q1": (0.0, 0.0),
                "q2": (0.0, 0.0),
                "q3": (0.0, 0.0),
            }
        
        self.q1_offset_x = torch.tensor(fixed_offsets["q1"][0], dtype=torch.float64)
        self.q1_offset_y = torch.tensor(fixed_offsets["q1"][1], dtype=torch.float64)
        self.q2_offset_x = torch.tensor(fixed_offsets["q2"][0], dtype=torch.float64)
        self.q2_offset_y = torch.tensor(fixed_offsets["q2"][1], dtype=torch.float64)
        self.q3_offset_x = torch.tensor(fixed_offsets["q3"][0], dtype=torch.float64)
        self.q3_offset_y = torch.tensor(fixed_offsets["q3"][1], dtype=torch.float64)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass with fixed offsets."""
        X = X.to(dtype=torch.float64)
        
        # Set magnet strengths
        self.Q1.k1 = X[..., 0]
        self.Q2.k1 = X[..., 1]
        self.CV.angle = X[..., 2]
        self.Q3.k1 = X[..., 3]
        self.CH.angle = X[..., 4]
        
        current_beam = self.incoming_beam
        
        # To Q1
        current_beam = self.D0(current_beam)
        current_beam = self._apply_offset(current_beam, self.q1_offset_x, self.q1_offset_y)
        current_beam = self.Q1(current_beam)
        current_beam = self._apply_offset(current_beam, -self.q1_offset_x, -self.q1_offset_y)
        
        # To Q2
        current_beam = self.D1(current_beam)
        current_beam = self._apply_offset(current_beam, self.q2_offset_x, self.q2_offset_y)
        current_beam = self.Q2(current_beam)
        current_beam = self._apply_offset(current_beam, -self.q2_offset_x, -self.q2_offset_y)
        
        # Through corrector section
        current_beam = self.D2(current_beam)
        current_beam = self.CV(current_beam)
        current_beam = self.D3(current_beam)
        
        # Q3
        current_beam = self._apply_offset(current_beam, self.q3_offset_x, self.q3_offset_y)
        current_beam = self.Q3(current_beam)
        current_beam = self._apply_offset(current_beam, -self.q3_offset_x, -self.q3_offset_y)
        
        # Final section
        current_beam = self.D4(current_beam)
        current_beam = self.CH(current_beam)
        current_beam = self.D5(current_beam)
        
        out_beam = current_beam
        
        ares_beam_mae = 0.25 * (
            out_beam.mu_x.abs() + 
            out_beam.sigma_x.abs() + 
            out_beam.mu_y.abs() + 
            out_beam.sigma_y.abs()
        )
        
        return ares_beam_mae
    
    def _apply_offset(self, beam, offset_x, offset_y):
        """Apply beam offset"""
        if torch.is_tensor(offset_x):
            offset_x = offset_x.to(dtype=beam.mu_x.dtype)
        else:
            offset_x = torch.tensor(offset_x, dtype=beam.mu_x.dtype)
            
        if torch.is_tensor(offset_y):
            offset_y = offset_y.to(dtype=beam.mu_y.dtype)
        else:
            offset_y = torch.tensor(offset_y, dtype=beam.mu_y.dtype)
        
        new_beam = cheetah.ParameterBeam.from_parameters(
            mu_x=beam.mu_x + offset_x,
            mu_px=beam.mu_px,
            mu_y=beam.mu_y + offset_y,
            mu_py=beam.mu_py,
            sigma_x=beam.sigma_x,
            sigma_px=beam.sigma_px,
            sigma_y=beam.sigma_y,
            sigma_py=beam.sigma_py,
            sigma_tau=beam.sigma_tau,
            sigma_p=beam.sigma_p,
            energy=beam.energy,
            total_charge=beam.total_charge,
        )
        return ensure_beam_float64(new_beam)

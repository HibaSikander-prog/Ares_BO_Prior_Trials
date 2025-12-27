"""
ARES BO Prior - REVISED VERSION using FODO-style manual segment creation

Key changes from original:
1. Manually create segment elements (like FODO does)
2. Use learnable offset parameters instead of misalignment attribute
3. Apply offsets via beam transformation (simple assignment)
4. Avoid touching Cheetah's misalignment attribute (which may not be differentiable)
"""

from typing import Dict, Optional

import cheetah
import torch
import torch.nn as nn
from gpytorch.constraints.constraints import Interval
from gpytorch.means import Mean
from gpytorch.priors import SmoothedBoxPrior


# ARES Problem - Modified to accept beam offsets instead of misalignments
def ares_problem_with_offsets(
    input_param: Dict[str, float],
    incoming_beam: Optional[cheetah.Beam] = None,
<<<<<<< Updated upstream
    misalignment_config: Optional[Dict[str, tuple]] = None,
=======
    beam_offsets: Optional[Dict[str, tuple]] = None,
>>>>>>> Stashed changes
) -> Dict[str, float]:
    """
    Simulate ARES accelerator with beam position offsets.
    
    Instead of moving magnets (misalignment), we offset the incoming beam position
    at each quadrupole. This is mathematically equivalent but uses differentiable operations.
    
    Args:
        input_param: Dictionary with magnet strengths
            - q1: AREAMQZM1 strength
            - q2: AREAMQZM2 strength
            - cv: AREAMCVM1 angle (vertical corrector)
            - q3: AREAMQZM3 strength
            - ch: AREAMCHM1 angle (horizontal corrector)
        incoming_beam: Beam parameters (uses default if None)
        beam_offsets: Dictionary with beam offsets at each quadrupole
            - Format: {"q1": (x_offset, y_offset), "q2": ..., "q3": ...}
            - Units: meters
    
    Returns:
<<<<<<< Updated upstream
        Dictionary with beam size metrics (mse, log_mse, mae, log_mae)
    """
    if incoming_beam is None:
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(2e-3),
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
=======
        Dictionary with beam metrics (mae, mu_x, mu_y, sigma_x, sigma_y)
    """
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
>>>>>>> Stashed changes
        )
    
    # Extract beam offsets (default to zero)
    if beam_offsets is None:
        beam_offsets = {"q1": (0.0, 0.0), "q2": (0.0, 0.0), "q3": (0.0, 0.0)}
    
<<<<<<< Updated upstream
    # Set magnet strengths
    ares_ea.AREAMQZM1.k1 = torch.tensor(input_param["q1"])
    ares_ea.AREAMQZM2.k1 = torch.tensor(input_param["q2"])
    ares_ea.AREAMCVM1.angle = torch.tensor(input_param["cv"])
    ares_ea.AREAMQZM3.k1 = torch.tensor(input_param["q3"])
    ares_ea.AREAMCHM1.angle = torch.tensor(input_param["ch"])
    
    # Apply misalignments if provided
    if misalignment_config is not None:
        for magnet_name, (dx, dy) in misalignment_config.items():
            magnet = getattr(ares_ea, magnet_name)
            magnet.misalignment = torch.tensor([dx, dy])
=======
    q1_offset_x, q1_offset_y = beam_offsets.get("q1", (0.0, 0.0))
    q2_offset_x, q2_offset_y = beam_offsets.get("q2", (0.0, 0.0))
    q3_offset_x, q3_offset_y = beam_offsets.get("q3", (0.0, 0.0))
    
    # Manually build the ARES segment from JSON specs
    # Using exact lengths from ARESlatticeStage3v1_9.json
    
    segment_elements = [
        # Initial drift to AREAMQZM1
        cheetah.Drift(length=torch.tensor(0.17504000663757324), name="Drift_to_Q1"),
        
        # Q1 with beam offset applied
        cheetah.Quadrupole(
            length=torch.tensor(0.12200000137090683),
            k1=torch.tensor(input_param["q1"]),
            name="AREAMQZM1"
        ),
        
        # Drift to Q2
        cheetah.Drift(length=torch.tensor(0.42800000309944153), name="Drift_Q1_to_Q2"),
        
        # Q2 with beam offset applied
        cheetah.Quadrupole(
            length=torch.tensor(0.12200000137090683),
            k1=torch.tensor(input_param["q2"]),
            name="AREAMQZM2"
        ),
        
        # Drift to vertical corrector
        cheetah.Drift(length=torch.tensor(0.20399999618530273), name="Drift_Q2_to_CV"),
        
        # Vertical corrector
        cheetah.VerticalCorrector(
            length=torch.tensor(0.019999999552965164),
            angle=torch.tensor(input_param["cv"]),
            name="AREAMCVM1"
        ),
        
        # Drift to Q3
        cheetah.Drift(length=torch.tensor(0.20399999618530273), name="Drift_CV_to_Q3"),
        
        # Q3 with beam offset applied
        cheetah.Quadrupole(
            length=torch.tensor(0.12200000137090683),
            k1=torch.tensor(input_param["q3"]),
            name="AREAMQZM3"
        ),
        
        # Drift to horizontal corrector
        cheetah.Drift(length=torch.tensor(0.17900000512599945), name="Drift_Q3_to_CH"),
        
        # Horizontal corrector
        cheetah.HorizontalCorrector(
            length=torch.tensor(0.019999999552965164),
            angle=torch.tensor(input_param["ch"]),
            name="AREAMCHM1"
        ),
        
        # Final drift
        cheetah.Drift(length=torch.tensor(0.44999998807907104), name="Drift_to_screen"),
    ]
>>>>>>> Stashed changes
    
    ares_segment = cheetah.Segment(elements=segment_elements)
    
<<<<<<< Updated upstream
    # Calculate metrics
    beam_size_mse = 0.5 * (out_beam.sigma_x**2 + out_beam.sigma_y**2)
    beam_size_mae = 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
=======
    # Apply beam offsets by modifying beam parameters before each quadrupole
    # This is done by propagating through segment piece by piece
    current_beam = incoming_beam
    
    # Propagate to Q1
    current_beam = ares_segment.elements[0](current_beam)  # Drift to Q1
    
    # Apply Q1 offset
    current_beam = apply_beam_offset(current_beam, q1_offset_x, q1_offset_y)
    
    # Through Q1
    current_beam = ares_segment.elements[1](current_beam)  # Q1
    
    # Propagate to Q2
    current_beam = ares_segment.elements[2](current_beam)  # Drift to Q2
    
    # Apply Q2 offset
    current_beam = apply_beam_offset(current_beam, q2_offset_x, q2_offset_y)
    
    # Through Q2
    current_beam = ares_segment.elements[3](current_beam)  # Q2
    
    # Through corrector section
    current_beam = ares_segment.elements[4](current_beam)  # Drift to CV
    current_beam = ares_segment.elements[5](current_beam)  # CV
    current_beam = ares_segment.elements[6](current_beam)  # Drift to Q3
    
    # Apply Q3 offset
    current_beam = apply_beam_offset(current_beam, q3_offset_x, q3_offset_y)
    
    # Through Q3 and final elements
    current_beam = ares_segment.elements[7](current_beam)  # Q3
    current_beam = ares_segment.elements[8](current_beam)  # Drift to CH
    current_beam = ares_segment.elements[9](current_beam)  # CH
    current_beam = ares_segment.elements[10](current_beam)  # Final drift
    
    out_beam = current_beam
    
    # Calculate objective (same as before)
    ares_beam_mae = 0.25 * (
        out_beam.mu_x.abs() + 
        out_beam.sigma_x.abs() + 
        out_beam.mu_y.abs() + 
        out_beam.sigma_y.abs()
    )
>>>>>>> Stashed changes
    
    return {
        "mse": beam_size_mse.detach().numpy(),
        "log_mse": beam_size_mse.log().detach().numpy(),
        "mae": beam_size_mae.detach().numpy(),
        "log_mae": beam_size_mae.log().detach().numpy(),
    }


<<<<<<< Updated upstream
# Prior Mean Functions for BO
class AresPriorMean(Mean):
    """ARES Lattice as a prior mean function for BO."""
=======
def apply_beam_offset(beam: cheetah.Beam, offset_x: float, offset_y: float) -> cheetah.Beam:
    """
    Apply position offset to beam (simulates misalignment effect).
    
    This is mathematically equivalent to moving the magnet, but uses
    simple parameter assignment (like FODO drift_length).
    """
    # For ParameterBeam, we can directly modify mu_x and mu_y
    if isinstance(beam, cheetah.ParameterBeam):
        # Create new beam with offset applied
        return cheetah.ParameterBeam.from_parameters(
            mu_x=beam.mu_x + torch.tensor(offset_x),
            mu_px=beam.mu_px,
            mu_y=beam.mu_y + torch.tensor(offset_y),
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
    else:
        # For other beam types, just return as-is
        # (Would need different approach for ParticleBeam)
        return beam


# Prior Mean Functions for BO - REVISED VERSION
class AresPriorMeanRevised(Mean):
    """
    ARES Lattice as a prior mean function - REVISED to use FODO-style approach.
    
    Key differences from original:
    1. Manually creates segment elements (like FodoPriorMean)
    2. Uses learnable offset parameters (like drift_length)
    3. Applies offsets via beam transformation (simple assignment)
    4. Avoids Cheetah's misalignment attribute
    """
>>>>>>> Stashed changes

    def __init__(self, incoming_beam: Optional[cheetah.Beam] = None):
        super().__init__()
        
        if incoming_beam is None:
            incoming_beam = cheetah.ParameterBeam.from_parameters(
<<<<<<< Updated upstream
                sigma_x=torch.tensor(1e-4),
                sigma_y=torch.tensor(2e-3),
                sigma_px=torch.tensor(1e-4),
                sigma_py=torch.tensor(1e-4),
                energy=torch.tensor(100e6),
=======
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
>>>>>>> Stashed changes
            )
        self.incoming_beam = incoming_beam
        
        # Manually build ARES segment (like FODO builds FODO segment)
        self.Q1 = cheetah.Quadrupole(
            length=torch.tensor(0.122), k1=torch.tensor(0.0), name="AREAMQZM1"
        )
        self.D1 = cheetah.Drift(length=torch.tensor(0.428), name="Drift_Q1_to_Q2")
        
<<<<<<< Updated upstream
        # Define learnable misalignment parameters (6 parameters:  x,y for 3 quadrupoles)
        # Constraint: misalignments between -0.5mm to 0.5mm (-0.0005m to 0.0005m)
        misalignment_constraint = Interval(-0.0005, 0.0005)
=======
        self.Q2 = cheetah.Quadrupole(
            length=torch.tensor(0.122), k1=torch.tensor(0.0), name="AREAMQZM2"
        )
        self.D2 = cheetah.Drift(length=torch.tensor(0.204), name="Drift_Q2_to_CV")
        
        self.CV = cheetah.VerticalCorrector(
            length=torch.tensor(0.020), angle=torch.tensor(0.0), name="AREAMCVM1"
        )
        self.D3 = cheetah.Drift(length=torch.tensor(0.204), name="Drift_CV_to_Q3")
>>>>>>> Stashed changes
        
        self.Q3 = cheetah.Quadrupole(
            length=torch.tensor(0.122), k1=torch.tensor(0.0), name="AREAMQZM3"
        )
        self.D4 = cheetah.Drift(length=torch.tensor(0.179), name="Drift_Q3_to_CH")
        
        self.CH = cheetah.HorizontalCorrector(
            length=torch.tensor(0.020), angle=torch.tensor(0.0), name="AREAMCHM1"
        )
        self.D5 = cheetah.Drift(length=torch.tensor(0.450), name="Drift_final")
        
        # Initial drift
        self.D0 = cheetah.Drift(length=torch.tensor(0.175), name="Drift_to_Q1")
        
        self.segment = cheetah.Segment(elements=[
            self.D0, self.Q1, self.D1, self.Q2, self.D2,
            self.CV, self.D3, self.Q3, self.D4, self.CH, self.D5
        ])
        
        # Define learnable offset parameters (like drift_length in FODO)
        # Constraint: offsets between -0.5mm to 0.5mm
        offset_constraint = Interval(-0.0005, 0.0005)
        
        # Q1 offsets
        self.register_parameter("raw_q1_offset_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q1_offset_y", nn.Parameter(torch.tensor(0.0)))
        self.register_prior(
            "q1_offset_x_prior", SmoothedBoxPrior(-0.0005, 0.0005),
            self._q1_offset_x_param, self._set_q1_offset_x
        )
        self.register_prior(
            "q1_offset_y_prior", SmoothedBoxPrior(-0.0005, 0.0005),
            self._q1_offset_y_param, self._set_q1_offset_y
        )
        self.register_constraint("raw_q1_offset_x", offset_constraint)
        self.register_constraint("raw_q1_offset_y", offset_constraint)
        
        # Q2 offsets
        self.register_parameter("raw_q2_offset_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q2_offset_y", nn.Parameter(torch.tensor(0.0)))
        self.register_prior(
            "q2_offset_x_prior", SmoothedBoxPrior(-0.0005, 0.0005),
            self._q2_offset_x_param, self._set_q2_offset_x
        )
        self.register_prior(
            "q2_offset_y_prior", SmoothedBoxPrior(-0.0005, 0.0005),
            self._q2_offset_y_param, self._set_q2_offset_y
        )
        self.register_constraint("raw_q2_offset_x", offset_constraint)
        self.register_constraint("raw_q2_offset_y", offset_constraint)
        
        # Q3 offsets
        self.register_parameter("raw_q3_offset_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q3_offset_y", nn.Parameter(torch.tensor(0.0)))
        self.register_prior(
            "q3_offset_x_prior", SmoothedBoxPrior(-0.0005, 0.0005),
            self._q3_offset_x_param, self._set_q3_offset_x
        )
        self.register_prior(
            "q3_offset_y_prior", SmoothedBoxPrior(-0.0005, 0.0005),
            self._q3_offset_y_param, self._set_q3_offset_y
        )
        self.register_constraint("raw_q3_offset_x", offset_constraint)
        self.register_constraint("raw_q3_offset_y", offset_constraint)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using FODO-style simple assignment.
        
        Args:
            X: Tensor of shape (..., 5) with columns [q1, q2, cv, q3, ch]
        
        Returns:
<<<<<<< Updated upstream
            Predicted beam size (MAE)
=======
            Predicted MAE
>>>>>>> Stashed changes
        """
        # Set magnet strengths (simple assignment, like FODO)
        self.Q1.k1 = X[..., 0]
        self.Q2.k1 = X[..., 1]
        self.CV.angle = X[..., 2]
        self.Q3.k1 = X[..., 3]
        self.CH.angle = X[..., 4]
        
<<<<<<< Updated upstream
        # Apply learnable misalignments
        self.ares_ea.AREAMQZM1.misalignment = torch.stack([
            self.q1_misalign_x, 
            self.q1_misalign_y
        ])
        self.ares_ea.AREAMQZM2.misalignment = torch.stack([
            self.q2_misalign_x,
            self.q2_misalign_y
        ])
        self.ares_ea.AREAMQZM3.misalignment = torch.stack([
            self.q3_misalign_x,
            self.q3_misalign_y
        ])
        
        # Simulate beam propagation
        out_beam = self.ares_ea(self.incoming_beam)
        beam_size_mae = 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
        
        return beam_size_mae

    # Properties and setters for Q1 misalignments
=======
        # Propagate beam with offsets applied at each quadrupole
        current_beam = self.incoming_beam
        
        # To Q1
        current_beam = self.D0(current_beam)
        
        # Apply Q1 offset (simple assignment!)
        current_beam = self._apply_offset(current_beam, 
                                          self.q1_offset_x, self.q1_offset_y)
        
        # Through Q1
        current_beam = self.Q1(current_beam)
        
        # To Q2
        current_beam = self.D1(current_beam)
        
        # Apply Q2 offset (simple assignment!)
        current_beam = self._apply_offset(current_beam,
                                          self.q2_offset_x, self.q2_offset_y)
        
        # Through Q2
        current_beam = self.Q2(current_beam)
        
        # Through corrector section
        current_beam = self.D2(current_beam)
        current_beam = self.CV(current_beam)
        current_beam = self.D3(current_beam)
        
        # Apply Q3 offset (simple assignment!)
        current_beam = self._apply_offset(current_beam,
                                          self.q3_offset_x, self.q3_offset_y)
        
        # Through Q3 and rest
        current_beam = self.Q3(current_beam)
        current_beam = self.D4(current_beam)
        current_beam = self.CH(current_beam)
        current_beam = self.D5(current_beam)
        
        out_beam = current_beam
        
        # Calculate MAE
        ares_beam_mae = 0.25 * (
            out_beam.mu_x.abs() + 
            out_beam.sigma_x.abs() + 
            out_beam.mu_y.abs() + 
            out_beam.sigma_y.abs()
        )
        
        return ares_beam_mae
    
    def _apply_offset(self, beam, offset_x, offset_y):
        """Apply beam offset (like FODO assigns drift_length)"""
        return cheetah.ParameterBeam.from_parameters(
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
    
    # Properties for Q1 offsets (like drift_length in FODO)
>>>>>>> Stashed changes
    @property
    def q1_offset_x(self):
        return self._q1_offset_x_param(self)
    
    @q1_offset_x.setter
    def q1_offset_x(self, value):
        self._set_q1_offset_x(self, value)
    
    def _q1_offset_x_param(self, m):
        return m.raw_q1_offset_x_constraint.transform(self.raw_q1_offset_x)
    
    def _set_q1_offset_x(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q1_offset_x)
        m.initialize(
            raw_q1_offset_x=m.raw_q1_offset_x_constraint.inverse_transform(value)
        )
    
    @property
    def q1_offset_y(self):
        return self._q1_offset_y_param(self)
    
<<<<<<< Updated upstream
    @q1_misalign_y.setter
    def q1_misalign_y(self, value:  torch.Tensor):
        self._set_q1_misalign_y(self, value)
=======
    @q1_offset_y.setter
    def q1_offset_y(self, value):
        self._set_q1_offset_y(self, value)
>>>>>>> Stashed changes
    
    def _q1_offset_y_param(self, m):
        return m.raw_q1_offset_y_constraint.transform(self.raw_q1_offset_y)
    
    def _set_q1_offset_y(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q1_offset_y)
        m.initialize(
            raw_q1_offset_y=m.raw_q1_offset_y_constraint.inverse_transform(value)
        )
    
    # Properties for Q2 offsets
    @property
    def q2_offset_x(self):
        return self._q2_offset_x_param(self)
    
<<<<<<< Updated upstream
    @q2_misalign_x.setter
    def q2_misalign_x(self, value:  torch.Tensor):
        self._set_q2_misalign_x(self, value)
=======
    @q2_offset_x.setter
    def q2_offset_x(self, value):
        self._set_q2_offset_x(self, value)
>>>>>>> Stashed changes
    
    def _q2_offset_x_param(self, m):
        return m.raw_q2_offset_x_constraint.transform(self.raw_q2_offset_x)
    
    def _set_q2_offset_x(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q2_offset_x)
        m.initialize(
            raw_q2_offset_x=m.raw_q2_offset_x_constraint.inverse_transform(value)
        )
    
    @property
    def q2_offset_y(self):
        return self._q2_offset_y_param(self)
    
<<<<<<< Updated upstream
    @q2_misalign_y.setter
    def q2_misalign_y(self, value: torch.Tensor):
        self._set_q2_misalign_y(self, value)
=======
    @q2_offset_y.setter
    def q2_offset_y(self, value):
        self._set_q2_offset_y(self, value)
>>>>>>> Stashed changes
    
    def _q2_offset_y_param(self, m):
        return m.raw_q2_offset_y_constraint.transform(self.raw_q2_offset_y)
    
    def _set_q2_offset_y(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q2_offset_y)
        m.initialize(
            raw_q2_offset_y=m.raw_q2_offset_y_constraint.inverse_transform(value)
        )
    
    # Properties for Q3 offsets
    @property
    def q3_offset_x(self):
        return self._q3_offset_x_param(self)
    
<<<<<<< Updated upstream
    @q3_misalign_x.setter
    def q3_misalign_x(self, value: torch.Tensor):
        self._set_q3_misalign_x(self, value)
=======
    @q3_offset_x.setter
    def q3_offset_x(self, value):
        self._set_q3_offset_x(self, value)
>>>>>>> Stashed changes
    
    def _q3_offset_x_param(self, m):
        return m.raw_q3_offset_x_constraint.transform(self.raw_q3_offset_x)
    
    def _set_q3_offset_x(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q3_offset_x)
        m.initialize(
            raw_q3_offset_x=m.raw_q3_offset_x_constraint.inverse_transform(value)
        )
    
    @property
    def q3_offset_y(self):
        return self._q3_offset_y_param(self)
    
    @q3_offset_y.setter
    def q3_offset_y(self, value):
        self._set_q3_offset_y(self, value)
    
    def _q3_offset_y_param(self, m):
        return m.raw_q3_offset_y_constraint.transform(self.raw_q3_offset_y)
    
    def _set_q3_offset_y(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q3_offset_y)
        m.initialize(
            raw_q3_offset_y=m.raw_q3_offset_y_constraint.inverse_transform(value)
        )
<<<<<<< Updated upstream
=======
"""
ARES BO Prior - SIMPLE FIX using element.misalignment directly

Key insight: Cheetah's element.misalignment IS differentiable!
The problem was using .item() which breaks gradients.

Solution: Assign tensors directly without .item()
"""

>>>>>>> Stashed changes
from typing import Dict, Optional

import cheetah
import torch
import torch.nn as nn
from gpytorch.constraints.constraints import Interval
from gpytorch.means import Mean
from gpytorch.priors import SmoothedBoxPrior


<<<<<<< Updated upstream
# ARES Problem
=======
>>>>>>> Stashed changes
def ares_problem(
    input_param: Dict[str, float],
    incoming_beam: Optional[cheetah.Beam] = None,
    misalignment_config: Optional[Dict[str, tuple]] = None,
) -> Dict[str, float]:
<<<<<<< Updated upstream
    """
    Simulate ARES accelerator and return beam quality metrics.
    
    Args:
        input_param: Dictionary with magnet strengths
            - q1: AREAMQZM1 strength
            - q2: AREAMQZM2 strength
            - cv: AREAMCVM1 angle (vertical corrector)
            - q3: AREAMQZM3 strength
            - ch: AREAMCHM1 angle (horizontal corrector)
        incoming_beam:  Beam parameters (uses default if None)
        misalignment_config: Dictionary with misalignments for each quadrupole
            - Format: {"AREAMQZM1": (x, y), "AREAMQZM2": (x, y), "AREAMQZM3": (x, y)}
            - Units: meters
    
    Returns:
        Dictionary with beam size metrics (mse, log_mse, mae, log_mae)
    """
=======
    """Simulate ARES with quadrupole misalignments."""
>>>>>>> Stashed changes
    if incoming_beam is None:
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(2e-3),
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
    
<<<<<<< Updated upstream
    # Load ARES lattice and extract the section of interest
    ares_segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
    ares_ea = ares_segment.subcell("AREASOLA1", "AREABSCR1")
    
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
    
    # Simulate beam propagation
    out_beam = ares_ea(incoming_beam)
    
    # Calculate metrics
    beam_size_mse = 0.5 * (out_beam.sigma_x**2 + out_beam.sigma_y**2)
    beam_size_mae = 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
=======
    # Extract misalignments
    if misalignment_config is None:
        misalignment_config = {
            "AREAMQZM1": (0.0, 0.0),
            "AREAMQZM2": (0.0, 0.0),
            "AREAMQZM3": (0.0, 0.0),
        }
    
    # Build segment manually
    D0 = cheetah.Drift(length=torch.tensor(0.17504000663757324), name="Drift_to_Q1")
    Q1 = cheetah.Quadrupole(
        length=torch.tensor(0.12200000137090683),
        k1=torch.tensor(input_param["q1"]),
        name="AREAMQZM1"
    )
    D1 = cheetah.Drift(length=torch.tensor(0.42800000309944153), name="Drift_Q1_to_Q2")
    Q2 = cheetah.Quadrupole(
        length=torch.tensor(0.12200000137090683),
        k1=torch.tensor(input_param["q2"]),
        name="AREAMQZM2"
    )
    D2 = cheetah.Drift(length=torch.tensor(0.20399999618530273), name="Drift_Q2_to_CV")
    CV = cheetah.VerticalCorrector(
        length=torch.tensor(0.019999999552965164),
        angle=torch.tensor(input_param["cv"]),
        name="AREAMCVM1"
    )
    D3 = cheetah.Drift(length=torch.tensor(0.20399999618530273), name="Drift_CV_to_Q3")
    Q3 = cheetah.Quadrupole(
        length=torch.tensor(0.12200000137090683),
        k1=torch.tensor(input_param["q3"]),
        name="AREAMQZM3"
    )
    D4 = cheetah.Drift(length=torch.tensor(0.17900000512599945), name="Drift_Q3_to_CH")
    CH = cheetah.HorizontalCorrector(
        length=torch.tensor(0.019999999552965164),
        angle=torch.tensor(input_param["ch"]),
        name="AREAMCHM1"
    )
    D5 = cheetah.Drift(length=torch.tensor(0.44999998807907104), name="Drift_final")
    
    # Set misalignments directly
    q1_mis_x, q1_mis_y = misalignment_config.get("AREAMQZM1", (0.0, 0.0))
    q2_mis_x, q2_mis_y = misalignment_config.get("AREAMQZM2", (0.0, 0.0))
    q3_mis_x, q3_mis_y = misalignment_config.get("AREAMQZM3", (0.0, 0.0))
    
    Q1.misalignment = torch.tensor([q1_mis_x, q1_mis_y], dtype=torch.float32)
    Q2.misalignment = torch.tensor([q2_mis_x, q2_mis_y], dtype=torch.float32)
    Q3.misalignment = torch.tensor([q3_mis_x, q3_mis_y], dtype=torch.float32)
    
    # Create segment and propagate
    segment = cheetah.Segment(elements=[D0, Q1, D1, Q2, D2, CV, D3, Q3, D4, CH, D5])
    out_beam = segment(incoming_beam)
    
    # Calculate MAE
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
class AresPriorMean(Mean):
    """
    ARES prior with learnable misalignments.
    
    SIMPLE FIX: Use element.misalignment directly with torch.stack (no .item())
    """
>>>>>>> Stashed changes

    def __init__(self, incoming_beam: Optional[cheetah.Beam] = None):
        super().__init__()
        
        if incoming_beam is None:
            incoming_beam = cheetah.ParameterBeam.from_parameters(
                sigma_x=torch.tensor(1e-4),
                sigma_y=torch.tensor(2e-3),
                sigma_px=torch.tensor(1e-4),
                sigma_py=torch.tensor(1e-4),
                energy=torch.tensor(100e6),
            )
        self.incoming_beam = incoming_beam
        
<<<<<<< Updated upstream
        # Load ARES lattice
        ares_segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
        self.ares_ea = ares_segment.subcell("AREASOLA1", "AREABSCR1")
        
        # Define learnable misalignment parameters (6 parameters:  x,y for 3 quadrupoles)
        # Constraint: misalignments between -0.5mm to 0.5mm (-0.0005m to 0.0005m)
        misalignment_constraint = Interval(-0.0005, 0.0005)
        
        # AREAMQZM1 misalignments
        self.register_parameter("raw_q1_misalign_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q1_misalign_y", nn.Parameter(torch.tensor(0.0)))
        self.register_prior(
            "q1_misalign_x_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            self._q1_misalign_x_param,
            self._set_q1_misalign_x,
        )
        self.register_prior(
            "q1_misalign_y_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            self._q1_misalign_y_param,
            self._set_q1_misalign_y,
        )
        self.register_constraint("raw_q1_misalign_x", misalignment_constraint)
        self.register_constraint("raw_q1_misalign_y", misalignment_constraint)
        
        # AREAMQZM2 misalignments
        self.register_parameter("raw_q2_misalign_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q2_misalign_y", nn.Parameter(torch.tensor(0.0)))
        self.register_prior(
            "q2_misalign_x_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            self._q2_misalign_x_param,
            self._set_q2_misalign_x,
        )
        self.register_prior(
            "q2_misalign_y_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            self._q2_misalign_y_param,
            self._set_q2_misalign_y,
        )
        self.register_constraint("raw_q2_misalign_x", misalignment_constraint)
        self.register_constraint("raw_q2_misalign_y", misalignment_constraint)
        
        # AREAMQZM3 misalignments
        self.register_parameter("raw_q3_misalign_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q3_misalign_y", nn.Parameter(torch.tensor(0.0)))
        self.register_prior(
            "q3_misalign_x_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            self._q3_misalign_x_param,
            self._set_q3_misalign_x,
        )
        self.register_prior(
            "q3_misalign_y_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            self._q3_misalign_y_param,
            self._set_q3_misalign_y,
        )
=======
        # Build segment elements (FODO style)
        self.D0 = cheetah.Drift(length=torch.tensor(0.17504000663757324), name="Drift_to_Q1")
        self.Q1 = cheetah.Quadrupole(length=torch.tensor(0.12200000137090683), k1=torch.tensor(0.0), name="AREAMQZM1")
        self.D1 = cheetah.Drift(length=torch.tensor(0.42800000309944153), name="Drift_Q1_to_Q2")
        self.Q2 = cheetah.Quadrupole(length=torch.tensor(0.12200000137090683), k1=torch.tensor(0.0), name="AREAMQZM2")
        self.D2 = cheetah.Drift(length=torch.tensor(0.20399999618530273), name="Drift_Q2_to_CV")
        self.CV = cheetah.VerticalCorrector(length=torch.tensor(0.019999999552965164), angle=torch.tensor(0.0), name="AREAMCVM1")
        self.D3 = cheetah.Drift(length=torch.tensor(0.20399999618530273), name="Drift_CV_to_Q3")
        self.Q3 = cheetah.Quadrupole(length=torch.tensor(0.12200000137090683), k1=torch.tensor(0.0), name="AREAMQZM3")
        self.D4 = cheetah.Drift(length=torch.tensor(0.17900000512599945), name="Drift_Q3_to_CH")
        self.CH = cheetah.HorizontalCorrector(length=torch.tensor(0.019999999552965164), angle=torch.tensor(0.0), name="AREAMCHM1")
        self.D5 = cheetah.Drift(length=torch.tensor(0.44999998807907104), name="Drift_final")
        
        self.segment = cheetah.Segment(elements=[
            self.D0, self.Q1, self.D1, self.Q2, self.D2,
            self.CV, self.D3, self.Q3, self.D4, self.CH, self.D5
        ])
        
        # Learnable misalignment parameters (6 total: x,y for 3 quads)
        misalignment_constraint = Interval(-0.0005, 0.0005)
        
        # Q1 misalignments
        self.register_parameter("raw_q1_misalign_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q1_misalign_y", nn.Parameter(torch.tensor(0.0)))
        self.register_prior("q1_misalign_x_prior", SmoothedBoxPrior(-0.0005, 0.0005), self._q1_misalign_x_param, self._set_q1_misalign_x)
        self.register_prior("q1_misalign_y_prior", SmoothedBoxPrior(-0.0005, 0.0005), self._q1_misalign_y_param, self._set_q1_misalign_y)
        self.register_constraint("raw_q1_misalign_x", misalignment_constraint)
        self.register_constraint("raw_q1_misalign_y", misalignment_constraint)
        
        # Q2 misalignments
        self.register_parameter("raw_q2_misalign_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q2_misalign_y", nn.Parameter(torch.tensor(0.0)))
        self.register_prior("q2_misalign_x_prior", SmoothedBoxPrior(-0.0005, 0.0005), self._q2_misalign_x_param, self._set_q2_misalign_x)
        self.register_prior("q2_misalign_y_prior", SmoothedBoxPrior(-0.0005, 0.0005), self._q2_misalign_y_param, self._set_q2_misalign_y)
        self.register_constraint("raw_q2_misalign_x", misalignment_constraint)
        self.register_constraint("raw_q2_misalign_y", misalignment_constraint)
        
        # Q3 misalignments
        self.register_parameter("raw_q3_misalign_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q3_misalign_y", nn.Parameter(torch.tensor(0.0)))
        self.register_prior("q3_misalign_x_prior", SmoothedBoxPrior(-0.0005, 0.0005), self._q3_misalign_x_param, self._set_q3_misalign_x)
        self.register_prior("q3_misalign_y_prior", SmoothedBoxPrior(-0.0005, 0.0005), self._q3_misalign_y_param, self._set_q3_misalign_y)
>>>>>>> Stashed changes
        self.register_constraint("raw_q3_misalign_x", misalignment_constraint)
        self.register_constraint("raw_q3_misalign_y", misalignment_constraint)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
<<<<<<< Updated upstream
        Forward pass through the prior mean.
        
        Args:
            X:  Tensor of shape (..., 5) with columns [q1, q2, cv, q3, ch]
        
        Returns:
            Predicted beam size (MAE)
        """
        # Set magnet strengths from input
        self.ares_ea.AREAMQZM1.k1 = X[..., 0]
        self.ares_ea.AREAMQZM2.k1 = X[..., 1]
        self.ares_ea.AREAMCVM1.angle = X[..., 2]
        self.ares_ea.AREAMQZM3.k1 = X[..., 3]
        self.ares_ea.AREAMCHM1.angle = X[..., 4]
        
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
        Forward pass - SIMPLE FIX: Use torch.stack instead of .item()
        """
        # Set magnet strengths
        self.Q1.k1 = X[..., 0]
        self.Q2.k1 = X[..., 1]
        self.CV.angle = X[..., 2]
        self.Q3.k1 = X[..., 3]
        self.CH.angle = X[..., 4]
        
        # SIMPLE FIX: Set misalignments using torch.stack (NO .item()!)
        # This maintains gradient flow through the computational graph
        self.Q1.misalignment = torch.stack([self.q1_misalign_x, self.q1_misalign_y], dim=-1)
        self.Q2.misalignment = torch.stack([self.q2_misalign_x, self.q2_misalign_y], dim=-1)
        self.Q3.misalignment = torch.stack([self.q3_misalign_x, self.q3_misalign_y], dim=-1)
        
        # Propagate beam
        out_beam = self.segment(self.incoming_beam)
        
        # Calculate MAE
        ares_beam_mae = 0.25 * (
            out_beam.mu_x.abs() + 
            out_beam.sigma_x.abs() + 
            out_beam.mu_y.abs() + 
            out_beam.sigma_y.abs()
        )
        
        return ares_beam_mae
    
    # Properties for Q1 misalignments
>>>>>>> Stashed changes
    @property
    def q1_misalign_x(self):
        return self._q1_misalign_x_param(self)
    
    @q1_misalign_x.setter
<<<<<<< Updated upstream
    def q1_misalign_x(self, value:  torch.Tensor):
=======
    def q1_misalign_x(self, value):
>>>>>>> Stashed changes
        self._set_q1_misalign_x(self, value)
    
    def _q1_misalign_x_param(self, m):
        return m.raw_q1_misalign_x_constraint.transform(self.raw_q1_misalign_x)
    
<<<<<<< Updated upstream
    def _set_q1_misalign_x(self, m, value:  torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q1_misalign_x)
        m.initialize(
            raw_q1_misalign_x=m.raw_q1_misalign_x_constraint.inverse_transform(value)
        )
=======
    def _set_q1_misalign_x(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q1_misalign_x)
        m.initialize(raw_q1_misalign_x=m.raw_q1_misalign_x_constraint.inverse_transform(value))
>>>>>>> Stashed changes
    
    @property
    def q1_misalign_y(self):
        return self._q1_misalign_y_param(self)
    
    @q1_misalign_y.setter
<<<<<<< Updated upstream
    def q1_misalign_y(self, value:  torch.Tensor):
=======
    def q1_misalign_y(self, value):
>>>>>>> Stashed changes
        self._set_q1_misalign_y(self, value)
    
    def _q1_misalign_y_param(self, m):
        return m.raw_q1_misalign_y_constraint.transform(self.raw_q1_misalign_y)
    
<<<<<<< Updated upstream
    def _set_q1_misalign_y(self, m, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q1_misalign_y)
        m.initialize(
            raw_q1_misalign_y=m.raw_q1_misalign_y_constraint.inverse_transform(value)
        )
    
    # Properties and setters for Q2 misalignments
=======
    def _set_q1_misalign_y(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q1_misalign_y)
        m.initialize(raw_q1_misalign_y=m.raw_q1_misalign_y_constraint.inverse_transform(value))
    
    # Q2 properties
>>>>>>> Stashed changes
    @property
    def q2_misalign_x(self):
        return self._q2_misalign_x_param(self)
    
    @q2_misalign_x.setter
<<<<<<< Updated upstream
    def q2_misalign_x(self, value:  torch.Tensor):
=======
    def q2_misalign_x(self, value):
>>>>>>> Stashed changes
        self._set_q2_misalign_x(self, value)
    
    def _q2_misalign_x_param(self, m):
        return m.raw_q2_misalign_x_constraint.transform(self.raw_q2_misalign_x)
    
<<<<<<< Updated upstream
    def _set_q2_misalign_x(self, m, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q2_misalign_x)
        m.initialize(
            raw_q2_misalign_x=m.raw_q2_misalign_x_constraint.inverse_transform(value)
        )
=======
    def _set_q2_misalign_x(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q2_misalign_x)
        m.initialize(raw_q2_misalign_x=m.raw_q2_misalign_x_constraint.inverse_transform(value))
>>>>>>> Stashed changes
    
    @property
    def q2_misalign_y(self):
        return self._q2_misalign_y_param(self)
    
    @q2_misalign_y.setter
<<<<<<< Updated upstream
    def q2_misalign_y(self, value: torch.Tensor):
=======
    def q2_misalign_y(self, value):
>>>>>>> Stashed changes
        self._set_q2_misalign_y(self, value)
    
    def _q2_misalign_y_param(self, m):
        return m.raw_q2_misalign_y_constraint.transform(self.raw_q2_misalign_y)
    
<<<<<<< Updated upstream
    def _set_q2_misalign_y(self, m, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q2_misalign_y)
        m.initialize(
            raw_q2_misalign_y=m.raw_q2_misalign_y_constraint.inverse_transform(value)
        )
    
    # Properties and setters for Q3 misalignments
=======
    def _set_q2_misalign_y(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q2_misalign_y)
        m.initialize(raw_q2_misalign_y=m.raw_q2_misalign_y_constraint.inverse_transform(value))
    
    # Q3 properties
>>>>>>> Stashed changes
    @property
    def q3_misalign_x(self):
        return self._q3_misalign_x_param(self)
    
    @q3_misalign_x.setter
<<<<<<< Updated upstream
    def q3_misalign_x(self, value: torch.Tensor):
=======
    def q3_misalign_x(self, value):
>>>>>>> Stashed changes
        self._set_q3_misalign_x(self, value)
    
    def _q3_misalign_x_param(self, m):
        return m.raw_q3_misalign_x_constraint.transform(self.raw_q3_misalign_x)
    
<<<<<<< Updated upstream
    def _set_q3_misalign_x(self, m, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q3_misalign_x)
        m.initialize(
            raw_q3_misalign_x=m.raw_q3_misalign_x_constraint.inverse_transform(value)
        )
=======
    def _set_q3_misalign_x(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q3_misalign_x)
        m.initialize(raw_q3_misalign_x=m.raw_q3_misalign_x_constraint.inverse_transform(value))
>>>>>>> Stashed changes
    
    @property
    def q3_misalign_y(self):
        return self._q3_misalign_y_param(self)
    
    @q3_misalign_y.setter
<<<<<<< Updated upstream
    def q3_misalign_y(self, value: torch.Tensor):
=======
    def q3_misalign_y(self, value):
>>>>>>> Stashed changes
        self._set_q3_misalign_y(self, value)
    
    def _q3_misalign_y_param(self, m):
        return m.raw_q3_misalign_y_constraint.transform(self.raw_q3_misalign_y)
    
<<<<<<< Updated upstream
    def _set_q3_misalign_y(self, m, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q3_misalign_y)
        m.initialize(
            raw_q3_misalign_y=m.raw_q3_misalign_y_constraint.inverse_transform(value)
        )
=======
    def _set_q3_misalign_y(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q3_misalign_y)
        m.initialize(raw_q3_misalign_y=m.raw_q3_misalign_y_constraint.inverse_transform(value))
>>>>>>> Stashed changes

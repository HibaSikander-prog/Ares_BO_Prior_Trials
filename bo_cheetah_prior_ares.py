from typing import Dict, Optional

import cheetah
import torch
import torch.nn as nn
from gpytorch.constraints.constraints import Interval
from gpytorch.means import Mean
from gpytorch.priors import SmoothedBoxPrior


# ARES Problem (UNCHANGED - this is correct)
def ares_problem(
    input_param: Dict[str, float],
    incoming_beam: Optional[cheetah.Beam] = None,
    misalignment_config: Optional[Dict[str, tuple]] = None,
) -> Dict[str, float]:
    """
    Simulate ARES accelerator and return beam quality metrics.
    
    Args:
        input_param: Dictionary with magnet strengths
            - q1: AREAMQZM1 strength
            - q2: AREAMQZM2 strength
            - cv: AREAMCVM1 angle (vertical corrector)
            - q3: AREAMQZM3 strength
            - ch:  AREAMCHM1 angle (horizontal corrector)
        incoming_beam:  Beam parameters (uses default if None)
        misalignment_config: Dictionary with misalignments for each quadrupole
            - Format: {"AREAMQZM1": (x, y), "AREAMQZM2": (x, y), "AREAMQZM3": (x, y)}
            - Units: meters
    
    Returns:
        Dictionary with beam metrics (mae, position_error, combined objective)
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
        )
    
    # Load ARES lattice and extract the section of interest
    ares_segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
    ares_ea = ares_segment.subcell("AREASOLA1", "AREABSCR1")
    
    # Set magnet strengths (uses dictionary keys - order doesn't matter here)
    ares_ea.AREAMQZM1.k1 = torch.tensor(input_param["q1"])
    ares_ea.AREAMQZM2.k1 = torch.tensor(input_param["q2"])
    ares_ea.AREAMCVM1.angle = torch.tensor(input_param["cv"])
    ares_ea.AREAMQZM3.k1 = torch.tensor(input_param["q3"])
    ares_ea.AREAMCHM1.angle = torch.tensor(input_param["ch"])
    
    # Apply misalignments if provided
    if misalignment_config is not None:
        for magnet_name, (dx, dy) in misalignment_config.items():
            magnet = getattr(ares_ea, magnet_name)
            magnet.misalignment = torch.tensor([dx, dy], dtype=torch.float32)
    else:
        # Explicitly set to zero if no misalignments provided
        ares_ea.AREAMQZM1.misalignment = torch.tensor([0.0, 0.0])
        ares_ea.AREAMQZM2.misalignment = torch.tensor([0.0, 0.0])
        ares_ea.AREAMQZM3.misalignment = torch.tensor([0.0, 0.0])
    
    # Simulate beam propagation
    out_beam = ares_ea(incoming_beam)
    
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
        "sigma_x": out_beam.sigma_x.detach().numpy(),
        "sigma_y": out_beam.sigma_y.detach().numpy(),
    }


# Prior Mean Functions for BO (FIXED)
class AresPriorMean(Mean):
    """ARES Lattice as a prior mean function for BO with trainable misalignments."""
    
    # IMPORTANT:  Xopt orders variables ALPHABETICALLY when converting to tensors
    # VOCS variables:  q1, q2, cv, q3, ch
    # Alphabetical order: ch, cv, q1, q2, q3
    # So tensor indices are: ch=0, cv=1, q1=2, q2=3, q3=4
    
    # Define the mapping from tensor index to variable name
    VAR_ORDER = ['ch', 'cv', 'q1', 'q2', 'q3']  # Alphabetical order
    IDX_CH = 0
    IDX_CV = 1
    IDX_Q1 = 2
    IDX_Q2 = 3
    IDX_Q3 = 4

    def __init__(self, incoming_beam: Optional[cheetah.Beam] = None):
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
        self.incoming_beam = incoming_beam
        
        # Load ARES lattice
        ares_segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
        self.ares_ea = ares_segment.subcell("AREASOLA1", "AREABSCR1")
        
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
        self.register_constraint("raw_q3_misalign_x", misalignment_constraint)
        self.register_constraint("raw_q3_misalign_y", misalignment_constraint)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the prior mean.
        
        CRITICAL:  Xopt orders variables ALPHABETICALLY when passing to the GP!
        
        VOCS definition order:  q1, q2, cv, q3, ch
        Tensor order (alphabetical): ch, cv, q1, q2, q3
        
        Args:
            X:  Tensor of shape (..., 5) with columns in ALPHABETICAL order: 
               [ch, cv, q1, q2, q3]
               - ch: AREAMCHM1 angle (horizontal corrector) - index 0
               - cv:  AREAMCVM1 angle (vertical corrector) - index 1
               - q1: AREAMQZM1 k1 strength - index 2
               - q2: AREAMQZM2 k1 strength - index 3
               - q3: AREAMQZM3 k1 strength - index 4
        
        Returns:
            Predicted MAE (beam size metric)
        """
        # Extract variables using ALPHABETICAL order indices
        ch = X[..., self.IDX_CH]   # index 0: horizontal corrector angle
        cv = X[..., self.IDX_CV]   # index 1: vertical corrector angle
        q1 = X[..., self.IDX_Q1]   # index 2: quadrupole 1 strength
        q2 = X[..., self.IDX_Q2]   # index 3: quadrupole 2 strength
        q3 = X[..., self.IDX_Q3]   # index 4: quadrupole 3 strength
        
        # Set magnet strengths with CORRECT mapping
        self.ares_ea.AREAMQZM1.k1 = q1
        self.ares_ea.AREAMQZM2.k1 = q2
        self.ares_ea.AREAMCVM1.angle = cv
        self.ares_ea.AREAMQZM3.k1 = q3
        self.ares_ea.AREAMCHM1.angle = ch

        # Set misalignments
        misalign_q1 = torch.stack([
            self.q1_misalign_x, 
            self.q1_misalign_y
        ], dim=0)
        misalign_q2 = torch.stack([
            self.q2_misalign_x,
            self.q2_misalign_y
        ], dim=0)
        misalign_q3 = torch.stack([
            self.q3_misalign_x,
            self.q3_misalign_y
        ], dim=0)
        
        self.ares_ea.AREAMQZM1.misalignment = misalign_q1
        self.ares_ea.AREAMQZM2.misalignment = misalign_q2
        self.ares_ea.AREAMQZM3.misalignment = misalign_q3

        # Simulate beam propagation
        out_beam = self.ares_ea(self.incoming_beam)

        # Calculate MAE - must match the objective in ares_problem()! 
        ares_beam_mae = 0.25 * (
            out_beam.mu_x.abs() + 
            out_beam.sigma_x.abs() + 
            out_beam.mu_y.abs() + 
            out_beam.sigma_y.abs()
        )
       
        return ares_beam_mae

    # Properties and setters for Q1 misalignments
    @property
    def q1_misalign_x(self):
        return self._q1_misalign_x_param(self)
    
    @q1_misalign_x.setter
    def q1_misalign_x(self, value:  torch.Tensor):
        self._set_q1_misalign_x(self, value)
    
    def _q1_misalign_x_param(self, m):
        return m.raw_q1_misalign_x_constraint.transform(self.raw_q1_misalign_x)
    
    def _set_q1_misalign_x(self, m, value:  torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q1_misalign_x)
        m.initialize(
            raw_q1_misalign_x=m.raw_q1_misalign_x_constraint.inverse_transform(value)
        )
    
    @property
    def q1_misalign_y(self):
        return self._q1_misalign_y_param(self)
    
    @q1_misalign_y.setter
    def q1_misalign_y(self, value:  torch.Tensor):
        self._set_q1_misalign_y(self, value)
    
    def _q1_misalign_y_param(self, m):
        return m.raw_q1_misalign_y_constraint.transform(self.raw_q1_misalign_y)
    
    def _set_q1_misalign_y(self, m, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q1_misalign_y)
        m.initialize(
            raw_q1_misalign_y=m.raw_q1_misalign_y_constraint.inverse_transform(value)
        )
    
    # Properties and setters for Q2 misalignments
    @property
    def q2_misalign_x(self):
        return self._q2_misalign_x_param(self)
    
    @q2_misalign_x.setter
    def q2_misalign_x(self, value:  torch.Tensor):
        self._set_q2_misalign_x(self, value)
    
    def _q2_misalign_x_param(self, m):
        return m.raw_q2_misalign_x_constraint.transform(self.raw_q2_misalign_x)
    
    def _set_q2_misalign_x(self, m, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q2_misalign_x)
        m.initialize(
            raw_q2_misalign_x=m.raw_q2_misalign_x_constraint.inverse_transform(value)
        )
    
    @property
    def q2_misalign_y(self):
        return self._q2_misalign_y_param(self)
    
    @q2_misalign_y.setter
    def q2_misalign_y(self, value: torch.Tensor):
        self._set_q2_misalign_y(self, value)
    
    def _q2_misalign_y_param(self, m):
        return m.raw_q2_misalign_y_constraint.transform(self.raw_q2_misalign_y)
    
    def _set_q2_misalign_y(self, m, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q2_misalign_y)
        m.initialize(
            raw_q2_misalign_y=m.raw_q2_misalign_y_constraint.inverse_transform(value)
        )
    
    # Properties and setters for Q3 misalignments
    @property
    def q3_misalign_x(self):
        return self._q3_misalign_x_param(self)
    
    @q3_misalign_x.setter
    def q3_misalign_x(self, value: torch.Tensor):
        self._set_q3_misalign_x(self, value)
    
    def _q3_misalign_x_param(self, m):
        return m.raw_q3_misalign_x_constraint.transform(self.raw_q3_misalign_x)
    
    def _set_q3_misalign_x(self, m, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q3_misalign_x)
        m.initialize(
            raw_q3_misalign_x=m.raw_q3_misalign_x_constraint.inverse_transform(value)
        )
    
    @property
    def q3_misalign_y(self):
        return self._q3_misalign_y_param(self)
    
    @q3_misalign_y.setter
    def q3_misalign_y(self, value: torch.Tensor):
        self._set_q3_misalign_y(self, value)
    
    def _q3_misalign_y_param(self, m):
        return m.raw_q3_misalign_y_constraint.transform(self.raw_q3_misalign_y)
    
    def _set_q3_misalign_y(self, m, value: torch.Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_q3_misalign_y)
        m.initialize(
            raw_q3_misalign_y=m.raw_q3_misalign_y_constraint.inverse_transform(value)
        )
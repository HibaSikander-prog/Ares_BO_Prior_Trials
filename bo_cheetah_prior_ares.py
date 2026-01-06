from typing import Dict, Optional

import cheetah
import torch
import torch.nn as nn
from gpytorch.constraints.constraints import Interval
from gpytorch.means import Mean
from gpytorch.priors import SmoothedBoxPrior


# ARES Problem
def ares_problem(
    input_param: Dict[str, float],
    incoming_beam:  Optional[cheetah.Beam] = None,
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
            - ch: AREAMCHM1 angle (horizontal corrector)
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


# Prior Mean Functions for BO
class AresPriorMean(Mean):
    """ARES Lattice as a prior mean function for BO with trainable misalignments."""

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
        
        Args:
            X:  Tensor of shape (..., 5) with columns [q1, q2, cv, q3, ch]
        
        Returns:
            Predicted combined objective (beam size + position error)
        """
        # Set magnet strengths from input
        self.ares_ea.AREAMQZM1.k1 = X[..., 0]
        self.ares_ea.AREAMQZM2.k1 = X[..., 1]
        self.ares_ea.AREAMCVM1.angle = X[..., 2]
        self.ares_ea.AREAMQZM3.k1 = X[..., 3]
        self.ares_ea.AREAMCHM1.angle = X[..., 4]

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

        # Calculate MAE (Mean Absolute Error) - must match problem function!
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
    def q1_misalign_y(self, value: torch.Tensor):
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
    def q2_misalign_x(self, value: torch.Tensor):
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
    def q2_misalign_y(self, value:  torch.Tensor):
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
    def q3_misalign_x(self, value:  torch.Tensor):
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
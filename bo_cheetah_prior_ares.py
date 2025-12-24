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
            - ch: AREAMCHM1 angle (horizontal corrector)
        incoming_beam:  Beam parameters (uses default if None)
        misalignment_config: Dictionary with misalignments for each quadrupole
            - Format: {"AREAMQZM1": (x, y), "AREAMQZM2": (x, y), "AREAMQZM3": (x, y)}
            - Units: meters
    
    Returns:
        Dictionary with beam size metrics (mse, log_mse, mae, log_mae)
    """
    if incoming_beam is None:
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(2e-3),
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
=======
=======
>>>>>>> Stashed changes
        #incoming_beam = cheetah.ParticleBeam.from_parameters(
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            #num_particles=10000,  # Use ParticleBeam for bmadx tracking
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
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
        )
    
    # Load ARES lattice and extract the section of interest
    ares_segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
    ares_ea = ares_segment.subcell("AREASOLA1", "AREABSCR1")
    
<<<<<<< Updated upstream
=======
    # Set tracking method to bmadx for quadrupoles (required for misalignments)
  #  ares_ea.AREAMQZM1.tracking_method = "bmadx"
 #   ares_ea.AREAMQZM2.tracking_method = "bmadx"
  #  ares_ea.AREAMQZM3.tracking_method = "bmadx"
    
>>>>>>> Stashed changes
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
    
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    # Calculate metrics
    beam_size_mse = 0.5 * (out_beam.sigma_x**2 + out_beam.sigma_y**2)
    beam_size_mae = 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
    
    return {
        "mse": beam_size_mse.detach().numpy(),
        "log_mse": beam_size_mse.log().detach().numpy(),
        "mae": beam_size_mae.detach().numpy(),
        "log_mae": beam_size_mae.log().detach().numpy(),
=======
    ares_beam_mae = 0.25 * (
    out_beam.mu_x.abs() + 
    out_beam.sigma_x.abs() + 
    out_beam.mu_y.abs() + 
    out_beam.sigma_y.abs()
)
    return {
=======
    ares_beam_mae = 0.25 * (
    out_beam.mu_x.abs() + 
    out_beam.sigma_x.abs() + 
    out_beam.mu_y.abs() + 
    out_beam.sigma_y.abs()
)
    return {
>>>>>>> Stashed changes
        "mae": ares_beam_mae.detach().numpy(),
        "mu_x": out_beam.mu_x.detach().numpy(),
        "mu_y": out_beam.mu_y.detach().numpy(),
        "sigma_x": out_beam.sigma_x.detach().numpy(),
        "sigma_y": out_beam.sigma_y.detach().numpy(),
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
    }


# Prior Mean Functions for BO
class AresPriorMean(Mean):
    """ARES Lattice as a prior mean function for BO."""

    def __init__(self, incoming_beam: Optional[cheetah.Beam] = None):
        super().__init__()
        
        if incoming_beam is None:
<<<<<<< Updated upstream
<<<<<<< Updated upstream
            incoming_beam = cheetah.ParameterBeam.from_parameters(
                sigma_x=torch.tensor(1e-4),
                sigma_y=torch.tensor(2e-3),
                sigma_px=torch.tensor(1e-4),
                sigma_py=torch.tensor(1e-4),
                energy=torch.tensor(100e6),
=======
=======
>>>>>>> Stashed changes
            #incoming_beam = cheetah.ParticleBeam.from_parameters(
            incoming_beam = cheetah.ParameterBeam.from_parameters(
                #num_particles=10000,  # Use ParticleBeam for bmadx tracking
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
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
            )
        self.incoming_beam = incoming_beam
        
        # Load ARES lattice
        ares_segment = cheetah.Segment.from_lattice_json("ARESlatticeStage3v1_9.json")
        self.ares_ea = ares_segment.subcell("AREASOLA1", "AREABSCR1")
        
<<<<<<< Updated upstream
=======
        # Set tracking method to bmadx for all quadrupoles
       # self.ares_ea.AREAMQZM1.tracking_method = "bmadx"
        #self.ares_ea.AREAMQZM2.tracking_method = "bmadx"
        #self.ares_ea.AREAMQZM3.tracking_method = "bmadx"
        
>>>>>>> Stashed changes
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
        self.register_constraint("raw_q3_misalign_x", misalignment_constraint)
        self.register_constraint("raw_q3_misalign_y", misalignment_constraint)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes

>>>>>>> Stashed changes
        """
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
        #self.ares_ea.AREAMQZM1.misalignment = torch.stack([
         #   self.q1_misalign_x, 
          #  self.q1_misalign_y
        #], dim=0)
        #self.ares_ea.AREAMQZM2.misalignment = torch.stack([
         #   self.q2_misalign_x,
          #  self.q2_misalign_y
        #], dim=0)
        #self.ares_ea.AREAMQZM3.misalignment = torch.stack([
         #   self.q3_misalign_x,
          #  self.q3_misalign_y
        #], dim=0)

        # Apply learnable misalignments - KEEP YOUR ORIGINAL torch.stack() approach
        # BUT add diagnostic prints and safety checks
        misalign_q1 = torch.stack([
            self.q1_misalign_x, 
            self.q1_misalign_y
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        ])
        self.ares_ea.AREAMQZM2.misalignment = torch.stack([
            self.q2_misalign_x,
            self.q2_misalign_y
        ])
        self.ares_ea.AREAMQZM3.misalignment = torch.stack([
=======
        ], dim=0)
        misalign_q2 = torch.stack([
            self.q2_misalign_x,
            self.q2_misalign_y
        ], dim=0)
        misalign_q3 = torch.stack([
>>>>>>> Stashed changes
=======
        ], dim=0)
        misalign_q2 = torch.stack([
            self.q2_misalign_x,
            self.q2_misalign_y
        ], dim=0)
        misalign_q3 = torch.stack([
>>>>>>> Stashed changes
            self.q3_misalign_x,
            self.q3_misalign_y
        ])
        
        # DIAGNOSTIC: Print shapes (remove after debugging)
       # if not hasattr(self, '_printed_shapes'):
        #    print(f"\n[DIAGNOSTIC] Misalignment tensor shapes:")
         #   print(f"  q1_misalign_x type: {type(self.q1_misalign_x)}")
          #  print(f"  q1_misalign_x shape: {self.q1_misalign_x.shape if hasattr(self.q1_misalign_x, 'shape') else 'no shape'}")
           # print(f"  misalign_q1 shape: {misalign_q1.shape}")
            #print(f"  Expected shape: torch.Size([2])")
            #self._printed_shapes = True
        
        # SAFETY CHECK: Assert correct shape
        try:
            assert misalign_q1.shape == torch.Size([2]), \
                f"Q1 misalignment shape error: expected [2], got {misalign_q1.shape}"
            assert misalign_q2.shape == torch.Size([2]), \
                f"Q2 misalignment shape error: expected [2], got {misalign_q2.shape}"
            assert misalign_q3.shape == torch.Size([2]), \
                f"Q3 misalignment shape error: expected [2], got {misalign_q3.shape}"
        except AssertionError as e:
            print(f"\n[ERROR] {e}")
            print(f"[ERROR] This means the tensor shape fix IS needed!")
            raise
        
        self.ares_ea.AREAMQZM1.misalignment = misalign_q1
        self.ares_ea.AREAMQZM2.misalignment = misalign_q2
        self.ares_ea.AREAMQZM3.misalignment = misalign_q3

        
        # DIAGNOSTIC: Print shapes (remove after debugging)
       # if not hasattr(self, '_printed_shapes'):
        #    print(f"\n[DIAGNOSTIC] Misalignment tensor shapes:")
         #   print(f"  q1_misalign_x type: {type(self.q1_misalign_x)}")
          #  print(f"  q1_misalign_x shape: {self.q1_misalign_x.shape if hasattr(self.q1_misalign_x, 'shape') else 'no shape'}")
           # print(f"  misalign_q1 shape: {misalign_q1.shape}")
            #print(f"  Expected shape: torch.Size([2])")
            #self._printed_shapes = True
        
        # SAFETY CHECK: Assert correct shape
        try:
            assert misalign_q1.shape == torch.Size([2]), \
                f"Q1 misalignment shape error: expected [2], got {misalign_q1.shape}"
            assert misalign_q2.shape == torch.Size([2]), \
                f"Q2 misalignment shape error: expected [2], got {misalign_q2.shape}"
            assert misalign_q3.shape == torch.Size([2]), \
                f"Q3 misalignment shape error: expected [2], got {misalign_q3.shape}"
        except AssertionError as e:
            print(f"\n[ERROR] {e}")
            print(f"[ERROR] This means the tensor shape fix IS needed!")
            raise
        
        self.ares_ea.AREAMQZM1.misalignment = misalign_q1
        self.ares_ea.AREAMQZM2.misalignment = misalign_q2
        self.ares_ea.AREAMQZM3.misalignment = misalign_q3

        
        # Simulate beam propagation
        out_beam = self.ares_ea(self.incoming_beam)
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        beam_size_mae = 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
        
        return beam_size_mae
=======
=======
>>>>>>> Stashed changes

           # Calculate MAE (Mean Absolute Error) - must match problem function!
        ares_beam_mae = 0.25 * (
        out_beam.mu_x.abs() + 
        out_beam.sigma_x.abs() + 
        out_beam.mu_y.abs() + 
        out_beam.sigma_y.abs()
        )
       
        return ares_beam_mae
        
        #return beam_size_mae
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

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
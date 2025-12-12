from typing import Dict, Optional

import cheetah
import torch
import torch.nn as nn
from gpytorch.means import Mean
from gpytorch.priors import SmoothedBoxPrior


# Test Problem
def ares_problem(
    input_param: Dict[str, float],
    incoming_beam:  Optional[cheetah.Beam] = None,
    lattice_config: Optional[Dict[str, float]] = {},
) -> Dict[str, float]: 
    """
    ARES lattice optimization problem.
    
    Args:
        input_param: Dictionary with keys 'q1', 'q2', 'q3', 'cv', 'ch'
        incoming_beam: Optional beam configuration
        lattice_config: Optional dictionary with misalignment values for elements
    
    Returns:
        Dictionary with beam size metrics (mse, mae, log_mse, log_mae)
    """
    if incoming_beam is None:
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            sigma_x=torch.tensor(1e-4),
            sigma_y=torch.tensor(2e-3),
            sigma_px=torch.tensor(1e-4),
            sigma_py=torch.tensor(1e-4),
            energy=torch.tensor(100e6),
        )
    
    # Drift lengths (from lattice specification)
    d1_length = torch.tensor(0.17504000663757324)
    d2_length = torch.tensor(0.42800000309944153)
    d3_length = torch.tensor(0.20399999618530273)
    d4_length = torch.tensor(0.20399999618530273)
    d5_length = torch.tensor(0.17900000512599945)
    d6_length = torch.tensor(0.44999998807907104)
    
    # Quadrupole length (all same)
    quad_length = torch.tensor(0.12200000137090683)
    
    # Corrector lengths
    corrector_length = torch.tensor(0.019999999552965164)
    
    # Get misalignments from config (default to 0.0)
    q1_misalignment = [
        lattice_config.get("q1_misalignment_x", 0.0),
        lattice_config.get("q1_misalignment_y", 0.0)
    ]
    q2_misalignment = [
        lattice_config.get("q2_misalignment_x", 0.0),
        lattice_config.get("q2_misalignment_y", 0.0)
    ]
    q3_misalignment = [
        lattice_config.get("q3_misalignment_x", 0.0),
        lattice_config.get("q3_misalignment_y", 0.0)
    ]
    
    # Build ARES segment:  d1 -> q1 -> d2 -> q2 -> d3 -> cv -> d4 -> q3 -> d5 -> ch -> d6
    ares_segment = cheetah.Segment(
        [
            cheetah.Drift(length=d1_length, name="D1"),
            cheetah.Quadrupole(
                length=quad_length,
                k1=torch.tensor(input_param["q1"]),
                misalignment=torch.tensor(q1_misalignment),
                name="Q1"
            ),
            cheetah.Drift(length=d2_length, name="D2"),
            cheetah.Quadrupole(
                length=quad_length,
                k1=torch.tensor(input_param["q2"]),
                misalignment=torch.tensor(q2_misalignment),
                name="Q2"
            ),
            cheetah.Drift(length=d3_length, name="D3"),
            cheetah.VerticalCorrector(
                length=corrector_length,
                angle=torch.tensor(input_param["cv"]),
                name="CV"
            ),
            cheetah.Drift(length=d4_length, name="D4"),
            cheetah.Quadrupole(
                length=quad_length,
                k1=torch.tensor(input_param["q3"]),
                misalignment=torch.tensor(q3_misalignment),
                name="Q3"
            ),
            cheetah.Drift(length=d5_length, name="D5"),
            cheetah.HorizontalCorrector(
                length=corrector_length,
                angle=torch.tensor(input_param["ch"]),
                name="CH"
            ),
            cheetah.Drift(length=d6_length, name="D6"),
        ]
    )
    
    out_beam = ares_segment(incoming_beam)
    
    beam_size_mse = 0.5 * (out_beam.sigma_x**2 + out_beam.sigma_y**2)
    beam_size_mae = 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
    
    return {
        "mse": beam_size_mse.detach().numpy(),
        "log_mse": beam_size_mse.log().detach().numpy(),
        "mae": beam_size_mae.detach().numpy(),
        "log_mae":  beam_size_mae.log().detach().numpy(),
    }


# Prior Mean Functions for BO
class AresPriorMean(Mean):
    """ARES Lattice as a prior mean function for BO."""
    
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
        
        # Drift lengths (fixed from lattice specification)
        d1_length = torch.tensor([0.17504000663757324])
        d2_length = torch.tensor([0.42800000309944153])
        d3_length = torch.tensor([0.20399999618530273])
        d4_length = torch.tensor([0.20399999618530273])
        d5_length = torch.tensor([0.17900000512599945])
        d6_length = torch.tensor([0.44999998807907104])
        
        # Quadrupole length (all same)
        quad_length = torch.tensor([0.12200000137090683])
        
        # Corrector lengths
        corrector_length = torch.tensor([0.019999999552965164])
        
        # Initialize elements
        self.D1 = cheetah.Drift(length=d1_length, name="D1")
        self.Q1 = cheetah.Quadrupole(
            length=quad_length,
            k1=torch.tensor([0.0]),
            misalignment=torch.tensor([0.0, 0.0]),
            name="Q1"
        )
        self.D2 = cheetah.Drift(length=d2_length, name="D2")
        self.Q2 = cheetah.Quadrupole(
            length=quad_length,
            k1=torch.tensor([0.0]),
            misalignment=torch.tensor([0.0, 0.0]),
            name="Q2"
        )
        self.D3 = cheetah.Drift(length=d3_length, name="D3")
        self.CV = cheetah.VerticalCorrector(
            length=corrector_length,
            angle=torch.tensor([0.0]),
            name="CV"
        )
        self.D4 = cheetah.Drift(length=d4_length, name="D4")
        self.Q3 = cheetah.Quadrupole(
            length=quad_length,
            k1=torch.tensor([0.0]),
            misalignment=torch.tensor([0.0, 0.0]),
            name="Q3"
        )
        self.D5 = cheetah.Drift(length=d5_length, name="D5")
        self.CH = cheetah.HorizontalCorrector(
            length=corrector_length,
            angle=torch.tensor([0.0]),
            name="CH"
        )
        self.D6 = cheetah.Drift(length=d6_length, name="D6")
        
        self.segment = cheetah.Segment(
            elements=[
                self.D1, self.Q1, self.D2, self.Q2, self.D3,
                self.CV, self.D4, self.Q3, self.D5, self.CH, self.D6
            ]
        )
        
        # Introduce trainable misalignment parameters
        # Each quadrupole has x and y misalignment
        # Note: No Positive constraint since misalignments can be negative
        self.register_parameter("raw_q1_misalignment_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q1_misalignment_y", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q2_misalignment_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q2_misalignment_y", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q3_misalignment_x", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("raw_q3_misalignment_y", nn.Parameter(torch.tensor(0.0)))
        
        # Register priors for misalignments (range:  -0.5mm to 0.5mm = -0.0005m to 0.0005m)
        self.register_prior(
            "q1_misalignment_x_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            lambda m: m.raw_q1_misalignment_x,
            lambda m, v: m._set_param("raw_q1_misalignment_x", v)
        )
        self.register_prior(
            "q1_misalignment_y_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            lambda m:  m.raw_q1_misalignment_y,
            lambda m, v: m._set_param("raw_q1_misalignment_y", v)
        )
        self.register_prior(
            "q2_misalignment_x_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            lambda m: m.raw_q2_misalignment_x,
            lambda m, v: m._set_param("raw_q2_misalignment_x", v)
        )
        self.register_prior(
            "q2_misalignment_y_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            lambda m: m.raw_q2_misalignment_y,
            lambda m, v: m._set_param("raw_q2_misalignment_y", v)
        )
        self.register_prior(
            "q3_misalignment_x_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            lambda m: m.raw_q3_misalignment_x,
            lambda m, v: m._set_param("raw_q3_misalignment_x", v)
        )
        self.register_prior(
            "q3_misalignment_y_prior",
            SmoothedBoxPrior(-0.0005, 0.0005),
            lambda m:  m.raw_q3_misalignment_y,
            lambda m, v: m._set_param("raw_q3_misalignment_y", v)
        )
    
    def _set_param(self, param_name, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(getattr(self, param_name))
        self.initialize(**{param_name: value})
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ARES lattice.
        
        Args:
            X:  Tensor of shape [... , 5] with [q1, q2, q3, cv, ch]
        
        Returns:
            Beam size MAE
        """
        # Set magnet strengths from input
        self.Q1.k1 = X[..., 0]
        self.Q2.k1 = X[..., 1]
        self.Q3.k1 = X[..., 2]
        self.CV.angle = X[..., 3]
        self.CH.angle = X[..., 4]
        
        # Set trainable misalignments
        self.Q1.misalignment = torch.stack([
            self.raw_q1_misalignment_x,
            self.raw_q1_misalignment_y
        ])
        self.Q2.misalignment = torch.stack([
            self.raw_q2_misalignment_x,
            self.raw_q2_misalignment_y
        ])
        self.Q3.misalignment = torch.stack([
            self.raw_q3_misalignment_x,
            self.raw_q3_misalignment_y
        ])
        
        out_beam = self.segment(self.incoming_beam)
        beam_size_mae = 0.5 * (out_beam.sigma_x.abs() + out_beam.sigma_y.abs())
        return beam_size_mae
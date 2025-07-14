import torch
from satsim.architecture import Module,VoidStateDict


class Eclipse(Module[VoidStateDict]):

    def forward(
        *args,
        spacecraft_position_in_inertial: torch.Tensor,
        sun_position_in_inertial: torch.Tensor,
        planet_position_in_inertial: torch.Tensor,
        **kwargs,
    ) -> tuple[VoidStateDict, tuple[torch.Tensor]]:
        
        
        

    


__all__ = [
    "SimplePowerSink",
]

import torch
from typing import TypedDict
from satsim.architecture import Module


class SimplePowerSinkStateDict(TypedDict):
    pass


class SimplePower(Module[SimplePowerSinkStateDict]):
    """
    A simple power sink module that consumes power.
    """

    def __init__(self, *args, power_out: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.register_buffer(
            "power_out",
            power_out,
            persistent=False,
        )

    def forward(
        self,
        state_dict: SimplePowerSinkStateDict,
        *args,
        **kwargs,
    ) -> tuple[SimplePowerSinkStateDict, tuple[torch.Tensor]]:
        """
        Compute and output current power usage
        
        Args:
            state_dict: Dictionary containing current module state
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        
        Returns:
            tuple: Updated state dictionary and power output tuple
        """

        return dict(), (self.get_buffer("power_out"), )

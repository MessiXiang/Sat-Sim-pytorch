__all__ = [
    'SimplePowerSink',
    'SimplePowerSinkStateDict',
]

from typing import TypedDict

import torch

from satsim.architecture import Module

from ..base.battery_base import BatteryStateDict, PowerNodeMixin


class SimplePowerSinkStateDict(TypedDict):
    pass


class SimplePowerSink(Module[SimplePowerSinkStateDict], PowerNodeMixin):

    def __init__(self, *args, power_efficiency: torch.Tensor,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.register_buffer(
            '_power_efficiency',
            power_efficiency,
            persistent=False,
        )

    @property
    def power_efficiency(self) -> torch.Tensor:
        return self.get_buffer('_power_efficiency')

    def forward(
        self,
        state_dict: SimplePowerSinkStateDict,
        *args,
        battery_state_dict: BatteryStateDict,
        **kwargs,
    ) -> tuple[
            SimplePowerSinkStateDict,
            tuple[torch.Tensor, BatteryStateDict],
    ]:

        power_status, battery = self.update_power_status(
            self.power_efficiency, battery_state_dict)

        return state_dict, (power_status, battery)

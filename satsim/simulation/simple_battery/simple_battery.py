__all__ = ['SimplePowerMonitorStateDict', 'SimplePowerMonitor']
from typing import TypedDict

import torch

from satsim.architecture.module import Module


class SimpleBatteryStateDict(TypedDict):
    stored_charge: torch.Tensor
    storage_capacity: torch.Tensor


class SimpleBattery(Module[SimpleBatteryStateDict]):

    def __init__(
        self,
        *args,
        storage_capacity: torch.Tensor,
        stored_charge_init: torch.Tensor | None = None,
        **kwargs,
    ):

        self.stored_charge_init = torch.zeros_like(
            storage_capacity
        ) if stored_charge_init is None else stored_charge_init
        self.register_buffer(
            '_storage_capacity',
            storage_capacity,
            persistent=False,
        )

    def reset(self) -> SimpleBatteryStateDict:
        return dict(stored_charge=self.stored_charge_init.clone())

    def forward(
        self,
        state_dict: SimpleBatteryStateDict,
        *args,
        net_power_efficiency: torch.Tensor,
        **kwargs,
    ) -> tuple[SimpleBatteryStateDict, tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
    ]]:
        """Calculate total power change. Deprecating in future version
        Returns:
            stored_charge: stored power in watt-hours
            max_capacity: maximum battery storage capacity
            current_net_power: current power efficiency
        """
        total_net_power_efficiency = net_power_efficiency.sum(-1, keepdim=True)
        stored_charge = state_dict['stored_charge']
        storage_capactiy = self.get_buffer('_storage_capacity')

        stored_charge = stored_charge + total_net_power_efficiency * self._timer.dt
        stored_charge = torch.clamp(
            stored_charge,
            min=0.,
            max=storage_capactiy,
        )

        state_dict['stored_charge'] = stored_charge

        return state_dict, (
            stored_charge,
            storage_capactiy,
            total_net_power_efficiency,
        )

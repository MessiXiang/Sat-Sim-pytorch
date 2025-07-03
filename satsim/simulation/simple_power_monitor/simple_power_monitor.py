__all__ = ['SimplePowerMonitorStateDict', 'SimplePowerMonitor']
from typing import TypedDict

import torch

from satsim.architecture.module import Module


class SimplePowerMonitorStateDict(TypedDict):
    stored_charge: torch.Tensor


class SimplePowerMonitor(Module[SimplePowerMonitorStateDict]):

    def reset(self) -> SimplePowerMonitorStateDict:
        return dict(stored_charge=torch.tensor(0.))

    def forward(
        self,
        state_dict: SimplePowerMonitorStateDict,
        *args,
        net_power: torch.Tensor,
        **kwargs,
    ) -> tuple[SimplePowerMonitorStateDict, tuple[
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
        total_net_power = net_power.sum(keepdim=True)
        stored_charge = state_dict['stored_charge']

        stored_charge = stored_charge + total_net_power * self._timer.dt
        state_dict['stored_charge'] = stored_charge

        return state_dict, (
            stored_charge,
            torch.tensor(-1.),
            total_net_power,
        )

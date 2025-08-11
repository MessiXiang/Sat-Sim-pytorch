__all__ = ['SimpleBattery', 'NoBattery']
import torch

from ..base import BatteryBase, BatteryStateDict


# Note
class SimpleBattery(BatteryBase):

    def forward(
        self,
        state_dict: BatteryStateDict,
        *args,
        **kwargs,
    ) -> tuple[BatteryStateDict, tuple]:
        """Calculate total power change. Deprecating in future version
        Returns:
            stored_charge: stored power in watt-hours
            max_capacity: maximum battery storage capacity
            current_net_power: current power efficiency
        """
        stored_charge_percentage = state_dict['stored_charge_percentage']
        current_net_power_percentage = state_dict[
            'current_net_power_percentage']

        stored_charge_percentage = stored_charge_percentage + current_net_power_percentage
        stored_charge_percentage = torch.clamp(
            stored_charge_percentage,
            min=0.,
            max=1.,
        )

        state_dict['stored_charge_percentage'] = stored_charge_percentage
        state_dict['current_net_power_percentage'] = torch.zeros_like(
            current_net_power_percentage)

        return state_dict


class NoBattery(SimpleBattery):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            storage_capacity=torch.inf,
            stored_charge_percentage_init=torch.tensor(0.5),
            **kwargs,
        )

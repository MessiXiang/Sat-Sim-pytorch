__all__ = ['BatteryStateDict', 'BatteryBase', 'PowerNodeMixin']
from typing import TypedDict

import torch

from satsim.architecture import Module, Timer


class BatteryStateDict(TypedDict):
    stored_charge_percentage: torch.Tensor  # stored power percentage
    storage_capacity: torch.Tensor
    current_net_power_percentage: torch.Tensor  # current power efficiency


class BatteryBase(Module[BatteryStateDict]):

    def __init__(
        self,
        *args,
        storage_capacity: torch.Tensor,
        stored_charge_percentage_init: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if stored_charge_percentage_init is not None and torch.any(
            (stored_charge_percentage_init < 0.)
                | (stored_charge_percentage_init > 1.)):
            raise ValueError("Initialized stored chage percentage not valid.")

        self.stored_charge_init = torch.zeros_like(
            storage_capacity
        ) if stored_charge_percentage_init is None else stored_charge_percentage_init
        self.register_buffer(
            '_storage_capacity',
            storage_capacity,
            persistent=False,
        )

    @property
    def storage_capacity(self) -> torch.Tensor:
        return self.get_buffer('_storage_capacity')

    def reset(self) -> BatteryStateDict:
        return BatteryStateDict(
            stored_charge_percentage=self.stored_charge_init.clone(),
            storage_capacity=self.storage_capacity,
            current_net_power_percentage=torch.zeros_like(
                self.stored_charge_init),
        )


class PowerNodeMixin:
    _timer: Timer

    def update_power_status(
        self,
        power: torch.Tensor,
        battery_state_dict: BatteryStateDict,
    ) -> tuple[torch.Tensor, BatteryStateDict]:
        storage_capacity = battery_state_dict['storage_capacity']  #[n_s]
        current_net_power_percentage = battery_state_dict[
            'current_net_power_percentage']  # [n_s]
        stored_charge_percentage = battery_state_dict[
            'stored_charge_percentage']

        power_usage_percentage = (power * self._timer.dt) / storage_capacity

        new_current_net_power_percentage = current_net_power_percentage + power_usage_percentage
        power_status = new_current_net_power_percentage < stored_charge_percentage

        # modify state_dict
        battery_state_dict['current_net_power_percentage'] = torch.where(
            power_status,
            new_current_net_power_percentage,
            current_net_power_percentage,
        )

        return power_status, battery_state_dict

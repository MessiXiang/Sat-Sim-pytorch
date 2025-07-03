__all__ = ['PowerStorageBaseOutput', 'PowerStorageBase']
from abc import abstractmethod
from typing import Generic, TypeVar

import torch

from satsim.architecture import Module


class PowerStorageBaseOutput:
    stored_charge: torch.Tensor  # stored power in watt-hours
    storage_capacity: torch.Tensor  # maximum battery storage capacity
    current_net_power: torch.Tensor  # current power efficiency


T = TypeVar('U')


class PowerStorageBase(Module[T], Generic[T]):

    @abstractmethod
    def forward(
        state_dict: T,
        *args,
        net_power: torch.Tensor,
        **kwargs,
    ) -> tuple[T, tuple[PowerStorageBaseOutput]]:
        '''Any child of PowerStorageBase should output stored_charge, storage_capacity and current_net_power 
        '''
        pass

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class SimpleBattery(SimulationModule):
    """
    storage_capacity: Watt * second
    """

    def __init__(self,
                 name: str = "SimpleBattery",
                 storage_capacity: float = 100.0,
                 fault_ratio: float = 1.0):
        super().__init__(name)

        self.storage_capacity = nn.Parameter(torch.tensor(storage_capacity),
                                             requires_grad=False)
        self.stored_charge = nn.Parameter(torch.tensor(0.0),
                                          requires_grad=False)

        self.fault_capacity_ratio = torch.tensor(fault_ratio)
        self.current_power = torch.tensor(0.0)

        if storage_capacity <= 0:
            raise ValueError("Storage capacity must be positive")

    def simulation_step(self, dt: Tensor, *args, **kwargs) -> Tensor:
        """
        Simulate one time step of the battery.

        Args:
            dt: Time step duration
            power_input: positive for charging, negative for discharging
            fault_ratio: decrease of capacity

        Returns:
            Tensor containing current stored charge
        """
        self.current_power = kwargs.get("power_input", torch.tensor(0.0))
        self.fault_capacity_ratio = kwargs.get("fault_ratio",
                                               torch.tensor(1.0))

        if self.fault_capacity_ratio > 1.0 or self.fault_capacity_ratio < 0.0:
            raise (ValueError("wrong fault_ratio(0.0-1.0)"))

        self.stored_charge.data = self.stored_charge.data + self.current_power * dt

        effective_capacity = self.storage_capacity * self.fault_capacity_ratio
        self.stored_charge.data = torch.clamp(self.stored_charge.data, 0.0,
                                              effective_capacity)

        return self.stored_charge.clone()

    def reset_simulation_state(self):
        self.stored_charge.data.fill_(0.0)
        self.fault_capacity_ratio.data.fill_(1.0)
        self.current_power.fill_(0.0)

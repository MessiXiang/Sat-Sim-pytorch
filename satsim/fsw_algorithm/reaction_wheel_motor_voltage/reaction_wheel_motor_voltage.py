__all__ = [
    'ReactionWheelMotorVoltage',
]

import torch
from typing import TypedDict
from satsim.architecture import Module, Timer

class ReactionWheelMotorVoltageStateDict(TypedDict):
    reaction_wheel_speed_old: torch.Tensor
    prior_time: torch.Tensor
    reset_flag: bool


class ReactionWheelMotorVoltage(Module[ReactionWheelMotorVoltageStateDict]):
    """
    Module that converts reaction wheel torque commands to motor voltages
    Function: Receives torque commands and wheel speed feedback, calculates and outputs motor voltage commands
    """
    def __init__(
        self,
        *args,
        timer: Timer,
        v_max: torch.Tensor = None,
        v_min: torch.Tensor = None,
        k: torch.Tensor = None,
        # max_eff_cnt: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, timer=timer, **kwargs)

        self.register_buffer(
            "v_max",
            v_max,
            persistent=False,
        )
        self.register_buffer(
            "v_min",
            v_min,
            persistent=False,
        )
        self.register_buffer(
            "k",
            k,
            persistent=False,
        )
        # self.register_buffer("max_eff_cnt", torch.tensor(max_eff_cnt, dtype=torch.int32), persistent=False)

        self.reaction_config_params_init: dict = None
        self.num_reaction_wheels_init: int = 0

    def reset(self) -> ReactionWheelMotorVoltageStateDict:
        reaction_wheel_speed_old = torch.zeros(self.num_reaction_wheels_init, dtype=torch.float32)
        prior_time = torch.tensor(self._timer.time(), dtype=torch.float32)
        reset_flag = False

        return {
            "reaction_wheel_speed_old": reaction_wheel_speed_old,
            "prior_time": prior_time,
            "reset_flag": reset_flag,
        }
    
    def forward(
        self,
        state_dict: ReactionWheelMotorVoltageStateDict,
        *args,
        reaction_wheel_motor_torque: torch.Tensor = None,
        reaction_wheel_speed: torch.Tensor = None,
        reaction_wheel_available: torch.Tensor = None,
        reaction_wheel_config_params: dict = None,
        **kwargs,
    ) -> tuple[ReactionWheelMotorVoltageStateDict, tuple[torch.Tensor]]:
        """
        Calculates reaction wheel motor voltage commands
        
        Args:
            state_dict: Current state dictionary of the module
            reaction_wheel_motor_torque: Torque command [num_rw] or [batch, num_rw]
            rw_speed: Wheel speed feedback [num_rw] or [batch, num_rw] (optional)
            rw_availability: Wheel availability [num_rw] (0=unavailable, 1=available) (optional)
            rw_config_params: Reaction wheel configuration parameters (including JsList, uMax, etc.)
            
        Returns:
            tuple: (Updated state dictionary, (Voltage command tensor,))
        """
        v_max = self.get_buffer("v_max")
        v_min = self.get_buffer("v_min")
        k = self.get_buffer("k")

        if self.reaction_config_params_init is None and reaction_wheel_config_params is not None:
            self.reaction_config_params_init = reaction_wheel_config_params
            self.num_reaction_wheels_init = reaction_wheel_config_params["num_reaction_wheels"]
            js_list = reaction_wheel_config_params["JsList"]
            u_max = reaction_wheel_config_params["uMax"]

        reaction_wheel_speed_old = state_dict["reaction_wheel_speed_old"]
        prior_time = state_dict["prior_time"]
        reset_flag = state_dict["reset_flag"]

        num_reaction_wheels = self.num_reaction_wheels_init
        if reaction_wheel_available is not None:
            reaction_wheel_available = torch.ones(num_reaction_wheels, dtype=torch.float32)
        if reaction_wheel_speed is None:
            reaction_wheel_speed = torch.zeros_like(reaction_wheel_motor_torque)
        
        voltage = torch.zeros_like(reaction_wheel_motor_torque, dtype=torch.float32)
        mask = (reaction_wheel_available == 1.0).float()

        # if the torque closed-loop is on, evaluate the feedback term
        if reaction_wheel_speed is not None:
            dt = self._timer.dt()
            # TODO: move prior time 
            if prior_time.item() != 0 and reset_flag == False:
                omega_dot = (reaction_wheel_speed - reaction_wheel_speed_old) / dt.unsqueeze(0)
                reaction_wheel_motor_torque = reaction_wheel_motor_torque - k * mask * (js_list.unsqueeze(0) * omega_dot - reaction_wheel_motor_torque)
            
            state_dict["reaction_wheel_speed_old"] = reaction_wheel_speed
            state_dict["prior_time"] = torch.tensor(self._timer.time(), dtype=torch.float32)
            state_dict["reset_flag"] = False
        
        voltage_base = (v_max - v_min) * reaction_wheel_motor_torque / u_max.unsqueeze(0)
        voltage_sign = torch.sign(voltage_base)
        voltage = mask * (voltage_base + v_min * voltage_sign)
        voltage = torch.clamp(voltage, min=v_min, max=v_max)

        return state_dict, (voltage, )

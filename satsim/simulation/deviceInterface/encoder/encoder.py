__all__ = [
    'Encoder',
    'EncoderSignal',
    'EncoderState',
]
import os
from typing import Any, TypedDict, cast
from enum import IntEnum
import torch

from satsim.architecture import (
    Module,
    Timer,
)


class EncoderSignal(IntEnum):
    """Enum for encoder state"""
    NOMINAL = 0
    OFF = 1
    STUCK = 2


class EncoderState(TypedDict):
    """Encoder's state dict"""
    reaction_wheels_signal_state: torch.Tensor
    remaining_clicks: torch.Tensor
    last_output: torch.Tensor


class Encoder(Module[EncoderState]):
    """encoder module"""

    def __init__(
        self,
        *args,
        timer: Timer,
        num_reaction_wheels: int,
        clicks_per_rotation: int,
        **kwargs,
    ) -> None:

        super().__init__(*args, timer=timer, **kwargs)

        self._num_rw = num_reaction_wheels
        self._clicks_per_rotation = clicks_per_rotation
        mode = os.environ.get('MODE', "C")

        self.mode = mode
        if mode == "C":
            self.kernel = torch.ops.encoder.encoder_kernel
        elif mode == "PY":
            self.kernel = torch.ops.encoder.encoder_py

    @property
    def _clicks_per_radian(self) -> float:
        return self._clicks_per_rotation / (2 * torch.pi)

    def reset(self, device: str = 'cpu') -> EncoderState:
        state_dict: dict[str, Any] = {}
        state_dict['reaction_wheels_signal_state'] = torch.zeros(self._num_rw)
        state_dict['remaining_clicks'] = torch.zeros(self._num_rw)
        state_dict['last_output'] = torch.zeros(self._num_rw)
        state_dict = self.move_state_to(state_dict, device)

        state_dict.update(super().reset(device))
        return cast(EncoderState, state_dict)

    def forward(
        self,
        state_dict: EncoderState,
        *args,
        wheel_speeds: torch.Tensor,
        **kwargs,
    ) -> tuple[EncoderState, tuple[torch.Tensor]]:
        if self._timer.step_count == 0:
            state_dict['last_output'] = wheel_speeds
            return state_dict, (wheel_speeds, )

        assert torch.isin(
            state_dict['reaction_wheels_signal_state'],
            torch.tensor(
                [
                    EncoderSignal.NOMINAL, EncoderSignal.OFF,
                    EncoderSignal.STUCK
                ],
                device=state_dict['reaction_wheels_signal_state'].device,
            ),
        ).all(), "encoder: un-modeled encoder signal mode selected."

        new_output, new_remaining_clicks = self.kernel(
            wheel_speeds, state_dict['remaining_clicks'],
            state_dict['reaction_wheels_signal_state'],
            state_dict['last_output'], self._clicks_per_radian, self._timer.dt)

        state_dict['remaining_clicks'] = new_remaining_clicks
        state_dict['last_output'] = new_output

        return state_dict, (new_output, )

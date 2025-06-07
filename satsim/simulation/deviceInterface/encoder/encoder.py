__all__ = [
    'Encoder',
    'EncoderSignal',
    'EncoderState',
]
from typing import TypedDict, cast
from enum import IntEnum
import pathlib
import torch
import cupy

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

    @property
    def _clicks_per_radian(self) -> float:
        return self._clicks_per_rotation / (2 * torch.pi)

    def reset(self) -> EncoderState:
        state_dict = {}
        state_dict.update(super().reset())
        state_dict['reaction_wheels_signal_state'] = torch.zeros(self._num_rw)
        state_dict['remaining_clicks'] = torch.zeros(self._num_rw)
        state_dict['last_output'] = torch.zeros(self._num_rw)
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
            torch.tensor([
                EncoderSignal.NOMINAL, EncoderSignal.OFF, EncoderSignal.STUCK
            ])).all(), "encoder: un-modeled encoder signal mode selected."

        new_output, new_remaining_clicks = torch.ops.encoder.encoder_kernel(
            wheel_speeds.cuda(), state_dict['remaining_clicks'].cuda(),
            state_dict['reaction_wheels_signal_state'].cuda(),
            state_dict['last_output'].cuda(), self._clicks_per_radian,
            self._timer.dt)

        state_dict['remaining_clicks'] = new_remaining_clicks
        state_dict['last_output'] = new_output

        return state_dict, (new_output, )

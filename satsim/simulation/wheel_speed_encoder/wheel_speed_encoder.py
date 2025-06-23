__all__ = [
    'WheelSpeedEncoder',
    'WheelSpeedEncoderSignal',
    'WheelSpeedEncoderStateDict',
]

from enum import IntEnum, auto
from typing import TypedDict

import torch

from satsim.architecture import Module
from satsim.utils import run_operator


class WheelSpeedEncoderSignal(IntEnum):
    NOMINAL = auto()
    STOPPED = auto()
    LOCKED = auto()

    @classmethod
    def validate(cls, signal: int) -> bool:
        return signal in cls._value2member_map_


class WheelSpeedEncoderStateDict(TypedDict):
    remaining_clicks: torch.Tensor
    signals: torch.Tensor
    speeds: torch.Tensor


class WheelSpeedEncoder(Module[WheelSpeedEncoderStateDict]):

    def __init__(self, *args, n: int, num_clicks: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._n = n
        self._num_clicks = num_clicks

    @property
    def _clicks_per_radian(self) -> float:
        return self._num_clicks / (2 * torch.pi)

    def reset(self) -> WheelSpeedEncoderStateDict:
        state_dict = super().reset()
        state_dict.update(
            signals=torch.ones(self._n, dtype=torch.int),
            remaining_clicks=torch.zeros(self._n),
            speeds=torch.zeros(self._n),
        )
        return state_dict

    def forward(
        self,
        state_dict: WheelSpeedEncoderStateDict,
        *args,
        target_speeds: torch.Tensor,
        signals: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[WheelSpeedEncoderStateDict, tuple[torch.Tensor]]:
        if signals is not None:
            assert all(map(WheelSpeedEncoderSignal.validate, signals.flatten().tolist())), \
                "encoder: un-modeled encoder signal mode selected."
            state_dict['signals'] = signals.int()

        if self._timer.step_count == 0:
            state_dict['speeds'] = target_speeds
            return state_dict, (target_speeds, )

        speeds, remaining_clicks = run_operator(
            target_speeds,
            state_dict['remaining_clicks'],
            state_dict['signals'],
            state_dict['speeds'],
            self._clicks_per_radian,
            self._timer.dt,
        )

        state_dict.update(
            remaining_clicks=remaining_clicks,
            speeds=speeds,
        )
        return state_dict, (speeds, )

__all__ = ['Encoder']
from typing import Any, TypedDict, cast
import torch

from ...architecture import (
    SIGNAL_NOMINAL,
    SIGNAL_OFF,
    SIGNAL_STUCK,
    Module,
    Timer,
)


class EncoderState(TypedDict):
    rw_signal_state: torch.Tensor
    remaining_clicks: torch.Tensor
    converted: torch.Tensor


class Encoder(Module):

    def __init__(self,
                 *args,
                 timer: Timer,
                 numRW: int = -1,
                 clicksPerRotation: int = -1,
                 **kwargs) -> None:

        super().__init__(*args, timer=timer, **kwargs)

        # Check if number of reaction wheels and clicks per rotation is valid
        assert clicksPerRotation > 0, "encoder: number of clicks must be a positive integer."
        assert numRW > 0, "encoder: number of reaction wheels must be a positive integer. It may not have been set."

        self._num_rw = numRW
        self._clicks_per_rotation = clicksPerRotation

        # #internal states
        # self.register_buffer("_rw_signal_state",
        #                      torch.zeros(self._num_rw, dtype=torch.int32))
        # self.register_buffer("_remaining_clicks",
        #                      torch.zeros(self._num_rw, dtype=torch.float32))
        # self.register_buffer("_converted",
        #                      torch.zeros(self._num_rw, dtype=torch.float32))

    @property
    def rw_signal_state(self) -> torch.Tensor:
        return self.get_buffer("_rw_signal_state")

    @property
    def remaining_clicks(self) -> torch.Tensor:
        return self.get_buffer("_remaining_clicks")

    @property
    def converted(self) -> torch.Tensor:
        return self.get_buffer("_converted")

    @property
    def _clicks_per_radian(self) -> float:
        return self._clicks_per_rotation / (2 * torch.pi)

    def reset(self) -> EncoderState:
        """
        Resets the encoder with the given simulation time in nanoseconds.
        
        Args:
            current_sim_nanos (int): Current simulation time in nanoseconds
        
        
        """
        state_dict: dict[str, Any] = super().reset()
        state_dict['rw_signal_state'] = torch.zeros(self._num_rw)
        state_dict['remaining_clicks'] = torch.zeros(self._num_rw)
        state_dict['converted'] = torch.zeros(self._num_rw)
        return cast(EncoderState, state_dict)

    def forward(self, *args, wheel_speeds: torch.Tensor,
                **kwargs) -> torch.Tensor:
        # At the beginning of the simulation, the encoder outputs the true RW speeds
        if self._timer.step_count == 0:
            return wheel_speeds

        new_output = torch.zeros_like(wheel_speeds)

        # TODO: CUPY

        ## check if all state is modeled
        assert torch.isin(
            self.rw_signal_state,
            torch.tensor([
                SIGNAL_NOMINAL, SIGNAL_OFF, SIGNAL_STUCK
            ])).all(), "encoder: un-modeled encoder signal mode selected."

        ## SIGNAL_NOMINAL_SITUATION

        signal_nominal_mask = self.rw_signal_state == SIGNAL_NOMINAL
        if torch.any(signal_nominal_mask):
            angle = wheel_speeds * self._timer.dt
            number_clicks = torch.trunc(angle * self._clicks_per_radian +
                                        self.remaining_clicks)

            self.remaining_clicks[
                signal_nominal_mask] = angle * self._clicks_per_radian + self.remaining_clicks[
                    signal_nominal_mask] - number_clicks

            new_output = torch.where(
                signal_nominal_mask,
                number_clicks / (self._clicks_per_radian * self._timer.dt),
                new_output)

        ## SIGNAL_OFF_SITUATION
        signal_off_mask = self.rw_signal_state == SIGNAL_OFF
        if torch.any(signal_off_mask):
            self.remaining_clicks[signal_off_mask] = 0.0
            new_output = torch.where(signal_off_mask, 0., new_output)

        ## SIGNAL_STUCK_SITUATION
        signal_stuck_mask = self.rw_signal_state == SIGNAL_STUCK
        if torch.any(signal_stuck_mask):
            new_output = torch.where(signal_stuck_mask, self.converted.clone(),
                                     new_output)

        self.converted.copy_(new_output)  # TODO: gradient
        return new_output

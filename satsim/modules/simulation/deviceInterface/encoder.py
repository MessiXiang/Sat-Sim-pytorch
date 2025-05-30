import torch

from ....architecture import (
    SIGNAL_NOMINAL, 
    SIGNAL_OFF, 
    SIGNAL_STUCK,
    Module,
    Timer,
)

class Encoder(Module):
    _rw_signal_state: torch.Tensor
    _remaining_clicks: torch.Tensor
    _converted: torch.Tensor
    
    def __init__(
            self,
            *args,
            timer: Timer,
            numRW: int = -1,
            clicksPerRotation: int = -1,
            **kwargs ) -> None:

        
        super().__init__(
            *args,
            timer = timer,
            **kwargs
        )

        # Check if number of reaction wheels and clicks per rotation is valid
        assert clicksPerRotation > 0, "encoder: number of clicks must be a positive integer."        
        assert numRW > 0, "encoder: number of reaction wheels must be a positive integer. It may not have been set."
        
        self._num_rw = numRW
        self._clicks_per_rotation = clicksPerRotation

        #internal states
        self.register_buffer("_rw_signal_state", torch.zeros(self._num_rw, dtype=torch.int32))
        self.register_buffer("_remaining_clicks", torch.zeros(self._num_rw, dtype=torch.float32))
        self.register_buffer("_converted", torch.zeros(self._num_rw, dtype=torch.float32))
    


    @property
    def rw_signal_state(self):
        return self._rw_signal_state

    @property
    def _clicks_per_radian(self):
        return self._clicks_per_rotation / (2 * torch.pi)

    def _reset(self) -> None:
        """
        Resets the encoder with the given simulation time in nanoseconds.
        
        Args:
            current_sim_nanos (int): Current simulation time in nanoseconds
        
        
        """
        self._rw_signal_state.fill_(0)
        self._remaining_clicks.fill_(0)
        
    

    def encode(self, wheelSpeeds: torch.Tensor) -> torch.Tensor:

        # At the beginning of the simulation, the encoder outputs the true RW speeds
        if self._timer.steps == 0:
            
            return wheelSpeeds
        else:
            new_output = torch.zeros_like(wheelSpeeds)
            
            ## check if all state is modeled
            assert torch.isin(self._rw_signal_state, torch.tensor([SIGNAL_NOMINAL, SIGNAL_OFF, SIGNAL_STUCK])).all(), "encoder: un-modeled encoder signal mode selected."

            ## SIGNAL_NOMINAL_SITUATION
            
            signal_nominal_mask = (self._rw_signal_state == SIGNAL_NOMINAL)
            if torch.any(signal_nominal_mask):
                angle = (wheelSpeeds * self._timer.dt)
                numberClicks = torch.trunc(angle * self._clicks_per_radian + self._remaining_clicks)

                self._remaining_clicks[signal_nominal_mask] = angle * self._clicks_per_radian + self._remaining_clicks[signal_nominal_mask] - numberClicks

                new_output = torch.where(signal_nominal_mask, numberClicks / (self._clicks_per_radian * self._timer.dt), new_output)

            ## SIGNAL_OFF_SITUATION
            signal_off_mask = (self._rw_signal_state == SIGNAL_OFF)
            if torch.any(signal_off_mask):
                self._remaining_clicks[signal_off_mask] = 0.0
                new_output = torch.where(signal_off_mask, 0. , new_output).requires_grad_()

            ## SIGNAL_STUCK_SITUATION
            
            signal_stuck_mask = self._rw_signal_state == SIGNAL_STUCK
            if torch.any(signal_stuck_mask):
                new_output = torch.where(signal_stuck_mask, self._converted, new_output)
        
        return new_output
    
    def _forward(self, *args, wheelSpeeds: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        
        """
        
        results = self.encode(
            wheelSpeeds = wheelSpeeds
        )
        return results

    


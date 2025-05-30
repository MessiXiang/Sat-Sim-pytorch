from functools import partial

import pytest
import torch
import torch.nn as nn

from satsim.modules import Encoder
from satsim.architecture import (
    Timer,
    BSKError,
    logger,
    SIGNAL_OFF, 
    SIGNAL_NOMINAL, 
    SIGNAL_STUCK
)
        
def _run_one_step(
        input_tensor: torch.Tensor,
        encoder: Encoder,
        timer: Timer,
        true_tensor: torch.Tensor,
        call_interval:int,
        accuracy: float
):
    for i in range(call_interval):
        encoder.zero_grad()
        output = encoder(wheelSpeeds = input_tensor)
        timer.step()
    assert torch.allclose(output, true_tensor, atol = accuracy)
    try:
        torch.sum(output).backward(retain_graph=True)
    except RuntimeError:
        raise BSKError("This module is not differentiable")

    
    


@pytest.mark.parametrize("accuracy", [1e-8])
@pytest.mark.parametrize("dt", [0.2, 0.5, 1.])
def test_encoder(accuracy, dt):

    timer = Timer(dt)
    call_interval  = int(1. / dt)
    unit_test_encoder = Encoder(
        timer=timer,
        call_interval=call_interval,
        numRW=3,
        clicksPerRotation=2
    )
    run_one_step = partial(_run_one_step, accuracy = accuracy, call_interval=call_interval)
    unit_test_encoder.reset()

    trueWheelSpeedsEncoded = torch.tensor([[100., 200., 300.],
                                       [ 97.38937226, 197.92033718, 298.45130209],
                                       [100.53096491, 201.06192983, 298.45130209],
                                       [0., 0., 0.],
                                       [499.51323192, 398.98226701, 298.45130209],
                                       [499.51323192, 398.98226701, 298.45130209]])

    input_tensor = torch.tensor(
            [100, 200, 300], 
            dtype = torch.float32,
            requires_grad=True
        )

    run_one_step(
        input_tensor,
        unit_test_encoder,
        timer,
        trueWheelSpeedsEncoded[0]
    )

    
    run_one_step(
        input_tensor,
        unit_test_encoder,
        timer,
        trueWheelSpeedsEncoded[1]
    )

    
    run_one_step(
        input_tensor,
        unit_test_encoder,
        timer,
        trueWheelSpeedsEncoded[2]
    )

    unit_test_encoder.rw_signal_state.fill_(SIGNAL_OFF)

    run_one_step(
        input_tensor,
        unit_test_encoder,
        timer,
        trueWheelSpeedsEncoded[3]
    )

    unit_test_encoder.rw_signal_state.fill_(SIGNAL_NOMINAL)
    input_tensor = torch.tensor(
        [500, 400, 300], 
        dtype = torch.float32,
        requires_grad=True
    )

    run_one_step(
        input_tensor,
        unit_test_encoder,
        timer,
        trueWheelSpeedsEncoded[4]
    )

    unit_test_encoder.rw_signal_state.fill_(SIGNAL_STUCK)
    input_tensor = torch.tensor(
        [100, 200, 300], 
        dtype = torch.float32,
        requires_grad=True
    )

    run_one_step(
        input_tensor,
        unit_test_encoder,
        timer,
        trueWheelSpeedsEncoded[5]
    )
    print(list(unit_test_encoder.parameters()))
    print(unit_test_encoder.state_dict())

if __name__ == "__main__":
    test_encoder(
        1e-8,    # accuracy
        1.
    )
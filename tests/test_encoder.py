from functools import partial

import pytest
import torch

from satsim.architecture import Timer
from satsim.simulation.deviceInterface.encoder import (
    Encoder,
    EncoderState,
    EncoderSignal,
)


def _run_one_step(
    state: EncoderState,
    input_tensor: torch.Tensor,
    encoder: Encoder,
    timer: Timer,
    true_tensor: torch.Tensor,
    accuracy: float,
) -> EncoderState:
    new_state, (output, ) = encoder(state, wheel_speeds=input_tensor)
    timer.step()
    assert torch.allclose(output, true_tensor, atol=accuracy)
    return new_state


@pytest.fixture()
def timer() -> Timer:
    timer = Timer(1.)
    timer.reset()
    return timer


@pytest.fixture()
def encoder(timer: Timer) -> Encoder:
    encoder = Encoder(timer=timer,
                      num_reaction_wheels=3,
                      clicks_per_rotation=2)
    return encoder


@pytest.fixture()
def state(encoder: Encoder) -> EncoderState:
    return encoder.reset()


@pytest.mark.parametrize("accuracy", [1e-8])
def test_encoder(accuracy: float, timer: Timer, encoder,
                 state: EncoderState) -> None:
    encoder = Encoder(timer=timer,
                      num_reaction_wheels=3,
                      clicks_per_rotation=2)
    run_one_step = partial(
        _run_one_step,
        accuracy=accuracy,
        encoder=encoder,
        timer=timer,
    )

    trueWheelSpeedsEncoded = torch.tensor(
        [[100., 200., 300.], [97.38937226, 197.92033718, 298.45130209],
         [100.53096491, 201.06192983, 298.45130209], [0., 0., 0.],
         [499.51323192, 398.98226701, 298.45130209],
         [499.51323192, 398.98226701, 298.45130209]])

    input_tensor = torch.tensor([100, 200, 300],
                                dtype=torch.float32,
                                requires_grad=True)

    state = run_one_step(state=state,
                         input_tensor=input_tensor,
                         true_tensor=trueWheelSpeedsEncoded[0])

    state = run_one_step(state=state,
                         input_tensor=input_tensor,
                         true_tensor=trueWheelSpeedsEncoded[1])

    state = run_one_step(state=state,
                         input_tensor=input_tensor,
                         true_tensor=trueWheelSpeedsEncoded[2])

    state['reaction_wheels_signal_state'].fill_(EncoderSignal.OFF)

    state = run_one_step(state=state,
                         input_tensor=input_tensor,
                         true_tensor=trueWheelSpeedsEncoded[3])

    state['reaction_wheels_signal_state'].fill_(EncoderSignal.NOMINAL)
    input_tensor = torch.tensor([500, 400, 300],
                                dtype=torch.float32,
                                requires_grad=True)

    state = run_one_step(state=state,
                         input_tensor=input_tensor,
                         true_tensor=trueWheelSpeedsEncoded[4])

    state['reaction_wheels_signal_state'].fill_(EncoderSignal.STUCK)
    input_tensor = torch.tensor([100, 200, 300],
                                dtype=torch.float32,
                                requires_grad=True)

    state = run_one_step(state=state,
                         input_tensor=input_tensor,
                         true_tensor=trueWheelSpeedsEncoded[5])


def test_initialization(encoder: Encoder, state: EncoderState) -> None:
    assert encoder._num_rw == 3
    assert encoder._clicks_per_rotation == 2
    assert torch.equal(state['reaction_wheels_signal_state'],
                       torch.zeros(3, dtype=torch.int32))
    assert torch.equal(state['remaining_clicks'],
                       torch.zeros(3, dtype=torch.float32))
    assert torch.allclose(torch.tensor(encoder._clicks_per_radian),
                          torch.tensor(2 / (2 * torch.pi)))


def test_reset(encoder: Encoder, state: EncoderState) -> None:
    state['reaction_wheels_signal_state'].fill_(1)
    state['remaining_clicks'].fill_(0.5)
    state = encoder.reset()
    assert torch.equal(state['reaction_wheels_signal_state'],
                       torch.zeros(3, dtype=torch.int32))
    assert torch.equal(state['remaining_clicks'],
                       torch.zeros(3, dtype=torch.float32))


def test_first_step_behavior(encoder: Encoder, state: EncoderState) -> None:
    wheel_speeds = torch.tensor([1.0, 2.0, 3.0])
    state, (result, ) = encoder(state, wheel_speeds=wheel_speeds)
    assert torch.allclose(result, wheel_speeds)


def test_zero_dt_behavior(timer: Timer) -> None:
    with pytest.raises(AssertionError):
        timer.step()
        timer.dt = 0


def test_invalid_signal_state(encoder: Encoder, timer: Timer,
                              state: EncoderState) -> None:
    timer.step()
    state['reaction_wheels_signal_state'][0] = 5  # unvalid signal state

    with pytest.raises(AssertionError, match="un-modeled encoder signal mode"):
        encoder(state, wheel_speeds=torch.tensor([1.0, 2.0, 3.0]))


def test_requires_grad_propagation(encoder: Encoder, timer: Timer,
                                   state: EncoderState) -> None:
    timer.step()
    wheel_speeds = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    state['reaction_wheels_signal_state'].fill_(EncoderSignal.NOMINAL)
    result: torch.Tensor
    state, (result, ) = encoder(state, wheel_speeds=wheel_speeds)
    result.sum().backward()
    print(wheel_speeds.grad)
    assert wheel_speeds.grad is not None


def test_state_dict(encoder: Encoder, state: EncoderState) -> None:
    assert len(list(encoder.parameters())) == 0
    state_dict = encoder.reset()
    assert 'reaction_wheels_signal_state' in state_dict and 'remaining_clicks' in state_dict and 'last_output' in state_dict


if __name__ == "__main__":
    raise RuntimeError("This test does not support direct run")

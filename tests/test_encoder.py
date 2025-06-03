from functools import partial

import pytest
import torch

from satsim.simulation import Encoder
from satsim.architecture import (Timer, SIGNAL_OFF, SIGNAL_NOMINAL,
                                 SIGNAL_STUCK)


def _run_one_step(input_tensor: torch.Tensor,
                  encoder: Encoder,
                  timer: Timer,
                  true_tensor: torch.Tensor,
                  accuracy: float,
                  check_grad=True) -> None:
    output = encoder(wheel_speeds=input_tensor)
    timer.step()
    assert torch.allclose(output, true_tensor, atol=accuracy)


@pytest.fixture()
def timer() -> Timer:
    timer = Timer(1.)
    timer.reset()
    return timer


@pytest.fixture()
def encoder(timer) -> Encoder:
    encoder = Encoder(timer=timer, numRW=3, clicksPerRotation=2)
    return encoder


@pytest.mark.parametrize("accuracy", [1e-8])
def test_encoder(accuracy: float, timer: Timer, encoder) -> None:
    encoder = Encoder(timer=timer, numRW=3, clicksPerRotation=2)
    run_one_step = partial(
        _run_one_step,
        accuracy=accuracy,
        encoder=encoder,
        timer=timer,
    )
    encoder.reset()

    trueWheelSpeedsEncoded = torch.tensor(
        [[100., 200., 300.], [97.38937226, 197.92033718, 298.45130209],
         [100.53096491, 201.06192983, 298.45130209], [0., 0., 0.],
         [499.51323192, 398.98226701, 298.45130209],
         [499.51323192, 398.98226701, 298.45130209]])

    input_tensor = torch.tensor([100, 200, 300],
                                dtype=torch.float32,
                                requires_grad=True)

    run_one_step(input_tensor=input_tensor,
                 true_tensor=trueWheelSpeedsEncoded[0])

    run_one_step(input_tensor=input_tensor,
                 true_tensor=trueWheelSpeedsEncoded[1])

    run_one_step(input_tensor=input_tensor,
                 true_tensor=trueWheelSpeedsEncoded[2])

    encoder.rw_signal_state.fill_(SIGNAL_OFF)

    run_one_step(input_tensor=input_tensor,
                 true_tensor=trueWheelSpeedsEncoded[3])

    encoder.rw_signal_state.fill_(SIGNAL_NOMINAL)
    input_tensor = torch.tensor([500, 400, 300],
                                dtype=torch.float32,
                                requires_grad=True)

    run_one_step(input_tensor=input_tensor,
                 true_tensor=trueWheelSpeedsEncoded[4])

    encoder.rw_signal_state.fill_(SIGNAL_STUCK)
    input_tensor = torch.tensor([100, 200, 300],
                                dtype=torch.float32,
                                requires_grad=True)

    run_one_step(input_tensor=input_tensor,
                 true_tensor=trueWheelSpeedsEncoded[5])


def test_initialization(encoder: Encoder) -> None:
    assert encoder._num_rw == 3
    assert encoder._clicks_per_rotation == 2
    assert torch.equal(encoder.rw_signal_state,
                       torch.zeros(3, dtype=torch.int32))
    assert torch.equal(encoder.remaining_clicks,
                       torch.zeros(3, dtype=torch.float32))
    assert torch.allclose(torch.tensor(encoder._clicks_per_radian),
                          torch.tensor(2 / (2 * torch.pi)))


def test_reset(encoder: Encoder) -> None:
    encoder.rw_signal_state.fill_(1)
    encoder.remaining_clicks.fill_(0.5)
    encoder.reset()
    assert torch.equal(encoder.rw_signal_state,
                       torch.zeros(3, dtype=torch.int32))
    assert torch.equal(encoder.remaining_clicks,
                       torch.zeros(3, dtype=torch.float32))


def test_first_step_behavior(encoder: Encoder) -> None:
    wheel_speeds = torch.tensor([1.0, 2.0, 3.0])
    result = encoder(wheel_speeds)
    assert torch.allclose(result, wheel_speeds)


def test_zero_dt_behavior() -> None:
    with pytest.raises(AssertionError):
        timer = Timer(dt=1)
        timer.step()
        timer.dt = 0


def test_invalid_signal_state(encoder: Encoder, timer: Timer) -> None:
    timer.step()
    encoder.rw_signal_state[0] = 5  # 无效状态

    with pytest.raises(AssertionError, match="un-modeled encoder signal mode"):
        encoder(torch.tensor([1.0]))


def test_requires_grad_propagation(encoder: Encoder, timer: Timer) -> None:
    timer.step()
    wheel_speeds = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    encoder.rw_signal_state.fill_(SIGNAL_NOMINAL)
    result: torch.Tensor = encoder(wheel_speeds)
    result.sum().backward()
    print(wheel_speeds.grad)
    assert wheel_speeds.grad is not None


def test_state_dict(encoder: Encoder) -> None:
    assert len(list(encoder.parameters())) == 0
    state_dict = encoder.state_dict()
    assert "_rw_signal_state" in state_dict and "_remaining_clicks" in state_dict and "_converted" in state_dict


if __name__ == "__main__":
    test_encoder(
        1e-8,  # accuracy
        1.)

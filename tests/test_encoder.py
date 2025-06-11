import pytest
import todd
import torch

from satsim import Timer
from satsim.simulation import (WheelSpeedEncoder, WheelSpeedEncoderSignal,
                               WheelSpeedEncoderStateDict)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_encoder(device: str) -> None:
    timer = Timer(1.)
    timer.reset()
    encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)
    state: WheelSpeedEncoderStateDict = encoder.reset()

    tensor_mover = todd.utils.NestedCollectionUtils()
    state = tensor_mover.map(lambda x: x.to(device), state)

    output: torch.Tensor

    input_tensor = torch.tensor(
        [100, 200, 300],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    true_tensor = torch.tensor([100., 200., 300.])
    state, (output, ) = encoder(state, target_speeds=input_tensor)
    timer.step()
    assert torch.allclose(output.cpu(), true_tensor)

    input_tensor = torch.tensor(
        [100, 200, 300],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    true_tensor = torch.tensor([97.38937226, 197.92033718, 298.45130209])
    state, (output, ) = encoder(state, target_speeds=input_tensor)
    timer.step()
    assert torch.allclose(output.cpu(), true_tensor)

    input_tensor = torch.tensor(
        [100, 200, 300],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    true_tensor = torch.tensor([100.53096491, 201.06192983, 298.45130209])
    state, (output, ) = encoder(state, target_speeds=input_tensor)
    timer.step()
    assert torch.allclose(output.cpu(), true_tensor)

    input_tensor = torch.tensor(
        [100, 200, 300],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    true_tensor = torch.tensor([0., 0., 0.], dtype=torch.float32)
    state, (output, ) = encoder(
        state,
        target_speeds=input_tensor,
        signals=torch.full(
            (3, ),
            WheelSpeedEncoderSignal.STOPPED,
            device=device,
        ),
    )
    timer.step()
    assert torch.allclose(output.cpu(), true_tensor)

    input_tensor = torch.tensor(
        [500, 400, 300],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    true_tensor = torch.tensor([499.51323192, 398.98226701, 298.45130209])
    state, (output, ) = encoder(
        state,
        target_speeds=input_tensor,
        signals=torch.full(
            (3, ),
            WheelSpeedEncoderSignal.NOMINAL,
            device=device,
        ),
    )
    timer.step()
    assert torch.allclose(output.cpu(), true_tensor)

    input_tensor = torch.tensor(
        [100, 200, 300],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    true_tensor = torch.tensor([499.51323192, 398.98226701, 298.45130209])
    state, (output, ) = encoder(
        state,
        target_speeds=input_tensor,
        signals=torch.full(
            (3, ),
            WheelSpeedEncoderSignal.LOCKED,
            device=device,
        ),
    )
    timer.step()
    assert torch.allclose(output.cpu(), true_tensor)


def test_initialization() -> None:
    timer = Timer(1.)
    timer.reset()
    encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)
    state = encoder.reset()

    assert encoder._n == 3
    assert encoder._num_clicks == 2
    assert torch.equal(state['signals'], torch.ones(3, dtype=torch.int32))
    assert torch.equal(state['remaining_clicks'],
                       torch.zeros(3, dtype=torch.float32))
    assert torch.allclose(torch.tensor(encoder._clicks_per_radian),
                          torch.tensor(2 / (2 * torch.pi)))


def test_reset() -> None:
    timer = Timer(1.)
    timer.reset()
    encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)
    state = encoder.reset()

    state['signals'].fill_(1)
    state['remaining_clicks'].fill_(0.5)
    state = encoder.reset()
    assert torch.equal(state['signals'], torch.ones(3, dtype=torch.int32))
    assert torch.equal(state['remaining_clicks'],
                       torch.zeros(3, dtype=torch.float32))


def test_first_step_behavior() -> None:
    timer = Timer(1.)
    timer.reset()
    encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)
    state = encoder.reset()

    target_speeds = torch.tensor([1.0, 2.0, 3.0])
    state, (result, ) = encoder(state, target_speeds=target_speeds)
    assert torch.allclose(result, target_speeds)


def test_invalid_signal_state() -> None:
    timer = Timer(1.)
    timer.reset()
    encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)
    state = encoder.reset()

    timer.step()
    with pytest.raises(AssertionError, match="un-modeled encoder signal mode"):
        encoder(
            state,
            target_speeds=torch.tensor([1.0, 2.0, 3.0]),
            signals=torch.full((3, ), 5),
        )


def test_nominal_grad_propagation() -> None:
    timer = Timer(1.)
    timer.reset()
    encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)
    state = encoder.reset()

    timer.step()
    target_speeds = torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    result: torch.Tensor
    state, (result, ) = encoder(state, target_speeds=target_speeds)
    result.sum().backward()
    assert torch.allclose(torch.ones_like(target_speeds), target_speeds.grad)


def test_stopped_grad_propagation() -> None:
    timer = Timer(1.)
    timer.reset()
    encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)
    state = encoder.reset()

    timer.step()
    target_speeds = torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    result: torch.Tensor
    state, (result, ) = encoder(
        state,
        target_speeds=target_speeds,
        signals=torch.full(
            [3],
            WheelSpeedEncoderSignal.STOPPED,
        ),
    )
    result.sum().backward()
    assert torch.allclose(target_speeds.grad, torch.zeros_like(target_speeds))


def test_locked_grad_propagation() -> None:
    timer = Timer(1.)
    timer.reset()
    encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)
    state = encoder.reset()
    assert state['speeds'].requires_grad == False

    timer.step()
    first_target_speeds = torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    state: WheelSpeedEncoderStateDict
    result: torch.Tensor
    state, (result, ) = encoder(state, target_speeds=first_target_speeds)
    assert state['speeds'].requires_grad == True

    second_target_speeds = torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )
    timer.step()
    state, (result, ) = encoder(
        state,
        target_speeds=first_target_speeds,
        signals=torch.full((3, ), WheelSpeedEncoderSignal.LOCKED),
    )
    result.sum().backward()
    assert torch.allclose(
        torch.ones_like(first_target_speeds),
        first_target_speeds.grad,
    )
    assert second_target_speeds.grad is None


def test_state_dict() -> None:
    timer = Timer(1.)
    timer.reset()
    encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)
    state = encoder.reset()

    assert len(list(encoder.parameters())) == 0
    assert 'signals' in state and 'remaining_clicks' in state and 'speeds' in state


if __name__ == "__main__":
    raise RuntimeError("This test does not support direct run")

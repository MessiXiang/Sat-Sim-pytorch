from typing import Any, Callable
import pytest
import todd
import torch

from satsim import Timer
from satsim.simulation import (
    WheelSpeedEncoder,
    WheelSpeedEncoderSignal,
    WheelSpeedEncoderStateDict,
)

EncoderOperator = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        float,
    ],
    tuple[torch.Tensor, torch.Tensor],
]


class TestEncoderOperator:

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('implementation', ['py_', 'c'])
    def test_nominal_multiple_forward(self, device: str, implementation: str):
        operator: EncoderOperator = getattr(torch.ops.encoder, implementation)

        remaining_clicks = torch.zeros(3).to(device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.NOMINAL,
        ).to(device).int()
        speeds = torch.zeros(3).to(device)

        target_speeds = torch.tensor([100, 200, 300]).float().to(device)
        true_speeds = torch.tensor([97.38937226, 197.92033718, 298.45130209])
        speeds, remaining_clicks = operator(
            target_speeds,
            remaining_clicks,
            signals,
            speeds,
            2. / (2 * torch.pi),
            1.,
        )
        assert torch.allclose(speeds.cpu(), true_speeds)

        target_speeds = torch.tensor([100, 200, 300]).float().to(device)
        true_speeds = torch.tensor([100.53096491, 201.06192983, 298.45130209])
        speeds, remaining_clicks = operator(
            target_speeds,
            remaining_clicks,
            signals,
            speeds,
            2. / (2 * torch.pi),
            1.,
        )
        assert torch.allclose(speeds.cpu(), true_speeds)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('implementation', ['py_', 'c'])
    def test_stopped_forward(self, device: str, implementation: str):
        operator: EncoderOperator = getattr(torch.ops.encoder, implementation)

        remaining_clicks = torch.rand(3).to(device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.STOPPED,
        ).to(device).int()
        speeds = torch.rand(3).to(device)

        target_speeds = torch.rand(3).to(device)
        true_speeds = torch.tensor([0., 0., 0.], dtype=torch.float32)
        speeds, remaining_clicks = operator(
            target_speeds,
            remaining_clicks,
            signals,
            speeds,
            2. / (2 * torch.pi),
            1.,
        )
        assert torch.allclose(speeds.cpu(), true_speeds)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('implementation', ['py_', 'c'])
    def test_stopped_forward(self, device: str, implementation: str):
        operator: EncoderOperator = getattr(torch.ops.encoder, implementation)

        remaining_clicks = torch.rand(3).to(device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.LOCKED,
        ).to(device).int()
        speeds = torch.rand(3).to(device)

        target_speeds = torch.rand(3).to(device)
        true_speeds = speeds
        speeds, remaining_clicks = operator(
            target_speeds,
            remaining_clicks,
            signals,
            speeds,
            2. / (2 * torch.pi),
            1.,
        )
        assert torch.allclose(speeds, true_speeds)


class TestEncoderInitialization:

    def test_initialization(self) -> None:
        timer = Timer(1.)
        timer.reset()
        encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)

        assert encoder._n == 3
        assert encoder._num_clicks == 2
        assert torch.allclose(torch.tensor(encoder._clicks_per_radian),
                              torch.tensor(2 / (2 * torch.pi)))

    def test_reset(self) -> None:
        timer = Timer(1.)
        timer.reset()
        encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)
        state = encoder.reset()

        assert len(list(encoder.parameters())) == 0
        assert 'signals' in state and 'remaining_clicks' in state and 'speeds' in state
        assert torch.equal(state['signals'], torch.ones(3, dtype=torch.int32))
        assert torch.equal(state['remaining_clicks'], torch.zeros(3))
        assert torch.equal(state['speeds'], torch.zeros(3))


class TestEncoderOperatorBackPropagation:

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('implementation', ['py_', 'c'])
    def test_nominal_back_propagation(
        self,
        device: str,
        implementation: str,
    ) -> None:
        operator: EncoderOperator = getattr(torch.ops.encoder, implementation)

        remaining_clicks = torch.zeros(3).to(device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.NOMINAL,
        ).to(device).int()
        speeds = torch.zeros(3).to(device)

        target_speeds = torch.rand(
            3,
            requires_grad=True,
            device=device,
        )
        speeds, remaining_clicks = operator(
            target_speeds,
            remaining_clicks,
            signals,
            speeds,
            2. / (2 * torch.pi),
            1.,
        )

        speeds.sum().backward()
        assert torch.allclose(
            torch.ones_like(target_speeds),
            target_speeds.grad,
        )

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('implementation', ['py_', 'c'])
    def test_stopped_grad_propagation(
        self,
        device: str,
        implementation: str,
    ) -> None:
        operator: EncoderOperator = getattr(torch.ops.encoder, implementation)

        remaining_clicks = torch.zeros(3).to(device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.STOPPED,
        ).to(device).int()
        speeds = torch.zeros(3).to(device)

        target_speeds = torch.rand(
            3,
            requires_grad=True,
            device=device,
        )
        speeds, remaining_clicks = operator(
            target_speeds,
            remaining_clicks,
            signals,
            speeds,
            2. / (2 * torch.pi),
            1.,
        )
        speeds.sum().backward()
        assert torch.allclose(target_speeds.grad,
                              torch.zeros_like(target_speeds))

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('implementation', ['py_', 'c'])
    def test_locked_grad_propagation(
        self,
        device: str,
        implementation: str,
    ) -> None:
        operator: EncoderOperator = getattr(torch.ops.encoder, implementation)

        remaining_clicks = torch.zeros(3).to(device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.NOMINAL,
        ).to(device).int()
        speeds = torch.zeros(3).to(device)
        assert speeds.requires_grad == False

        first_target_speeds = torch.rand(
            3,
            requires_grad=True,
            device=device,
        )
        speeds, remaining_clicks = operator(
            first_target_speeds,
            remaining_clicks,
            signals,
            speeds,
            2. / (2 * torch.pi),
            1.,
        )
        assert speeds.requires_grad == True

        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.LOCKED,
        ).to(device).int()

        second_target_speeds = torch.rand(
            3,
            requires_grad=True,
            device=device,
        )
        speeds, remaining_clicks = operator(
            first_target_speeds,
            remaining_clicks,
            signals,
            speeds,
            2. / (2 * torch.pi),
            1.,
        )
        speeds.sum().backward()
        assert torch.allclose(
            torch.ones_like(first_target_speeds),
            first_target_speeds.grad,
        )
        assert second_target_speeds.grad is None


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


if __name__ == "__main__":
    raise RuntimeError("This test does not support direct run")

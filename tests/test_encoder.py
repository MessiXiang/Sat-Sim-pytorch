from typing import Callable

import pytest
import torch

from satsim.architecture import Timer
from satsim.simulation.wheel_speed_encoder import WheelSpeedEncoder, WheelSpeedEncoderSignal


def get_operator(
    implementation: str,
) -> Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        float,
    ],
        tuple[torch.Tensor, torch.Tensor],
]:
    return getattr(torch.ops.wheel_speed_encoder, implementation)


class TestWheelSpeedEncoderOperator:
    dt: float = 1.
    num_clicks: int = 2
    clicks_per_radian = num_clicks / (2 * torch.pi)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('implementation', ['py_', 'c'])
    def test_nominal_forward(self, device: str, implementation: str):
        operator = get_operator(implementation)

        remaining_clicks = torch.zeros(3, device=device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.NOMINAL,
            device=device,
            dtype=torch.int,
        )
        speeds = torch.zeros(3, device=device)

        target_speeds = torch.tensor([100., 200., 300.], device=device)
        true_speeds = torch.tensor(
            [97.38937226, 197.92033718, 298.45130209],
            device=device,
        )
        speeds, remaining_clicks = operator(
            target_speeds,
            remaining_clicks,
            signals,
            speeds,
            self.clicks_per_radian,
            self.dt,
        )
        assert torch.allclose(speeds, true_speeds)

        target_speeds = torch.tensor([100., 200., 300.], device=device)
        true_speeds = torch.tensor(
            [100.53096491, 201.06192983, 298.45130209],
            device=device,
        )
        speeds, remaining_clicks = operator(
            target_speeds,
            remaining_clicks,
            signals,
            speeds,
            self.clicks_per_radian,
            self.dt,
        )
        assert torch.allclose(speeds, true_speeds)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('implementation', ['py_', 'c'])
    def test_stopped_forward(self, device: str, implementation: str):
        operator = get_operator(implementation)

        remaining_clicks = torch.rand(3, device=device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.STOPPED,
            dtype=torch.int,
            device=device,
        )
        speeds = torch.rand(3, device=device)

        target_speeds = torch.rand(3, device=device)
        true_speeds = torch.zeros(3, device=device)
        speeds, remaining_clicks = operator(
            target_speeds,
            remaining_clicks,
            signals,
            speeds,
            self.clicks_per_radian,
            self.dt,
        )
        assert torch.allclose(speeds, true_speeds)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('implementation', ['py_', 'c'])
    def test_locked_forward(self, device: str, implementation: str):
        operator = get_operator(implementation)

        remaining_clicks = torch.rand(3, device=device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.LOCKED,
            dtype=torch.int,
            device=device,
        )
        speeds = torch.rand(3, device=device)

        target_speeds = torch.rand(3, device=device)
        true_speeds = speeds
        speeds, remaining_clicks = operator(
            target_speeds,
            remaining_clicks,
            signals,
            speeds,
            self.clicks_per_radian,
            self.dt,
        )
        assert torch.allclose(speeds, true_speeds)

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize('implementation', ['py_', 'c'])
    def test_nominal_back_propagation(
        self,
        device: str,
        implementation: str,
    ) -> None:
        operator = get_operator(implementation)

        remaining_clicks = torch.rand(3, device=device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.NOMINAL,
            dtype=torch.int,
            device=device,
        )
        speeds = torch.rand(3, device=device)

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
            self.clicks_per_radian,
            self.dt,
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
        operator = get_operator(implementation)

        remaining_clicks = torch.rand(3, device=device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.STOPPED,
            dtype=torch.int,
            device=device,
        )
        speeds = torch.rand(3, device=device)

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
            self.clicks_per_radian,
            self.dt,
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
        operator = get_operator(implementation)

        remaining_clicks = torch.rand(3, device=device)
        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.NOMINAL,
            dtype=torch.int,
            device=device,
        )
        speeds = torch.rand(3, device=device)
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
            self.clicks_per_radian,
            self.dt,
        )
        assert speeds.requires_grad == True

        signals = torch.full(
            [3],
            WheelSpeedEncoderSignal.LOCKED,
            dtype=torch.int,
            device=device,
        )

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
            self.clicks_per_radian,
            self.dt,
        )
        speeds.sum().backward()
        assert torch.allclose(
            torch.ones_like(first_target_speeds),
            first_target_speeds.grad,
        )
        assert second_target_speeds.grad is None


class TestWheelSpeedEncoder:

    @pytest.fixture
    def encoder(self):
        timer = Timer(1.)
        timer.reset()
        encoder = WheelSpeedEncoder(timer=timer, n=3, num_clicks=2)
        return encoder

    def test_initialization(self, encoder: WheelSpeedEncoder) -> None:
        assert encoder._n == 3
        assert encoder._num_clicks == 2
        assert torch.allclose(torch.tensor(encoder._clicks_per_radian),
                              torch.tensor(2 / (2 * torch.pi)))

    def test_reset(self, encoder: WheelSpeedEncoder) -> None:
        state = encoder.reset()

        assert len(list(encoder.parameters())) == 0
        assert 'signals' in state and 'remaining_clicks' in state and 'speeds' in state
        assert torch.equal(state['signals'], torch.ones(3, dtype=torch.int32))
        assert torch.equal(state['remaining_clicks'], torch.zeros(3))
        assert torch.equal(state['speeds'], torch.zeros(3))

    def test_first_step_behavior(self, encoder: WheelSpeedEncoder) -> None:
        state = encoder.reset()

        target_speeds = torch.rand(3)
        state, (result, ) = encoder(state, target_speeds=target_speeds)
        assert torch.allclose(result, target_speeds)

    def test_invalid_signal_state(self, encoder: WheelSpeedEncoder) -> None:
        state = encoder.reset()
        encoder._timer.step()

        with pytest.raises(
                AssertionError,
                match="un-modeled encoder signal mode",
        ):
            encoder(
                state,
                target_speeds=torch.tensor([1.0, 2.0, 3.0]),
                signals=torch.full((3, ), 5),
            )


if __name__ == "__main__":
    raise RuntimeError("This test does not support direct run")

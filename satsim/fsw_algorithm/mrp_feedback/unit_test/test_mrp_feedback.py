import pytest
import torch
from satsim.architecture import Timer
from satsim.fsw_algorithm.mrp_feedback import (
    MRPFeedback,
    MRPFeedbackStateDict,
    softclamp,
)


@pytest.mark.parametrize("integral_gain", [0.01, -1])
@pytest.mark.parametrize("integral_limit", [0, 20])
@pytest.mark.parametrize("control_law_type", [0, 1])
@pytest.mark.parametrize("with_reaction_wheel", [True, False])
def test_mrp_feedback_with_reaction_wheel(
    integral_gain: float,
    integral_limit: float,
    control_law_type: int,
    with_reaction_wheel: bool,
) -> None:
    timer = Timer(0.5)
    timer.reset()
    mrp_feedback = MRPFeedback(
        timer=timer,
        k=0.15,
        ki=integral_gain,
        p=150.0,
        integral_limit=integral_limit,
        control_law_Type=control_law_type,
    )
    mrp_feedback_state_dict = mrp_feedback.reset()
    mrp_feedback_state_dict['known_torque_point_b_body'] = torch.ones(3)
    mrp_feedback_state_dict['inertia_spacecraft_point_b_body'] = torch.diag(
        torch.tensor([1000., 800., 800.]))

    Js = torch.full([4], 0.1) if with_reaction_wheel else None
    gsHat_B = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.577350269190, 0.577350269190, 0.577350269190],
    ]).t() if with_reaction_wheel else None

    sigma_BR = torch.tensor([0.3, -0.5, 0.7])
    omega_BR_B = torch.tensor([0.010, -0.020, 0.015])
    omega_RN_B = torch.tensor([-0.02, -0.01, 0.005])
    domega_RN_B = torch.tensor([0.0002, 0.0003, 0.0001])
    wheel_speeds = torch.tensor([10.0, 25.0, 50.0, 100.0])

    attitude_command_torque_true = find_true_torques(
        mrp_feedback,
        mrp_feedback_state_dict,
        sigma_BR,
        omega_BR_B,
        omega_RN_B,
        domega_RN_B,
        control_law_type,
        wheel_speeds,
        gsHat_B,
        Js,
    )

    for true_value in attitude_command_torque_true:
        mrp_feedback_state_dict, (simulated_value, _) = mrp_feedback(
            mrp_feedback_state_dict,
            sigma_BR=sigma_BR,
            omega_BR_B=omega_BR_B,
            omega_RN_B=omega_RN_B,
            domega_RN_B=domega_RN_B,
            wheel_speeds=wheel_speeds,
            gsHat_B=gsHat_B,
            Js=Js,
        )
        timer.step()
        assert torch.allclose(
            true_value,
            simulated_value,
        )


@pytest.mark.parametrize("integral_gain", [0.01, -1])
@pytest.mark.parametrize("integral_limit", [0, 20])
@pytest.mark.parametrize("control_law_type", [0, 1])
def test_mrp_feedback_without_reaction_wheel(
    integral_gain: float,
    integral_limit: float,
    control_law_type: int,
) -> None:
    timer = Timer(0.5)
    timer.reset()
    mrp_feedback = MRPFeedback(
        timer=timer,
        k=0.15,
        ki=integral_gain,
        p=150.0,
        integral_limit=integral_limit,
        control_law_Type=control_law_type,
    )
    mrp_feedback_state_dict = mrp_feedback.reset()
    mrp_feedback_state_dict['known_torque_point_b_body'] = torch.ones(3)
    mrp_feedback_state_dict['inertia_spacecraft_point_b_body'] = torch.diag(
        torch.tensor([1000., 800., 800.]))

    Js = None
    gsHat_B = None

    sigma_BR = torch.tensor([0.3, -0.5, 0.7])
    omega_BR_B = torch.tensor([0.010, -0.020, 0.015])
    omega_RN_B = torch.tensor([-0.02, -0.01, 0.005])
    domega_RN_B = torch.tensor([0.0002, 0.0003, 0.0001])
    wheel_speeds = torch.tensor([10.0, 25.0, 50.0, 100.0])

    attitude_command_torque_true = find_true_torques(
        mrp_feedback,
        mrp_feedback_state_dict,
        sigma_BR,
        omega_BR_B,
        omega_RN_B,
        domega_RN_B,
        control_law_type,
        wheel_speeds,
        gsHat_B,
        Js,
    )

    for true_value in attitude_command_torque_true:
        mrp_feedback_state_dict, (simulated_value, _) = mrp_feedback(
            mrp_feedback_state_dict,
            sigma_BR=sigma_BR,
            omega_BR_B=omega_BR_B,
            omega_RN_B=omega_RN_B,
            domega_RN_B=domega_RN_B,
            wheel_speeds=wheel_speeds,
            gsHat_B=gsHat_B,
            Js=Js,
        )
        timer.step()
        assert torch.allclose(
            true_value,
            simulated_value,
        )


def find_true_torques(
    module: MRPFeedback,
    state_dict: MRPFeedbackStateDict,
    sigma_BR: torch.Tensor,
    omega_BR_B: torch.Tensor,
    omega_RN_B: torch.Tensor,
    domega_RN_B: torch.Tensor,
    control_law_type: int,
    wheel_speed: torch.Tensor,
    gsHat_B: torch.Tensor,
    Js: torch.Tensor,
):
    k = module.k
    p = module.p
    ki = module.ki
    inertia_spacecraft_point_b_body = state_dict[
        'inertia_spacecraft_point_b_body']
    known_torque_point_b_body = state_dict['known_torque_point_b_body']
    num_reaction_wheels = Js.size(-1) if Js else 0
    omega_BN_B = omega_BR_B + omega_RN_B

    sigma_integral = torch.zeros(3)
    steps = [0, 0, .5, .5, .5]

    for step in steps:
        if step == 0:
            sigma_integral = torch.zeros(3)

        if ki > 0:
            sigma_integral = k * step * sigma_BR + sigma_integral
            sigma_integral = softclamp(
                sigma_integral,
                max=module.integral_limit,
                min=-module.integral_limit,
            )
            z = sigma_integral + inertia_spacecraft_point_b_body @ omega_BR_B
        else:
            z = torch.zeros(3)

        attitude_command_torque = k * sigma_BR + p * omega_BR_B + p * ki * z

        gshs = torch.zeros(3)
        if num_reaction_wheels > 0:
            gshs = (Js * (omega_BN_B @ gsHat_B + wheel_speed) *
                    gsHat_B).sum(-1)

        if control_law_type == 0:
            attitude_command_torque = attitude_command_torque - torch.cross(
                omega_RN_B + ki * z,
                inertia_spacecraft_point_b_body @ omega_BN_B + gshs,
            )
        else:
            attitude_command_torque = attitude_command_torque - torch.cross(
                omega_BN_B,
                inertia_spacecraft_point_b_body @ omega_BN_B + gshs,
            )

        attitude_command_torque = attitude_command_torque + inertia_spacecraft_point_b_body @ (
            -domega_RN_B +
            omega_BN_B.cross(omega_RN_B)) + known_torque_point_b_body

        yield -attitude_command_torque

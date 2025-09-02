import pytest
import torch
import tqdm

from satsim.architecture import Timer, constants
from satsim.simulation.reaction_wheels import (
    HoneywellHR12Small,
    ReactionWheels,
    ReactionWheelsStateDict,
)
from satsim.simulation.spacecraft import (
    Spacecraft,
    SpacecraftStateDict,
    SpacecraftStateOutput,
)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_reaction_wheels_dynamic_without_torque(axis: int, ) -> None:
    timer = Timer(0.1)
    reaction_wheels = ReactionWheels(
        timer=timer,
        reaction_wheels=[
            HoneywellHR12Small.build(angular_velocity_init=0.)
            for _ in range(3)
        ],
    )
    moment_of_inertia = 2.
    angular_velocity_init = torch.zeros(1, 3)
    angular_velocity_init[0, axis] = 0.35
    spacecraft = Spacecraft(
        timer=timer,
        mass=torch.tensor([5.0]),
        moment_of_inertia_matrix_wrt_body_point=torch.diag(
            torch.ones(3) * moment_of_inertia +
            reaction_wheels.moment_of_inertia_wrt_spin.squeeze(-2)).unsqueeze(
                0),
        position=torch.zeros(1, 3),
        velocity=torch.zeros(1, 3),
        attitude=torch.zeros(1, 3),
        angular_velocity=angular_velocity_init,
        reaction_wheels=reaction_wheels,
    )

    timer.reset()
    spacecraft_state_dict = spacecraft.reset()

    reaction_wheels_speed_init = torch.zeros(1, 1, 3)
    reaction_wheels_speed_init[0, 0, axis] = 100. * constants.RPM
    spacecraft_state_dict['_reaction_wheels']['dynamic_params'][
        'angular_velocity'] = reaction_wheels_speed_init

    current_torque = torch.zeros(1, 1, 3)
    reaction_wheels_state_dict = spacecraft_state_dict['_reaction_wheels']
    reaction_wheels_state_dict['current_torque'] = current_torque

    assert torch.allclose(
        spacecraft_state_dict['_reaction_wheels']['dynamic_params']
        ['angular_velocity'], reaction_wheels_speed_init)
    assert torch.allclose(
        spacecraft_state_dict['_reaction_wheels']['current_torque'],
        current_torque)
    assert torch.allclose(
        spacecraft_state_dict['_hub']['dynamic_params']['angular_velocity'],
        angular_velocity_init)

    p_bar = tqdm.tqdm(total=5.)

    while timer.time < 5.:
        spacecraft_state_dict: SpacecraftStateDict
        spacecraft_output: SpacecraftStateOutput
        spacecraft_state_dict, spacecraft_output = spacecraft(
            spacecraft_state_dict)
        timer.step()

        assert torch.allclose(
            spacecraft_output.angular_velocity,
            angular_velocity_init,
        )

        angle = 4. * torch.arctan(spacecraft_output.attitude)

        assert torch.allclose(
            angle,
            torch.tensor([[0, 0, 0.35]]) * timer.time,
        )

        assert torch.allclose(
            spacecraft_state_dict['_reaction_wheels']['dynamic_params']
            ['angular_velocity'],
            reaction_wheels_speed_init,
        )
        p_bar.update(timer.dt)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_reaction_wheels_dynamic_with_torque(axis: int, ) -> None:
    timer = Timer(0.1)
    reaction_wheels = ReactionWheels(
        timer=timer,
        reaction_wheels=[
            HoneywellHR12Small.build(angular_velocity_init=0.)
            for _ in range(3)
        ],
    )
    moment_of_inertia = 2.
    reaction_wheels_moment_of_inertia = 12 / 6000 / constants.RPM
    angular_velocity_init = torch.zeros(1, 3)
    angular_velocity_init[0, axis] = 0.35
    spacecraft = Spacecraft(
        timer=timer,
        mass=torch.tensor([5.0]),
        moment_of_inertia_matrix_wrt_body_point=torch.diag(
            torch.ones(3) * moment_of_inertia +
            reaction_wheels.moment_of_inertia_wrt_spin.squeeze(-2)).unsqueeze(
                0),
        position=torch.zeros(1, 3),
        velocity=torch.zeros(1, 3),
        attitude=torch.zeros(1, 3),
        angular_velocity=angular_velocity_init,
        reaction_wheels=reaction_wheels,
    )

    timer.reset()
    spacecraft_state_dict = spacecraft.reset()

    reaction_wheels_speed_init = torch.zeros(1, 1, 3)
    reaction_wheels_speed_init[0, 0, axis] = 100. * constants.RPM
    spacecraft_state_dict['_reaction_wheels']['dynamic_params'][
        'angular_velocity'] = reaction_wheels_speed_init

    current_torque = torch.zeros(1, 1, 3)
    motor_torque = 0.2
    current_torque[0, 0, axis] = motor_torque
    reaction_wheels_state_dict = spacecraft_state_dict['_reaction_wheels']
    reaction_wheels_state_dict['current_torque'] = current_torque

    assert torch.allclose(
        spacecraft_state_dict['_reaction_wheels']['dynamic_params']
        ['angular_velocity'], reaction_wheels_speed_init)
    assert torch.allclose(
        spacecraft_state_dict['_reaction_wheels']['current_torque'],
        current_torque)
    assert torch.allclose(
        spacecraft_state_dict['_hub']['dynamic_params']['angular_velocity'],
        angular_velocity_init)

    p_bar = tqdm.tqdm(total=5.)

    while timer.time < 5.:
        spacecraft_state_dict: SpacecraftStateDict
        spacecraft_output: SpacecraftStateOutput
        spacecraft_state_dict, spacecraft_output = spacecraft(
            spacecraft_state_dict)
        timer.step()

        assert torch.allclose(
            spacecraft_output.angular_velocity,
            angular_velocity_init -
            current_torque.squeeze(1) / moment_of_inertia * timer.time,
            atol=1e-5,
        )

        angle = 4. * torch.arctan(spacecraft_output.attitude)
        assert torch.allclose(
            angle,
            angular_velocity_init * timer.time - 0.5 *
            current_torque.squeeze(1) / moment_of_inertia * timer.time**2,
            atol=1e-5,
        )

        assert torch.allclose(
            spacecraft_state_dict['_reaction_wheels']['dynamic_params']
            ['angular_velocity'],
            reaction_wheels_speed_init + current_torque * timer.time *
            (1. / moment_of_inertia + 1. / reaction_wheels_moment_of_inertia),
            atol=1e-5,
        )
        p_bar.update(timer.dt)


if __name__ == '__main__':
    test_reaction_wheels_dynamic_with_torque(axis=0)

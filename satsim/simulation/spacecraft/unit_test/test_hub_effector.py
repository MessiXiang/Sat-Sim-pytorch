import pytest
import torch

from satsim.architecture import Timer
from satsim.simulation.spacecraft import HubEffector, HubEffectorDynamicParams
from satsim.utils import Bmat, mrp_to_rotation_matrix


@pytest.fixture
def hub_effector():
    timer = Timer(1.)
    mass = torch.tensor([100.0])
    moi = torch.eye(3) * 1000.0
    pos = torch.tensor([1.0, 2.0, 3.0])
    velocity = torch.tensor([0.1, 0.2, 0.3])
    sigma = torch.tensor([0.1, 0.2, 0.3])
    omega = torch.tensor([0.01, 0.02, 0.03])

    return HubEffector(
        timer=timer,
        mass=mass,
        moment_of_inertia_matrix_wrt_body_point=moi,
        position=pos,
        velocity=velocity,
        attitude=sigma,
        angular_velocity=omega,
    )


def test_hub_effector_init(hub_effector: HubEffector):
    assert torch.allclose(
        hub_effector.mass,
        torch.tensor([100.0]),
    )
    assert torch.allclose(
        hub_effector.moment_of_inertia_matrix_wrt_body_point,
        torch.eye(3) * 1000.0,
    )
    assert torch.allclose(hub_effector._position_init,
                          torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(hub_effector._velocity_init,
                          torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(hub_effector._attitude_init,
                          torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(hub_effector._angular_velocity_init,
                          torch.tensor([0.01, 0.02, 0.03]))


def test_hub_effector_reset(hub_effector: HubEffector):
    state_dict = hub_effector.reset()

    assert 'dynamic_params' in state_dict
    assert 'mass_props' in state_dict
    dynamic_params = state_dict['dynamic_params']

    assert torch.allclose(dynamic_params['pos'], hub_effector._position_init)
    assert torch.allclose(dynamic_params['velocity'],
                          hub_effector._velocity_init)
    assert torch.allclose(dynamic_params['sigma'], hub_effector._attitude_init)
    assert torch.allclose(dynamic_params['omega'],
                          hub_effector._angular_velocity_init)
    assert torch.allclose(dynamic_params['grav_velocity'],
                          hub_effector._velocity_init)
    assert torch.allclose(dynamic_params['grav_velocity_bc'],
                          hub_effector._velocity_init)

    assert torch.allclose(state_dict['mass_props']['mass'],
                          torch.tensor([100.0]))
    assert torch.allclose(
        state_dict['mass_props']['moment_of_inertia_matrix_wrt_body_point'],
        torch.eye(3) * 1000.0)


def test_compute_derivatives(hub_effector: HubEffector):
    state_dict = hub_effector.reset()
    back_sub_matrices = {
        'vec_rot': torch.zeros(3),
        'vec_trans': torch.zeros(3),
        'matrix_a': torch.eye(3),
        'matrix_b': torch.zeros(3, 3),
        'matrix_c': torch.zeros(3, 3),
        'matrix_d': torch.eye(3)
    }
    g_N = torch.tensor([0.0, 0.0, -9.81])

    derivatives = hub_effector.compute_derivatives(
        state_dict=state_dict,
        integrate_time_step=0.1,
        rDDot_BN_N=None,
        omegaDot_BN_B=None,
        sigma_BN=None,
        gravity_acceleration=g_N,
        back_substitution_matrices=back_sub_matrices)

    assert torch.allclose(derivatives['pos'],
                          state_dict['dynamic_params']['velocity'])
    assert torch.allclose(derivatives['grav_velocity'], g_N)
    assert torch.allclose(derivatives['grav_velocity_bc'], g_N)

    # Test sigma_dot calculation
    expected_sigma_dot = 0.25 * torch.matmul(
        Bmat(state_dict['dynamic_params']['sigma']),
        state_dict['dynamic_params']['omega'].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(derivatives['sigma'], expected_sigma_dot)


def test_update_energy_momentum_contributions(hub_effector: HubEffector):
    state_dict = hub_effector.reset()
    rotAngMomPntCContr_B = torch.zeros(3)
    rotEnergyContr = torch.tensor(0.0)
    omega_BN_B = torch.tensor([0.01, 0.02, 0.03])

    hub_effector.moment_of_inertia_matrix_wrt_body_point = hub_effector.moment_of_inertia_matrix_wrt_body_point

    ang_mom, energy = hub_effector.update_energy_momentum_contributions(
        state_dict=state_dict,
        integrate_time_step=0.1,
        rotAngMomPntCContr_B=rotAngMomPntCContr_B,
        rotEnergyContr=rotEnergyContr,
        omega_BN_B=omega_BN_B)

    expected_ang_mom = torch.matmul(
        hub_effector.moment_of_inertia_matrix_wrt_body_point,
        omega_BN_B,
    )
    expected_energy = 0.5 * torch.dot(
        omega_BN_B,
        torch.matmul(
            hub_effector.moment_of_inertia_matrix_wrt_body_point,
            omega_BN_B.unsqueeze(-1),
        ).squeeze(-1),
    )

    assert torch.allclose(ang_mom, expected_ang_mom)
    assert torch.allclose(energy, expected_energy)


def test_modify_states(hub_effector: HubEffector):
    state_dict = hub_effector.reset()

    # Test with normal sigma (norm <= 1)
    modified_state = hub_effector.normalize_attitude(state_dict, 0.1)
    assert torch.allclose(modified_state['dynamic_params']['sigma'],
                          state_dict['dynamic_params']['sigma'])

    # Test with sigma norm > 1
    state_dict['dynamic_params']['sigma'] = torch.tensor([1.0, 1.0, 1.0])
    modified_state = hub_effector.normalize_attitude(state_dict, 0.1)
    expected_sigma = -torch.tensor([1.0, 1.0, 1.0]) / torch.tensor(
        [1.0, 1.0, 1.0]).norm()
    assert torch.allclose(modified_state['dynamic_params']['sigma'],
                          expected_sigma)


def test_match_gravity_to_velocity_state(hub_effector: HubEffector):
    state_dict = hub_effector.reset()
    v_CN_N = torch.tensor([0.2, 0.3, 0.4])

    updated_state = hub_effector.match_gravity_to_velocity_state(
        state_dict, v_CN_N)

    assert torch.allclose(
        updated_state['dynamic_params']['grav_velocity'],
        state_dict['dynamic_params']['velocity'],
    )
    assert torch.allclose(
        updated_state['dynamic_params']['grav_velocity_bc'],
        v_CN_N,
    )

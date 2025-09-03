import pytest
import torch
import tqdm

from satsim.architecture import Timer
from satsim.simulation.gravity import (Ephemeris, GravityField,
                                       PointMassGravityBody)
from satsim.simulation.spacecraft import Spacecraft
from satsim.utils import dict_recursive_apply


@pytest.mark.parametrize('device', ['cuda:0', 'cpu'])
def test_translation(device: str):
    timer = Timer(0.01)

    earth = PointMassGravityBody(
        timer=timer,
        name='EARTH',
        gm=0.3986004415E+15,
        equatorial_radius=0.,
        polar_radius=0.,
    )
    earth.set_central()
    gravity_field = GravityField(timer=timer, gravity_bodies=earth)

    spacecraft = Spacecraft(
        timer=timer,
        mass=torch.tensor([100.], dtype=torch.float64),
        position=torch.tensor(
            [-4020338.690396649, 7490566.741852513, 5248299.211589362],
            dtype=torch.float64,
        ),
        velocity=torch.tensor(
            [-5199.77710904224, -3436.681645356935, 1041.576797498721],
            dtype=torch.float64,
        ),
        gravity_field=gravity_field,
    )
    timer.reset()
    spacecraft_state_dict = spacecraft.reset()

    _move = lambda x: x.to(device=device, dtype=torch.float64)
    spacecraft_state_dict = dict_recursive_apply(spacecraft_state_dict, _move)
    spacecraft.to(device=device, dtype=torch.float64)

    assert torch.allclose(
        spacecraft_state_dict['_hub']['dynamic_params']['pos'],
        torch.tensor(
            [-4020338.690396649, 7490566.741852513, 5248299.211589362],
            device=device,
            dtype=torch.float64,
        ),
        atol=1e-3,
    )

    assert torch.allclose(
        spacecraft_state_dict['_hub']['dynamic_params']['velocity'],
        torch.tensor(
            [-5199.77710904224, -3436.681645356935, 1041.576797498721],
            device=device,
            dtype=torch.float64,
        ),
        atol=1e-3,
    )

    central_body_ephemeris = Ephemeris(
        position_in_inertial=torch.zeros(
            3,
            device=device,
            dtype=torch.float64,
        ),
        velocity_in_inertial=torch.zeros(
            3,
            device=device,
            dtype=torch.float64,
        ),
    )

    stop_time = 10.
    p_bar = tqdm.tqdm(total=int(stop_time / timer.dt) + 1)
    while timer.time < stop_time:
        spacecraft_state_dict, _ = spacecraft(
            state_dict=spacecraft_state_dict,
            central_body_ephemeris=central_body_ephemeris,
        )
        timer.step()
        p_bar.update(1)
    p_bar.close()

    assert torch.allclose(
        spacecraft_state_dict['_hub']['dynamic_params']['pos'],
        torch.tensor(
            [-4072255.7737936215, 7456050.4649078, 5258610.029627514],
            device=device,
            dtype=torch.float64,
        ))


@pytest.mark.parametrize('device', ['cuda:0', 'cpu'])
def test_translation_and_rotation(device: str):
    timer = Timer(0.001)

    earth = PointMassGravityBody(
        timer=timer,
        name='EARTH',
        gm=0.3986004415E+15,
        equatorial_radius=0.,
        polar_radius=0.,
    )
    earth.set_central()
    gravity_field = GravityField(timer=timer, gravity_bodies=earth)

    spacecraft = Spacecraft(
        timer=timer,
        mass=torch.tensor([100.], dtype=torch.float64),
        moment_of_inertia_matrix_wrt_body_point=torch.tensor([
            [500, 0.0, 0.0],
            [0.0, 200, 0.0],
            [0.0, 0.0, 300],
        ]),
        position=torch.tensor(
            [-4020338.690396649, 7490566.741852513, 5248299.211589362], ),
        velocity=torch.tensor(
            [-5199.77710904224, -3436.681645356935, 1041.576797498721], ),
        angular_velocity=torch.tensor([0.5, -0.4, 0.7]),
        gravity_field=gravity_field,
    )
    timer.reset()
    spacecraft_state_dict = spacecraft.reset()

    _move = lambda x: x.to(device=device, dtype=torch.float64)
    spacecraft_state_dict = dict_recursive_apply(spacecraft_state_dict, _move)
    spacecraft.to(device=device, dtype=torch.float64)

    stop_time = 10.
    p_bar = tqdm.tqdm(total=int(stop_time / timer.dt))
    sigmas = []
    while timer.time < stop_time:

        spacecraft_state_dict, _ = spacecraft(
            state_dict=spacecraft_state_dict, )
        sigmas.append(spacecraft_state_dict['_hub']['dynamic_params']['sigma'])
        timer.step()
        p_bar.update(1)
    p_bar.close()

    assert torch.allclose(
        spacecraft_state_dict['_hub']['dynamic_params']['pos'],
        torch.tensor(
            [-4072255.7737936215, 7456050.4649078, 5258610.029627514],
            device=device,
            dtype=torch.float64,
        ),
    )
    assert torch.allclose(
        spacecraft_state_dict['_hub']['dynamic_params']['sigma'],
        torch.tensor(
            [3.73034285e-01, -2.39564413e-03, 2.08570797e-01],
            device=device,
            dtype=torch.float64,
        ),
        atol=1e-3,
    )


if __name__ == "__main__":
    test_translation_and_rotation(device='cpu')

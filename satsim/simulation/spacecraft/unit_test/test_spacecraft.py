import pytest
import torch

from satsim.architecture import constants
from satsim.simulation.gravity import Ephemeris, PointMassGravityField
from satsim.simulation.spacecraft import HubEffector, Spacecraft


@pytest.fixture
def gravity_field():
    return PointMassGravityField(
        central_body_mu=constants.MU_EARTH,
        planet_bodies_mu=[constants.MU_SUN],
    )


def test_hub_effector_initialization():
    # Test default initialization
    hub_effector = HubEffector(
        mass=torch.tensor([1.0]),
        moment_of_inertia_matrix_wrt_body_point=torch.eye(3),
        pos=torch.zeros(3),
        velocity=torch.zeros(3),
    )

    assert hub_effector.mass.item() == 1.0
    assert torch.all(
        hub_effector.moment_of_inertia_matrix_wrt_body_point == torch.eye(3))

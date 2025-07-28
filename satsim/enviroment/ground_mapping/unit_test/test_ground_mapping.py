import math

import numpy as np
from typing import Any, Callable
import pytest
import torch

from satsim.architecture import Timer
from satsim.simulation.environment.groundMapping import (
    GroundMapping,

)

import pytest

@pytest.mark.parametrize("maxRange", [1e9, -1.0, 2.0])

def test_groundMapping(maxRange):
    r"""
    This test checks two points to determine if they are accessible for mapping or not. One point should be mapped,
    and one point should not be mapped.

    The inertial, planet-fixed planet-centered, and spacecraft body frames are all aligned.
    The spacecraft is in the -y direction of the inertial frame. The first point is along the line from the spacecraft
    to the origin. The second point is along the z-axis. The first point should be accessible because a.) the spacecraft
    is within the point's visibility cone and the point is within the spacecraft's visibility cone. The second point is
    not accessible because the spacecraft is not within the point's visibility cone and the point is not within the
    spacecraft's visibility cone.
    """
    state_dict, (accessDict,currentGroundState) = groundMappingTestFunction(maxRange)

    expected_hasaccess = torch.tensor([True, False])
    accessDict_hasaccess = torch.tensor(
        [accessDict[0]["has_Access"],
         accessDict[1]["has_Access"]
        ],dtype=torch.bool)
    assert torch.equal(accessDict_hasaccess, expected_hasaccess)
    return accessDict,currentGroundState



def groundMappingTestFunction(maxRange):
    """Test method"""
    testFailCount = 0
    testMessages = []
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"


    J20002Pfix = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    PositionVector = torch.tensor([0., 0., 0.])

    r_BN_N = torch.tensor([0., -1., 0.])
    sigma_BN = torch.tensor([0., 0., 0.])
    v_BN_N = torch.tensor([0., 0., 0.])

    # Create the initial imaging target
    groundMap = GroundMapping(maximum_Range=torch.tensor(maxRange))

    mapping_Points = [torch.tensor([0., -0.1, 0.]), torch.tensor([0., 0., math.tan(np.radians(22.5))+0.1])]

    groundMap.minimum_Elevation = torch.tensor(np.radians(45.))

    groundMap.camera_Pos_B = torch.tensor([0., 0., 0.])
    groundMap.nHat_B = torch.tensor([0., 1., 0.])
    groundMap.halfField_Of_View = torch.tensor(np.radians(22.5))


    state_dict, (accessDict,currentGroundState) = groundMap.forward(
        None,
        dcm_inertial_to_PlanetFix=J20002Pfix,
        planet_Position_in_inertial=PositionVector,
        r_BN_N=r_BN_N,
        v_BN_N=v_BN_N,
        sigma_BN=sigma_BN,
        mapping_Points=mapping_Points
    )

    return state_dict, (accessDict,currentGroundState)


if __name__ == "__main__":
    #accessDict,currentGroundState = test_groundMapping(1e9)
    #accessDict,currentGroundState = test_groundMapping(0.001)
    #accessDict,currentGroundState = test_groundMapping(1e-12)
    #accessDict,currentGroundState = test_groundMapping(2.0)
    #accessDict,currentGroundState = test_groundMapping(-1.0)
    raise RuntimeError("This test does not support direct run")

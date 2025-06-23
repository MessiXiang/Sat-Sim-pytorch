import pytest

from satsim.architecture import Timer
from satsim.simulation.reaction_wheels import (ReactionWheels,
                                               ReactionWheelModels,
                                               ReactionWheelStateEffector)

# def default_reaction_wheel_config() -> dict:
#     return dict(rWB_B=[[0.], [0.], [0.]],
#                 gsHat_B=[[1.], [0.], [0.]],
#                 w2Hat0_B=[[0.], [1.], [0.]],
#                 w3Hat0_B=[[0.], [0.], [1.]],
#                 RWModel=ReactionWheelModels.BALANCED_WHEELS)

# @pytest.fixture
# def reaction_wheel_state_effector() -> ReactionWheelStateEffector:
#     timer = Timer(1.)
#     return ReactionWheelStateEffector(timer=timer)

# def test_basic_behavior(
#         reaction_wheel_state_effector: ReactionWheelStateEffector):
#     state_dict = reaction_wheel_state_effector.reset()
#     reaction_wheels = state_dict['reaction_wheels']

#     num_reaction_wheel = 2
#     for i in range(num_reaction_wheel):
#         reaction_wheels.add_reaction_wheel(**default_reaction_wheel_config())

#     state_dict, (reaction_wheel_log, ) = reaction_wheel_state_effector(
#         state_dict, )

# def test_saturation_behacior(
#         reaction_wheel_state_effector: ReactionWheelStateEffector):
#     state_dict = reaction_wheel_state_effector.reset()
#     reaction_wheels = state_dict['reaction_wheels']

#     num_reaction_wheel = 3
#     for i in range(num_reaction_wheel):
#         reaction_wheels.add_reaction_wheel(**default_reaction_wheel_config())
